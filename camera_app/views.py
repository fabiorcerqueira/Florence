from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
import torch
from PIL import Image
import io
from transformers import BlipProcessor, BlipForQuestionAnswering
from lavis.models import load_model_and_preprocess
from django.views.decorators.csrf import csrf_exempt
import os, traceback, datetime, numpy as np, cv2
from threading import Lock
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Configuracao
from .form import ConfiguracaoForm
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.hashers import make_password

processing_lock = Lock()

DEBUG_FRAMES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frames_debug")
os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)

# ------------------ Views básicas ------------------

def home(request):
    return render(request, 'contas/home.html', {'usuario': request.user})


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)

            # 🔒 Ignora qualquer redirecionamento antigo
            return redirect('home')
        else:
            messages.error(request, 'Usuário ou senha inválidos.')
            return render(request, 'contas/login.html')

    # Se for GET (ou abriu a página)
    return render(request, 'contas/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')  

# ------------------ Câmera ------------------

@login_required
def camera_view(request):
    # busca ou cria configuração do usuário logado
    config, created = Configuracao.objects.get_or_create(user=request.user)

    # o campo já está em milissegundos, então não precisa multiplicar por 1000
    intervalo = config.intervalo if config.intervalo else 30000

    return render(request, 'contas/camera.html', {"intervalo": intervalo})


# ------------------ IA: BLIP + LAVIS ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

_model_itm = None
_vis_processors_itm = None
_text_processors_itm = None

def get_itm_model():
    global _model_itm, _vis_processors_itm, _text_processors_itm

    if _model_itm is not None:
        return _model_itm, _vis_processors_itm, _text_processors_itm

    try:
        print(f"🔄 Carregando BLIP ITM (device={device})")
        _model_itm, _vis_processors_itm, _text_processors_itm = load_model_and_preprocess(
            name="blip_image_text_matching",
            model_type="base",
            is_eval=True,
            device=device
        )
        print("✅ Modelo BLIP ITM carregado com sucesso.")
    except Exception as e:
        print("❌ Erro ao carregar ITM:", e)
        tb = traceback.format_exc()
        with open(os.path.join(DEBUG_FRAMES_DIR, "itm_error.log"), "a", encoding="utf-8") as f:
            f.write(f"\n\n[{datetime.datetime.now()}] Erro ao carregar BLIP ITM:\n{tb}\n")
        _model_itm = None
        _vis_processors_itm = {"eval": lambda x: x}
        _text_processors_itm = {"eval": lambda x: x}

    return _model_itm, _vis_processors_itm, _text_processors_itm

# ---------------- process_frame ----------------
@csrf_exempt
@login_required
def process_frame(request):
    print("📩 Requisição recebida em /process_frame")

    if request.method != "POST" or not request.FILES.get("frame"):
        return JsonResponse({"error": "Nenhum frame recebido"}, status=400)

    file = request.FILES["frame"]
    file_bytes = file.read()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # decodificação do frame
    img = None
    try:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_cv is not None:
            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
    except Exception as e:
        print("⚠️ Erro ao decodificar com OpenCV:", e)

    if img is None:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return JsonResponse({"error": "Falha ao decodificar frame"}, status=400)

    # --- VQA ---
    questions = [
        "Is there a person in the image?",
        "Is there a human in this photo?",
        "Is someone visible in this image?"
    ]
    vqa_positive = False
    for q in questions:
        try:
            inputs = processor(img, q, return_tensors="pt").to(device)
            out = model_vqa.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)
            if "yes" in answer.lower() or "sim" in answer.lower():
                vqa_positive = True
                break
        except Exception as e:
            print("⚠️ Erro no VQA:", e)

    if not vqa_positive:
        return JsonResponse({"detected": False, "sentiment": None})

    # --- ITM ---
    model_itm, vis_processors_itm, text_processors_itm = get_itm_model()
    if model_itm is None:
        return JsonResponse({"error": "Modelo ITM não carregado"}, status=500)

    frases = [
        'Person is in pain',
        'Person is angry',
        'Person is disgusted',
        'Person is happy',
        'Person is sad',
        'Person is surprised'
    ]

    textos_itm = [text_processors_itm["eval"](f) for f in frases]
    img_itm = vis_processors_itm["eval"](img).unsqueeze(0).to(device)

    scores = []
    for f, txt in zip(frases, textos_itm):
        try:
            with torch.no_grad():
                itm_output = model_itm({"image": img_itm, "text_input": txt}, match_head="itm")
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                prob = itm_scores[:, 1].item()
                scores.append(prob)
        except Exception as e:
            print("⚠️ Erro no ITM:", e)

    if not scores:
        return JsonResponse({"error": "ITM falhou"}, status=500)

    maior = max(scores)
    resultado_sentimento = frases[scores.index(maior)]
    print(f"✅ Sentimento escolhido: {resultado_sentimento} (score={maior:.4f})")

    # --- Enviar e-mail se detectar dor ---
    if "pain" in resultado_sentimento.lower():
        user_email = request.user.email
        if user_email:
            try:
                send_mail(
                    subject="Alerta: Detecção de dor",
                    message=f"O sistema detectou possível dor em {request.user.username} no frame capturado.",
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[user_email],
                    fail_silently=False,
                )
                print(f"📧 E-mail enviado para {user_email}")
            except Exception as e:
                print("❌ Falha ao enviar e-mail:", e)

    return JsonResponse({"detected": True, "sentiment": resultado_sentimento})

@login_required
def editar_configuracao(request):
    config, created = Configuracao.objects.get_or_create(user=request.user)

    if request.method == "POST":
        form = ConfiguracaoForm(request.POST, instance=config)
        if form.is_valid():
            form.save()
            return redirect('home')  # ou qualquer página depois de salvar
    else:
        form = ConfiguracaoForm(instance=config)

    return render(request, "contas/configuracao.html", {"form": form})

@login_required
def salvar_intervalo(request):
    if request.method == "POST":
        segundos = request.POST.get("segundos")
        if not segundos:
            return JsonResponse({"erro": "Valor não informado"}, status=400)

        try:
            segundos = int(segundos)
        except ValueError:
            return JsonResponse({"erro": "Valor inválido"}, status=400)

        # Busca ou cria a configuração do usuário atual
        config, _ = Configuracao.objects.get_or_create(user=request.user)
        config.intervalo = segundos * 1000  # salva em milissegundos
        config.save()

        return JsonResponse({"mensagem": "Intervalo atualizado com sucesso!"})

    return JsonResponse({"erro": "Método não permitido"}, status=405)
def alterar_senha(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        nova_senha = request.POST.get('nova_senha')
        confirmar_senha = request.POST.get('confirmar_senha')

        # --- Verifica se todos os campos estão preenchidos ---
        if not username or not nova_senha or not confirmar_senha:
            messages.error(request, 'Preencha todos os campos.')
            return render(request, 'contas/alterar_senha.html')


        # --- Verifica se as senhas coincidem ---
        if nova_senha != confirmar_senha:
            messages.error(request, 'As senhas não coincidem.')
            return render(request, 'contas/alterar_senha.html')


        try:
            usuario = User.objects.get(username=username)
            usuario.password = make_password(nova_senha)
            usuario.save()
            messages.success(request, 'Senha alterada com sucesso!')
            return render(request, 'contas/login.html', {"mensagem": "Senha alterada com sucesso!"})
        except User.DoesNotExist:
            messages.error(request, 'Usuário não encontrado.')
            return render(request, 'contas/alterar_senha.html')


    # Se for GET (ou primeira vez que abre a página)
    return render(request, 'contas/alterar_senha.html')
