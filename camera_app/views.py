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
from .qwen_inference import classify_expression, QwenInferenceError
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


# ------------------ IA: BLIP VQA + Qwen2.5-VL-3B ------------------
# A checagem de presença humana continua usando o BLIP VQA original.
# A classificação afetiva passa a utilizar o Qwen2.5-VL-3B-Instruct.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_vqa_processor = None
_vqa_model = None

def get_vqa_model():
    global _vqa_processor, _vqa_model
    if _vqa_model is None:
        _vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        _vqa_model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        ).to(device).eval()
    return _vqa_processor, _vqa_model

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

    # Protege os modelos de chamadas simultâneas de inferência.
    with processing_lock:
        try:
            processor, model_vqa = get_vqa_model()
            questions = [
                "Is there a person in the image?",
                "Is there a human in this photo?",
                "Is someone visible in this image?",
            ]
            vqa_positive = False
            with torch.inference_mode():
                for question in questions:
                    inputs = processor(img, question, return_tensors="pt").to(device)
                    output = model_vqa.generate(**inputs)
                    answer = processor.decode(output[0], skip_special_tokens=True).strip().lower()
                    if answer in ("yes", "sim"):
                        vqa_positive = True
                        break
            if not vqa_positive:
                return JsonResponse({"detected": False, "sentiment": None})

            resultado_sentimento = classify_expression(img)
        except QwenInferenceError as exc:
            print("Erro na classificação Qwen:", exc)
            return JsonResponse({"error": "Classificação inconclusiva"}, status=503)
        except Exception as exc:
            print("Erro na inferência Florence:", type(exc).__name__, str(exc))
            return JsonResponse({"error": "Falha no processamento da imagem"}, status=503)

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
