import io
import time

import cv2
import numpy as np
import torch
from PIL import Image
from threading import Lock

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.core.cache import cache
from django.core.mail import send_mail
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST
from transformers import BlipProcessor, BlipForQuestionAnswering

from .form import ConfiguracaoForm
from .models import Configuracao
from .qwen_inference import classify_expression, QwenInferenceError

processing_lock = Lock()

# ------------------ Views básicas ------------------

@login_required
def home(request):
    return render(request, 'contas/home.html', {'usuario': request.user})


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            # Ignora qualquer redirecionamento antigo
            return redirect('home')

        messages.error(request, 'Usuário ou senha inválidos.')
        return render(request, 'contas/login.html')

    return render(request, 'contas/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def alterar_senha(request):
    """Troca a senha do próprio usuário logado, exigindo a senha atual.

    Usa o PasswordChangeForm do Django, que confere a senha atual e aplica os
    validadores de AUTH_PASSWORD_VALIDATORS.
    """
    if request.method == 'POST':
        form = PasswordChangeForm(
            request.user,
            {
                'old_password': request.POST.get('senha_atual', ''),
                'new_password1': request.POST.get('nova_senha', ''),
                'new_password2': request.POST.get('confirmar_senha', ''),
            },
        )
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # mantém o usuário logado
            messages.success(request, 'Senha alterada com sucesso!')
            return redirect('home')
        for errors in form.errors.values():
            for error in errors:
                messages.error(request, error)

    return render(request, 'contas/alterar_senha.html')

# ------------------ Câmera ------------------

@login_required
@ensure_csrf_cookie  # garante o cookie csrftoken usado pelo JavaScript da página
def camera_view(request):
    config, _ = Configuracao.objects.get_or_create(user=request.user)
    intervalo = config.intervalo if config.intervalo else 30000  # milissegundos
    return render(request, 'contas/camera.html', {"intervalo": intervalo})


@login_required
def editar_configuracao(request):
    config, _ = Configuracao.objects.get_or_create(user=request.user)

    if request.method == "POST":
        form = ConfiguracaoForm(request.POST, instance=config)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = ConfiguracaoForm(instance=config)

    return render(request, "contas/configuracao.html", {"form": form})


MIN_INTERVAL_SECONDS = 5


@require_POST
def salvar_intervalo(request):
    if not request.user.is_authenticated:
        return JsonResponse({"erro": "Sessão expirada. Faça login novamente."}, status=401)

    segundos = request.POST.get("segundos")
    if not segundos:
        return JsonResponse({"erro": "Valor não informado"}, status=400)

    try:
        segundos = int(segundos)
    except ValueError:
        return JsonResponse({"erro": "Valor inválido"}, status=400)

    if segundos < MIN_INTERVAL_SECONDS:
        return JsonResponse(
            {"erro": f"O intervalo mínimo é de {MIN_INTERVAL_SECONDS} segundos."}, status=400
        )

    config, _ = Configuracao.objects.get_or_create(user=request.user)
    config.intervalo = segundos * 1000  # salva em milissegundos
    config.save()

    return JsonResponse({"mensagem": "Intervalo atualizado com sucesso!", "segundos": segundos})

# ------------------ IA: BLIP VQA + Qwen2.5-VL-3B ------------------
# A checagem de presença humana usa o BLIP VQA.
# A classificação da expressão usa o Qwen2.5-VL-3B-Instruct.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_vqa_processor = None
_vqa_model = None

PRESENCE_QUESTIONS = [
    "Is there a person in the image?",
    "Is there a human in this photo?",
    "Is someone visible in this image?",
]


def get_vqa_model():
    global _vqa_processor, _vqa_model
    if _vqa_model is None:
        _vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        _vqa_model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        ).to(device).eval()
    return _vqa_processor, _vqa_model


def person_present(img):
    processor, model_vqa = get_vqa_model()
    with torch.inference_mode():
        for question in PRESENCE_QUESTIONS:
            inputs = processor(img, question, return_tensors="pt").to(device)
            output = model_vqa.generate(**inputs)
            answer = processor.decode(output[0], skip_special_tokens=True).strip().lower()
            if answer in ("yes", "sim"):
                return True
    return False


def decode_frame(file_bytes):
    try:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_cv is not None:
            return Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print("Erro ao decodificar com OpenCV:", e)
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return None


def send_pain_alert(user):
    """Envia o e-mail de alerta e devolve um status legível para a interface.

    Status possíveis: "sent", "cooldown" (alerta recente já enviado),
    "no_email" (usuário sem e-mail cadastrado) e "failed" (erro no envio).
    """
    if not user.email:
        return "no_email"

    cooldown = settings.FLORENCE_ALERT_COOLDOWN_SECONDS
    cache_key = f"florence:last_pain_alert:{user.pk}"
    last = cache.get(cache_key)
    now = time.time()
    if cooldown > 0 and last is not None and now - last < cooldown:
        return "cooldown"

    try:
        send_mail(
            subject="Alerta: Detecção de dor",
            message=(
                f"O sistema detectou uma expressão facial compatível com dor em um quadro "
                f"capturado na sessão de {user.username}. Verifique a pessoa monitorada. "
                f"Este alerta não é um diagnóstico."
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
    except Exception as e:
        print("Falha ao enviar e-mail:", e)
        return "failed"

    if cooldown > 0:
        cache.set(cache_key, now, timeout=cooldown)
    return "sent"


@require_POST
def process_frame(request):
    # Requisição AJAX: responde 401 em JSON em vez de redirecionar para o login,
    # para que a página possa avisar que a sessão expirou.
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Sessão expirada. Faça login novamente."}, status=401)

    file = request.FILES.get("frame")
    if not file:
        return JsonResponse({"error": "Nenhum frame recebido"}, status=400)

    img = decode_frame(file.read())
    if img is None:
        return JsonResponse({"error": "Falha ao decodificar frame"}, status=400)

    # Protege os modelos de chamadas simultâneas de inferência.
    with processing_lock:
        try:
            if not person_present(img):
                return JsonResponse({"detected": False, "sentiment": None})
            category, sentiment = classify_expression(img)
        except QwenInferenceError as exc:
            print("Resposta inconclusiva do Qwen:", exc)
            return JsonResponse(
                {"detected": True, "inconclusive": True,
                 "error": "Classificação inconclusiva"},
                status=503,
            )
        except Exception as exc:
            print("Erro na inferência Florence:", type(exc).__name__, str(exc))
            return JsonResponse({"error": "Falha no processamento da imagem"}, status=503)

    response = {"detected": True, "sentiment": sentiment, "category": category}
    if category == "pain":
        response["alert"] = send_pain_alert(request.user)
    return JsonResponse(response)
