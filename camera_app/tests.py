"""Testes da aplicação Florence sem carregar modelos.

Os modelos BLIP e Qwen são substituídos por mocks, então os testes rodam em CPU,
sem GPU e sem baixar pesos:  python manage.py test camera_app
"""
import io
from unittest import mock

from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.test import TestCase, override_settings
from PIL import Image

from . import views
from .models import Configuracao
from .qwen_inference import QwenInferenceError, parse_category


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="JPEG")
    buf.seek(0)
    buf.name = "frame.jpg"
    return buf


class ParseCategoryTests(TestCase):
    def test_accepts_the_six_categories_case_and_punctuation_insensitive(self):
        self.assertEqual(parse_category("Pain"), "pain")
        self.assertEqual(parse_category("surprise."), "surprise")

    def test_rejects_answers_outside_the_categories(self):
        for answer in ("No pain", "Painful", "Pain, because the brow is furrowed"):
            with self.assertRaises(QwenInferenceError):
                parse_category(answer)


class PasswordChangeTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user("alice", password="senha-original-123")
        self.bob = User.objects.create_user("bob", password="senha-do-bob-456")

    def test_requires_login(self):
        resp = self.client.post("/alterar_senha/", {
            "senha_atual": "x", "nova_senha": "Nova-senha-789", "confirmar_senha": "Nova-senha-789",
        })
        self.assertEqual(resp.status_code, 302)
        self.bob.refresh_from_db()
        self.assertTrue(self.bob.check_password("senha-do-bob-456"))

    def test_cannot_change_another_user_by_username(self):
        self.client.login(username="alice", password="senha-original-123")
        self.client.post("/alterar_senha/", {
            "username": "bob", "senha_atual": "senha-original-123",
            "nova_senha": "Nova-senha-789", "confirmar_senha": "Nova-senha-789",
        })
        self.bob.refresh_from_db()
        self.assertTrue(self.bob.check_password("senha-do-bob-456"))

    def test_wrong_current_password_is_rejected(self):
        self.client.login(username="alice", password="senha-original-123")
        self.client.post("/alterar_senha/", {
            "senha_atual": "errada", "nova_senha": "Nova-senha-789", "confirmar_senha": "Nova-senha-789",
        })
        self.alice.refresh_from_db()
        self.assertTrue(self.alice.check_password("senha-original-123"))

    def test_changes_own_password_and_stays_logged_in(self):
        self.client.login(username="alice", password="senha-original-123")
        resp = self.client.post("/alterar_senha/", {
            "senha_atual": "senha-original-123",
            "nova_senha": "Nova-senha-789", "confirmar_senha": "Nova-senha-789",
        })
        self.assertRedirects(resp, "/")
        self.alice.refresh_from_db()
        self.assertTrue(self.alice.check_password("Nova-senha-789"))
        self.assertEqual(self.client.get("/camera/").status_code, 200)


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
    FLORENCE_ALERT_COOLDOWN_SECONDS=600,
)
class ProcessFrameTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user("alice", email="alice@example.org", password="senha-original-123")
        self.client.login(username="alice", password="senha-original-123")

    def post_frame(self):
        return self.client.post("/process_frame/", {"frame": jpeg_bytes()})

    def test_session_expired_returns_json_401(self):
        self.client.logout()
        resp = self.post_frame()
        self.assertEqual(resp.status_code, 401)
        self.assertIn("error", resp.json())

    def test_csrf_is_enforced(self):
        client = self.client_class(enforce_csrf_checks=True)
        client.login(username="alice", password="senha-original-123")
        resp = client.post("/process_frame/", {"frame": jpeg_bytes()})
        self.assertEqual(resp.status_code, 403)

    @mock.patch.object(views, "person_present", return_value=False)
    def test_no_person(self, _):
        resp = self.post_frame()
        self.assertEqual(resp.json(), {"detected": False, "sentiment": None})
        self.assertEqual(len(mail.outbox), 0)

    @mock.patch.object(views, "classify_expression", return_value=("pain", "Person is in pain"))
    @mock.patch.object(views, "person_present", return_value=True)
    def test_pain_sends_one_email_then_cooldown(self, *_):
        first = self.post_frame().json()
        second = self.post_frame().json()
        self.assertEqual(first["alert"], "sent")
        self.assertEqual(second["alert"], "cooldown")
        self.assertEqual(len(mail.outbox), 1)

    @mock.patch.object(views, "classify_expression", return_value=("pain", "Person is in pain"))
    @mock.patch.object(views, "person_present", return_value=True)
    def test_pain_without_email_is_reported(self, *_):
        self.user.email = ""
        self.user.save()
        self.assertEqual(self.post_frame().json()["alert"], "no_email")
        self.assertEqual(len(mail.outbox), 0)

    @mock.patch.object(views, "classify_expression", return_value=("happiness", "Person is happy"))
    @mock.patch.object(views, "person_present", return_value=True)
    def test_non_pain_category_sends_no_email(self, *_):
        data = self.post_frame().json()
        self.assertEqual(data["category"], "happiness")
        self.assertNotIn("alert", data)
        self.assertEqual(len(mail.outbox), 0)

    @mock.patch.object(views, "classify_expression", side_effect=QwenInferenceError("x"))
    @mock.patch.object(views, "person_present", return_value=True)
    def test_inconclusive_answer(self, *_):
        resp = self.post_frame()
        self.assertEqual(resp.status_code, 503)
        self.assertTrue(resp.json()["inconclusive"])
        self.assertEqual(len(mail.outbox), 0)


class ConfiguracaoTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user("alice", password="senha-original-123")
        self.bob = User.objects.create_user("bob", password="senha-do-bob-456")
        self.client.login(username="alice", password="senha-original-123")

    def test_form_cannot_reassign_owner(self):
        self.client.post("/configuracao/", {"user": self.bob.pk, "intervalo": 60000})
        config = Configuracao.objects.get(user=self.alice)
        self.assertEqual(config.intervalo, 60000)
        self.assertFalse(Configuracao.objects.filter(user=self.bob).exists())

    def test_interval_minimum(self):
        resp = self.client.post("/salvar_intervalo/", {"segundos": "1"})
        self.assertEqual(resp.status_code, 400)
        resp = self.client.post("/salvar_intervalo/", {"segundos": "45"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Configuracao.objects.get(user=self.alice).intervalo, 45000)
