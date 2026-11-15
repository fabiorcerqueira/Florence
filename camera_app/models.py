from django.contrib.auth.models import User
from django.db import models


class Configuracao(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    intervalo = models.IntegerField(default=30000, help_text="Intervalo entre capturas (em milissegundos)")

    def __str__(self):
        return f"Configuração de {self.user.username}"
