from django.db import models
from django.contrib.auth.models import User


class Categoria(models.Model):
    nome = models.CharField(max_length=100)
    dt_criacao = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.nome


class Transação(models.Model):
    data = models.DateTimeField()
    descricao = models.CharField(max_length=200)
    valor = models.DecimalField(max_digits=7, decimal_places=2)
    categoria = models.ForeignKey(Categoria, on_delete=models.CASCADE)
    observacoes = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name_plural = 'Transaçoes'

    def __str__(self):
        return self.descricao
    
class Configuracao(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    intervalo = models.IntegerField(default=30000, help_text="Intervalo entre capturas (em milissegundos)")

    def __str__(self):
        return f"Configuração de {self.user.username}"