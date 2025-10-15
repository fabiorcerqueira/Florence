from django.forms import ModelForm
from .models import Transação
from django import forms
from .models import Configuracao   

class TransacaoForm(ModelForm):
    class Meta:
        model = Transação
        fields = ['data','descricao','valor','observacoes','categoria']

class ConfiguracaoForm(forms.ModelForm):
    class Meta:
        model = Configuracao
        fields = ['user', 'intervalo']  # coloque aqui os campos corretos

