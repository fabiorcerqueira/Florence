from django import forms

from .models import Configuracao


class ConfiguracaoForm(forms.ModelForm):
    """Edita apenas o intervalo do próprio usuário (o dono vem da view)."""

    class Meta:
        model = Configuracao
        fields = ['intervalo']

    def clean_intervalo(self):
        intervalo = self.cleaned_data['intervalo']
        if intervalo < 5000:
            raise forms.ValidationError("O intervalo mínimo é de 5000 ms (5 segundos).")
        return intervalo
