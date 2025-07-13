from .models import Articles
from django.forms import ModelForm

class ArticlesForm(ModelForm):
    class Meta:
        Model = Articles
        fields = ['title']