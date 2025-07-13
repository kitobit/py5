from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def index(request):
    data = {
        'title':'Главная страница',
        'values': ['some', 'hello', 3, 4, 5,6,7,8,9,10,11,12,13,14,15],
        'obj': {
            'car': 'BMW',
            'color': 'red',
            'age': '18',
        }
    }
    return render(request, 'main/index.html',data)

def about(request):
    return render(request, 'main/about.html')