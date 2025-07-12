from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def index(request):
    return HttpResponse('<H4>Test 2</H4>')

def about(request):
    return HttpResponse('<H4>Текст страницы about</H4>')