from django.shortcuts import render
from .models import Articles
from .forms import ArticlesForm
from datetime import date
from django.http import HttpResponse
from django.views.generic import DetailView, UpdateView, DeleteView

def create(request):

    # form = ArticlesForm()

    # data={'form': form}

    tom = Articles.objects.create(title="Tom",date=date.today())

    return render(request, 'news/create.html')

def news_home(request):

    news = Articles.objects.order_by('-date')

    return render(request, 'news/news_home.html',{'news':news})

def postuser(request):
    title1 = request.POST.get("title", "Undefined")
    anonse1 = request.POST.get("anonse", "Undefined")
    full_test1 = request.POST.get("full_text", "Undefined")
    date1 = request.POST.get("date", 1)

    # tom = Articles.objects.create(title=title1, anonse=anonse1, full_text = full_text1,  date=date1)

    #tom = Articles(title= f"{title1}", anonse=f"{anonse1}", full_text = f"{full_text1}",  date=f"{date1}")
    tom = Articles.objects.create(title=title1, anonse = anonse1, full_test = full_test1,  date=date1)
    # tom.save()


    #return HttpResponse(f"<h2>title: {title1} <br> anonse: {anonse1} <br> full_text: {full_text1} <br> date: {date1} </h2>")

    news = Articles.objects.order_by('-date')
    return render(request, 'news/news_home.html', {'news': news})

class NewsDetailView(DetailView):
    model = Articles
    template_name = 'news/details_view.html'
    context_object_name = 'article'

class NewsUpdateView(UpdateView):
    model = Articles
    template_name = 'news/create.html'
    context_object_name = 'article'
    form_class = ArticlesForm

class NewsDeleteView(DeleteView):
    model = Articles
    success_url = '/news/'
    template_name = 'news/news-delete.html'