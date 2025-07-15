
from django.urls import path, include
from . import views
from news.views import ArticlesApiView

urlpatterns = [
    path('',views.news_home, name = 'news_home'),
    path('create',views.create, name = 'create'),
    path("postuser/", views.postuser),
    path("<int:pk>", views.NewsDetailView.as_view(), name="news-detail"),
    path("<int:pk>/update", views.NewsUpdateView.as_view() , name="news-update"),
    path("<int:pk>/delete", views.NewsDeleteView.as_view() , name="news-delete"),
    path("api/cat/", views.ArticlesApiView.as_view(), name="news-api"),
    path("numpy_nn/", views.numpy_nn, name="numpy_nn"),
]
