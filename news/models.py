from django.db import models

class Articles(models.Model):
    title = models.CharField('Заголовок', max_length=50)
    anonse = models.CharField('Описание', max_length=250)
    full_test = models.TextField('Текст статьи')
    date = models.DateField('Дата публикации')

def __str__(self):
    return self.title

class Meta:
    verbose_name = 'Новость'
    verbose_name_plural = 'Новости'

def get_adsolute_url(self):
    return f'/news/{self.id}'