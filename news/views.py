from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import generics

from .models import Articles
from .forms import ArticlesForm
from datetime import date
from django.http import HttpResponse
from django.views.generic import DetailView, UpdateView, DeleteView

from .serializers import ArticlesSerializer

import numpy as np
import matplotlib.pyplot as plt

def create(request):
    # form = ArticlesForm()

    # data={'form': form}

    tom = Articles.objects.create(title="Tom", date=date.today())

    return render(request, 'news/create.html')


def news_home(request):
    news = Articles.objects.order_by('-date')

    return render(request, 'news/news_home.html', {'news': news})


def postuser(request):
    title1 = request.POST.get("title", "Undefined")
    anonse1 = request.POST.get("anonse", "Undefined")
    full_test1 = request.POST.get("full_text", "Undefined")
    date1 = request.POST.get("date", 1)

    # tom = Articles.objects.create(title=title1, anonse=anonse1, full_text = full_text1,  date=date1)

    #tom = Articles(title= f"{title1}", anonse=f"{anonse1}", full_text = f"{full_text1}",  date=f"{date1}")

    tom = Articles.objects.create(title=title1, anonse=anonse1, full_test=full_test1, date=date1)
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


class ArticlesApiView(generics.ListAPIView):
    queryset = Articles.objects.all()
    serializer_class = ArticlesSerializer


def numpy_nn(request):
    images, labels = load_dataset()

    weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
    weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
    bias_input_to_hidden = np.zeros((20, 1))
    bias_hidden_to_output = np.zeros((10, 1))

    epochs = 3
    e_loss = 0
    e_correct = 0
    learning_rate = 0.01

    for epoch in range(epochs):
        print(f"Epoch №{epoch}")

        for image, label in zip(images, labels):
            image = np.reshape(image, (-1, 1))
            label = np.reshape(label, (-1, 1))

            # Forward propagation (to hidden layer)
            hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
            hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

            # Forward propagation (to output layer)
            output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
            output = 1 / (1 + np.exp(-output_raw))

            # Loss / Error calculation
            e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
            e_correct += int(np.argmax(output) == np.argmax(label))

            # Backpropagation (output layer)
            delta_output = output - label
            weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
            bias_hidden_to_output += -learning_rate * delta_output

            # Backpropagation (hidden layer)
            delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
            weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
            bias_input_to_hidden += -learning_rate * delta_hidden

        # DONE

        # print some debug info between epochs
        print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
        print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
        e_loss = 0
        e_correct = 0

    # CHECK CUSTOM
    test_image = plt.imread("/home/kitobit/py5/static/custom.jpg", format="jpeg")

    # Grayscale + Unit RGB + inverse colors
    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    test_image = 1 - (gray(test_image).astype("float32") / 255)

    # Reshape
    test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

    # Predict
    image = np.reshape(test_image, (-1, 1))

    # Forward propagation (to hidden layer)
    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid
    # Forward propagation (to output layer)
    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    plt.imshow(test_image.reshape(28, 28), cmap="Greys")
    plt.title(f"NN suggests the CUSTOM number is: {output.argmax()}")
    # plt.show()
    print(f"NN suggests the CUSTOM number is: {output.argmax()}")

    #news = Articles.objects.order_by('-date')

    # return HttpResponse(f"NN suggests the CUSTOM number is: {output.argmax()}")
    data = {
        'title': "numpy nn",
        'numpy_rezult': f"NN suggests the CUSTOM number is: {output.argmax()}",
        'src':  "http://127.0.0.1:8000/static/main/img/custom.jpg",
        'src2': "https://kitobit.pythonanywhere.com/static/main/img/custom.jpg"
    }
    return render(request, 'news/numpy_nn.html', data)


def load_dataset():
    with np.load('/home/kitobit/py5/static/mnist.npz') as f:
        # convert from RGB to Unit RGB
        x_train = f['x_train'].astype("float32") / 255

        # reshape from (60000, 28, 28) into (60000, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

        # labels
        y_train = f['y_train']

        # convert to output layer format
        y_train = np.eye(10)[y_train]

        return x_train, y_train
