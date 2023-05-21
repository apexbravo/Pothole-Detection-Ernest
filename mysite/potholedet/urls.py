from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'potholedet'

urlpatterns = [
    path('', views.index, name='index'),
    path('videotest/',
         views.videotest, name='videotest'),
    path('result/', views.result, name='result'),
    path('potholemap/', views.potholemap, name='potholemap'),
    path('about/', views.about, name='about'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
