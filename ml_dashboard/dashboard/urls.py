from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('train/<int:dataset_id>/', views.train, name='train'),
    path('test/', views.test, name='test'),
    path('delete/<int:dataset_id>/', views.delete_dataset, name='delete'),  # Add this line
]