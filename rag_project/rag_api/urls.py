from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('query', views.query_view, name='query'),
    path('subjects', views.subjects_view, name='subjects'),
    path('health', views.health_view, name='health'),
]