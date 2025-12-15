from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat'),  # Add this - home page
    path('query', views.query_view, name='query'),
    path('health', views.health_view, name='health'),
]