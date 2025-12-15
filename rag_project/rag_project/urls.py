# rag_project/urls.py (main urls.py)

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('rag_api.urls')),
]