from django.urls import path
from .views import chatbot_view
from . import views

urlpatterns = [
    path("", chatbot_view, name="chatbot"),
    path('chat/', views.chat, name='chat')
] 