from django.urls import path
from .views import chatbot_view, chat_api

urlpatterns = [
    path("", chatbot_view, name="chatbot"),
    path('chat/', chat_api, name='chat_api'),
]