from django.urls import path
from .views import get_answer

urlpatterns = [
    path("api/ask/", get_answer, name="get_answer"),
]
