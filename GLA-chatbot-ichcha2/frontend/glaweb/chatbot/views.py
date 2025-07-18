from django.shortcuts import render

def chatbot_view(request):
    return render(request, "gla_web/chatbot.html")
