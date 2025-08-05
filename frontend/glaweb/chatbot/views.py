from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import google.generativeai as genai
from .database import db
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

def detect_language(text):
    """Simple language detection based on character patterns"""
    # Hindi/Devanagari script detection
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    # Arabic script detection
    elif re.search(r'[\u0600-\u06FF]', text):
        return 'arabic'
    # Chinese characters detection
    elif re.search(r'[\u4e00-\u9fff]', text):
        return 'chinese'
    # Japanese characters detection
    elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return 'japanese'
    # Korean characters detection
    elif re.search(r'[\uac00-\ud7af]', text):
        return 'korean'
    # Spanish detection (basic)
    elif re.search(r'[ñáéíóúü]', text.lower()):
        return 'spanish'
    # French detection (basic)
    elif re.search(r'[àâäéèêëïîôùûüÿç]', text.lower()):
        return 'french'
    # German detection (basic)
    elif re.search(r'[äöüß]', text.lower()):
        return 'german'
    else:
        return 'english'

def get_language_instruction(language):
    """Get language-specific instruction for the prompt"""
    language_instructions = {
        'hindi': 'Respond in Hindi (हिंदी में उत्तर दें)',
        'arabic': 'Respond in Arabic (أجب باللغة العربية)',
        'chinese': 'Respond in Chinese (用中文回答)',
        'japanese': 'Respond in Japanese (日本語で答えてください)',
        'korean': 'Respond in Korean (한국어로 답변해주세요)',
        'spanish': 'Respond in Spanish (Responde en español)',
        'french': 'Respond in French (Répondez en français)',
        'german': 'Respond in German (Antworten Sie auf Deutsch)',
        'english': 'Respond in English'
    }
    return language_instructions.get(language, 'Respond in English')

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
try:
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    print(f"Gemini API configured successfully with model: gemini-1.5-flash")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    try:
        # Fallback to another available model
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("Fallback to gemini-2.0-flash successful")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        model = None

def chatbot_view(request):
    """Render the chatbot interface"""
    # Initialize database on first load
    db.initialize_database()
    return render(request, 'gla_web/chatbot.html')

@csrf_exempt
def chat_api(request):
    """Handle chat API requests with RAG"""
    if request.method == 'POST':
        try:
            print("Chat API called")
            data = json.loads(request.body)
            question = data.get('question', '')
            print(f"Question received: {question}")
            
            if not question:
                return JsonResponse({'error': 'No question provided'}, status=400)
            
            # Detect the language of the user's question
            detected_language = detect_language(question)
            language_instruction = get_language_instruction(detected_language)
            print(f"Detected language: {detected_language}")
            
            # Search for relevant context in database
            print("Searching database...")
            search_results = db.search_similar(question, n_results=5)
            print(f"Search results: {search_results}")
            
            if not search_results or not search_results['documents']:
                # Create language-specific fallback message
                fallback_messages = {
                    'hindi': 'मुझे खुशी होगी अगर आप GLA विश्वविद्यालय के बारे में कुछ और पूछें - प्रवेश, शुल्क, प्लेसमेंट, सुविधाएं, या कैंपस जीवन के बारे में।',
                    'spanish': 'Lo siento, no pude encontrar información relevante. Por favor pregunta sobre admisiones, tarifas, colocaciones, instalaciones o vida universitaria de GLA.',
                    'french': 'Désolé, je n\'ai pas trouvé d\'informations pertinentes. Veuillez poser des questions sur les admissions, les frais, les placements, les installations ou la vie sur le campus de GLA.',
                    'german': 'Entschuldigung, ich konnte keine relevanten Informationen finden. Bitte fragen Sie nach Zulassungen, Gebühren, Praktika, Einrichtungen oder dem Campus-Leben der GLA.',
                    'english': 'I apologize, but I couldn\'t find relevant information about your query. Please ask about GLA University admissions, academics, fees, placements, facilities, or campus life.'
                }
                return JsonResponse({
                    'answer': fallback_messages.get(detected_language, fallback_messages['english'])
                })
            
            # Prepare context from search results
            context_documents = search_results['documents'][0]  # First result set
            context = "\n".join(context_documents)
            print(f"Context prepared: {context[:200]}...")
            
            # Create prompt for Gemini with language instruction
            prompt = f"""You are a helpful GLA University assistant. Answer based ONLY on the provided context about GLA University.

Context from GLA University database:
{context}

User Question: {question}

Instructions:
1. {language_instruction}
2. Answer based only on the provided context
3. Be helpful and informative
4. Keep responses concise but comprehensive
5. Include specific details like fees, dates, or requirements when available
6. If the question is not related to GLA University, politely redirect to university-related topics
7. If information is not available in the context, suggest contacting the university directly

Answer:"""
            
            # Generate response using Gemini
            print("Generating response with Gemini...")
            if model is None:
                raise Exception("Gemini model is not initialized. Please check your API key configuration.")
            
            response = model.generate_content(prompt)
            answer = response.text
            print(f"Response generated successfully")
            
            return JsonResponse({'answer': answer})
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in chat_api: {str(e)}")
            print(f"Full traceback: {error_details}")
            return JsonResponse({
                'answer': f'I apologize, but I encountered an error while processing your request. Error: {str(e)}. Please try again or contact the university directly.'
            })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)