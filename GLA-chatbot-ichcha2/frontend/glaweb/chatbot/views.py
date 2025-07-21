from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
import json
import re
import os
import logging
from uuid import uuid4
from dotenv import load_dotenv
import torch

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize SentenceTransformer model
try:
    embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    raise

# Initialize Hinglish model and tokenizer
try:
    hinglish_model = AutoModelForCausalLM.from_pretrained(
        "Hinglish-Project/llama-3-8b-English-to-Hinglish",
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    hinglish_tokenizer = AutoTokenizer.from_pretrained("Hinglish-Project/llama-3-8b-English-to-Hinglish")
    # Set pad_token_id to eos_token_id if not set
    if hinglish_tokenizer.pad_token_id is None:
        hinglish_tokenizer.pad_token_id = hinglish_tokenizer.eos_token_id
    logger.info("Hinglish model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Hinglish model: {e}")
    raise

# Initialize ChromaDB client with persistent storage
try:
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), 'chroma_db'))
    collection_name = 'gla-university-chatbot'
    collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    logger.info("ChromaDB collection initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise

# Keyword mapping for topic determination (includes Hindi and Hinglish keywords)
KEYWORD_MAP = {
    'admissions': {
        'keywords': ['admission', 'eligibility', 'criteria', 'apply', 'application', 'entrance', 'exam', 'glaet', 'jee', 'counselling', 'document', 'verification', 'direct', 'sports quota', 'international', 'प्रवेश', 'पात्रता', 'आवेदन', 'परीक्षा', 'काउंसलिंग', 'bhai admission', 'bata admission'],
    },
    'academics': {
        'keywords': ['academic', 'b.tech', 'm.tech', 'mba', 'bca', 'mca', 'program', 'specialization', 'course', 'exam', 'examination', 'grading', 'faculty', 'teaching', 'phd', 'online course', 'dress code', 'laptop', 'students', 'branch change', 'शैक्षणिक', 'कोर्स', 'प्रोग्राम', 'शिक्षक', 'bhai course', 'bata course'],
    },
    'fees': {
        'keywords': ['fee', 'fees', 'scholarship', 'payment', 'charge', 'refund', 'hostel charges', 'mba fee', 'फीस', 'स्कॉलरशिप', 'भुगतान', 'bhai fees', 'bata fees'],
    },
    'placements': {
        'keywords': ['placement', 'recruiters', 'package', 'internship', 'job', 'career', 'placement cell', 'bca placement', 'प्लेसमेंट', 'नौकरी', 'पैकेज', 'bhai placement', 'bata job'],
    },
    'facilities': {
        'keywords': ['hostel', 'library', 'laboratory', 'lab', 'sports', 'medical', 'it', 'infrastructure', 'gym', 'wifi', 'wi-fi', 'transportation', 'हॉस्टल', 'पुस्तकालय', 'सुविधाएं', 'परिवहन', 'bhai hostel', 'bata facilities', 'kya-kya facilities', 'hein'],
    },
    'campus': {
        'keywords': ['campus', 'club', 'society', 'event', 'cultural', 'technical', 'fest', 'canteen', 'mess', 'food', 'transportation', 'security', 'vehicle', 'hackathon', 'ragging', 'कैंपस', 'क्लब', 'इवेंट', 'रैगिंग', 'bhai campus', 'bata event'],
    },
    'others': {
        'keywords': ['ranking', 'alumni', 'affiliation', 'research', 'international', 'contact', 'ugc', 'naac', 'incubation', 'foreign collaboration', 'parents', 'रैंकिंग', 'संपर्क', 'शोध', 'bhai ranking', 'bata contact'],
    }
}

# Load and index part1.json data
def load_and_index_data():
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'part1.json')
        logger.debug(f"Attempting to load part1.json from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            logger.warning("part1.json is empty.")
            return

        ids = []
        embeddings = []
        metadatas = []
        for item in data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            if not question or not answer:
                logger.warning(f"Skipping invalid entry: question={question}, answer={answer}")
                continue
            text = f"{question} {answer}"
            embedding = embed_model.encode(text).tolist()
            vector_id = str(uuid4())
            topic = determine_topic(question)
            metadata = {
                'question': question,
                'answer': answer,
                'topic': topic
            }
            ids.append(vector_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
        
        if ids:
            collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logger.info(f"Indexed {len(ids)} entries in ChromaDB.")
            count = collection.count()
            logger.debug(f"ChromaDB collection contains {count} vectors.")
            sample = collection.peek(limit=2)
            logger.debug(f"Sample vectors: {sample}")
        else:
            logger.warning("No valid entries to index in ChromaDB.")
    except FileNotFoundError:
        logger.error(f"part1.json not found at {json_path}")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in part1.json")
        raise
    except Exception as e:
        logger.error(f"Error indexing data: {e}")
        raise

def determine_topic(question):
    """Determine the topic of a question based on keywords."""
    question_clean = clean_text(question)
    for topic, data in KEYWORD_MAP.items():
        if any(keyword in question_clean for keyword in data['keywords']):
            logger.debug(f"Topic determined: {topic} for question: {question}")
            return topic
    logger.debug(f"Topic defaulted to 'others' for question: {question}")
    return 'others'

def clean_text(text):
    """Remove punctuation, convert to lowercase, and normalize whitespace."""
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.lower().split())
    return text

# Generate Hinglish response using llama-3-8b-English-to-Hinglish
def generate_hinglish_response(answer):
    try:
        # Improved prompt with examples
        prompt = (
            "Translate the following English sentence into a natural Hinglish response. "
            "Keep the tone conversational and mix Hindi and English naturally as a young Indian would speak. "
            "Example: Input: 'The hostel has Wi-Fi and a gym.' Output: 'Bhai, hostel mein Wi-Fi hai aur gym bhi milta hai.'\n"
            f"Input: {answer}\nOutput:"
        )
        inputs = hinglish_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: val.to(hinglish_model.device) for key, val in inputs.items()}
        outputs = hinglish_model.generate(
            **inputs,
            max_length=250,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=hinglish_tokenizer.pad_token_id
        )
        hinglish_answer = hinglish_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up the response
        hinglish_answer = hinglish_answer.replace(prompt, '').strip()
        logger.debug(f"Hinglish response generated: {hinglish_answer}")
        if not hinglish_answer or "Output:" in hinglish_answer:
            logger.warning("Hinglish model returned empty or invalid response, falling back to GoogleTranslator.")
            return GoogleTranslator(source='en', target='hi').translate(answer)
        return hinglish_answer
    except Exception as e:
        logger.error(f"Error generating Hinglish response: {e}")
        return GoogleTranslator(source='en', target='hi').translate(answer)

# Load data on startup
load_and_index_data()

@csrf_exempt
@require_POST
def chat(request):
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        logger.debug(f"Received question: {question}")

        if not question:
            logger.warning("Empty question received.")
            return JsonResponse({
                'answer': '<strong>❓ Error:</strong><br><br>Please enter a valid question.'
            })

        # Detect language and check for Hinglish
        try:
            lang = detect(question)
            logger.debug(f"Detected language: {lang}")
            is_hinglish = any(word in clean_text(question) for word in ['bhai', 'bata', 'kya', 'hein', 'bol', 'dekh'])
            if is_hinglish:
                lang = 'hi'
                logger.debug("Hinglish detected, forcing language to Hindi.")
        except LangDetectException:
            lang = 'hi'
            logger.warning("Language detection failed, defaulting to Hindi.")

        # Generate embedding for the query
        query_embedding = embed_model.encode(question).tolist()

        # Query ChromaDB for top match
        query_response = collection.query(query_embeddings=[query_embedding], n_results=1)
        results = query_response['metadatas'][0] if query_response['metadatas'] else []
        distances = query_response['distances'][0] if query_response['distances'] else []

        if results and distances and distances[0] < 1.2:
            answer = results[0]['answer']
            topic = results[0]['topic']
            logger.debug(f"Match found: topic={topic}, answer={answer}, distance={distances[0]}")
        else:
            answer = "I’m here to assist with all your questions about GLA University! Please ask about admissions, academics, fees, placements, facilities, campus life, or other topics for detailed information. Visit www.gla.ac.in."
            topic = 'general'
            logger.debug(f"No match found or distance too high: {distances[0] if distances else 'N/A'}")

        # Process response based on language
        if lang == 'hi' and is_hinglish:
            # Use Hinglish model
            translated_answer = generate_hinglish_response(answer)
            formatted_answer = f"<strong>Bhai, yeh lo jawab:</strong><br><br>{translated_answer}<br><br><em>Zyada info ke liye www.gla.ac.in check karo.</em>"
        else:
            # Use GoogleTranslator for Hindi or other languages
            try:
                translated_answer = GoogleTranslator(source='en', target=lang).translate(answer)
                if not translated_answer:
                    logger.warning("Translation returned empty, using original answer.")
                    translated_answer = answer
            except Exception as e:
                logger.error(f"Translation error: {e}")
                translated_answer = answer
            formatted_answer = f"<strong>{'प्रतिक्रिया' if lang == 'hi' else 'Response'}:</strong><br><br>{translated_answer}<br><br><em>{'अधिक जानकारी के लिए www.gla.ac.in पर जाएं।' if lang == 'hi' else 'Visit www.gla.ac.in for more details.'}</em>"

        return JsonResponse({
            'answer': formatted_answer
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JsonResponse({
            'answer': '<strong>❌ Error:</strong><br><br>Something went wrong. Please try again later.'
        })

def chatbot_view(request):
    return render(request, 'gla_web/chatbot.html')