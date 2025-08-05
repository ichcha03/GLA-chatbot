from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load vector DB
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./gla_chroma", embedding_function=embedding)

# Setup LLM and prompt
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant for GLA University. Answer clearly, with tables or visual formatting if helpful.

    Context: {context}

    Question: {question}

    Answer:
    """
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

@csrf_exempt
def get_answer(request):
    if request.method == "POST":
        body = json.loads(request.body)
        question = body.get("question", "")

        if not question:
            return JsonResponse({"error": "No question provided"}, status=400)

        try:
            answer = qa.run(question)
            return JsonResponse({"answer": answer})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


