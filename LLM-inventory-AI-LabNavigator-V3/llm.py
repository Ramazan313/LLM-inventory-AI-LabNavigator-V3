from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriver_mongo import retrieve
import nltk
from nltk.tokenize import sent_tokenize
from ollama import chat

# NLTK tokenizer kontrolü
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI()

# CORS ayarları
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama model
OLLAMA_MODEL = "gemma3:4b"

class Query(BaseModel):
    question: str


def generate_short_answer(question: str, top_k: int = 2):
    context_chunks = retrieve(question, top_k=top_k)
    if not context_chunks:
        return {"answer": "Bilgi almak için teknokent yetkilisi ile görüşebilirsiniz iletişim no:(0264) 346 02 02 ", "sources": []}

    prompt = (
        "Aşağıdaki kaynaklardan yararlanarak soruya kısa, öz ve net cevap ver.\n"
        "Cevap 1-2 cümleyi geçmesin.\n\n"
        "Kaynaklar:\n" +
        "\n".join([f"{i+1}. {c[:1000]}" for i, c in enumerate(context_chunks)]) +
        f"\n\nSoru: {question}\nCevap:"
    )

    # Streaming başlat
    stream = chat(
        model=OLLAMA_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    # Akışı birleştir
    full_response = ""
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            full_response += chunk["message"]["content"]

    # Cümle sınırlandırma
    answer_sentences = sent_tokenize(full_response, language='turkish')
    answer = ". ".join(answer_sentences[:2]).rstrip(".") + "."

    if any(phrase in answer.lower() for phrase in [
        "bulunmamaktadır", "yer almamaktadır", "bilgi yok", "verilen kaynaklarda"
    ]):
        return {
            "answer": "Bilgi almak için teknokent yetkilisi ile görüşebilirsiniz. İletişim no: (0264) 346 02 02",
            "sources": []
        }


    return {
        "answer": answer.strip(),
        "sources": context_chunks
    }


@app.get("/ask")
def ask_question(q: str, top_k: int = 2):
    return generate_short_answer(q, top_k=top_k)


@app.post("/ask")
def ask(q: Query, top_k: int = 2):
    return generate_short_answer(q.question, top_k=top_k)
