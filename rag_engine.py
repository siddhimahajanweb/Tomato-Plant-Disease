
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load disease knowledge
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "disease_knowledge.json")

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
    knowledge = json.load(file)

# Convert disease information into searchable text
documents = []

for disease, details in knowledge.items():
    text = f"""
    Disease: {disease}
    Description: {details['description']}
    Symptoms: {details['symptoms']}
    Prevention: {details['prevention']}
    Treatment: {details['treatment']}
    """
    documents.append(text)

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True)
embeddings = np.asarray(embeddings, dtype="float32")

# Build FAISS index
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Search disease knowledge
def search_knowledge(question, top_k=3):
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(documents))
    )

    results = []

    for idx in indices[0]:
        if idx >= 0:
            results.append(documents[idx])

    return results
from dotenv import load_dotenv
from google import genai
import os

# Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env file")

client = genai.Client(api_key=api_key)

def generate_answer(question):

    results = search_knowledge(question)
    context = "\n\n".join(results)

    prompt = f"""
    You are TomatoCare AI, a tomato plant disease assistant.

    Answer using the knowledge below.
    If the knowledge does not contain the answer, say so.
    Do not invent treatments or diagnoses.

    Knowledge:
    {context}

    User question:
    {question}
    """

    print("Sending question to Gemini...")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    print("Gemini response received!")

    return response.text