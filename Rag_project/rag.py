
import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"


# Generate embeddings
def embed(texts):
    response = requests.post(
        f"{OLLAMA}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=60
    )

    response.raise_for_status()

    return np.array(response.json()["embeddings"])


# Ask LLM
def ask_llm(prompt):
    response = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# Test LLM
# answer = embed("What is the capital of India?")
# print("Embedding Response:")
# # print("LLM Response:")
# print(answer)

chunks = []
for file in Path("docs").glob("*.md"):
    chunks += [p.strip() for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()]

chun_vectors = embed(chunks)
# print(f"Embedding Response: {chun_vectors.shape[0]} vectors")
print(f"Indexed {len(chunks)} chunks \n")

def retrieve(question, k=3):
    question_vector = embed([question])[0]
    scores = chunk_vectors @ question_vector / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(question_vector))
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_k_indices]

#Calling the LLM with the retrieved context
# def get_llm_response(question):

while True:
    question = input("Ask a question: ")
    context = "\n\n".join(retrieve(question))
    prompt = (
        "Answer the question based on the context below:\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    print(f"LLM Response: {ask_llm(prompt)}")
