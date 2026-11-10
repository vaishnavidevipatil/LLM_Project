
import requests
import numpy as np
from pathlib import Path
import re

OLLAMA = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

# Use the actual folder name: doc or docs
DOCS_DIR = Path(__file__).resolve().parent / "doc"


# 1. Generate embeddings
def embed(texts):

    if not texts:
        raise ValueError("Cannot generate embeddings: text list is empty")

    response = requests.post(
        f"{OLLAMA}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    return np.array(embeddings, dtype=np.float32)


# 2. Ask LLM
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


# 3. Load Markdown documents and create chunks
def load_documents():

    chunks = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Document folder not found: {DOCS_DIR}"
        )

    files = list(DOCS_DIR.glob("*.md"))

    print(f"Found {len(files)} Markdown files")

    for file in files:

        text = file.read_text(encoding="utf-8")

        # Split document into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)

        file_chunks = [
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ]

        for chunk in file_chunks:
            chunks.append(chunk)

        print(f"Loaded: {file.name}")
        print(f"Chunks: {len(file_chunks)}")

    return chunks


# 4. Load documents and generate vectors
chunks = load_documents()

if not chunks:
    raise ValueError(
        "No chunks found. Check your doc folder and Markdown files."
    )

chunk_vectors = embed(chunks)

print("\n========== INDEXING COMPLETE ==========")
print(f"Indexed chunks: {len(chunks)}")
print(f"Embedding shape: {chunk_vectors.shape}")


# 5. Normalize vectors for cosine similarity
chunk_norms = np.linalg.norm(chunk_vectors, axis=1)

if np.any(chunk_norms == 0):
    raise ValueError("One or more document embeddings are zero vectors")

chunk_vectors = chunk_vectors / chunk_norms[:, None]


# 6. Retrieve relevant chunks
def retrieve(question, k=3):

    question_vector = embed([question])[0]

    question_norm = np.linalg.norm(question_vector)

    if question_norm == 0:
        raise ValueError("Question embedding is a zero vector")

    question_vector = question_vector / question_norm

    scores = chunk_vectors @ question_vector

    top_k_indices = np.argsort(scores)[::-1][:k]

    print("\nRelevant chunk scores:")

    for index in top_k_indices:
        print(f"Chunk {index}: {scores[index]:.4f}")

    return [chunks[i] for i in top_k_indices]


# 7. Ask questions using retrieved context
def get_llm_response(question):

    retrieved_chunks = retrieve(question)

    context = "\n\n".join(retrieved_chunks)

    prompt = (
        "You are a helpful assistant. Answer the question "
        "using only the provided context.\n"
        "If the answer is not present in the context, "
        "say that the document does not contain the answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    return ask_llm(prompt)


# 8. Main application
if __name__ == "__main__":

    print("\n========== RAG APPLICATION ==========")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nAsk a question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("RAG application stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = get_llm_response(question)

            print("\n========== LLM RESPONSE ==========")
            print(answer)

        except requests.RequestException as error:
            print(f"Ollama API error: {error}")

        except Exception as error:
            print(f"Error: {error}")
