
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def get_embedding_model():
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "GEMINI_API_KEY missing. Check your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
    )


def embed_code_chunks(texts):
    if not texts:
        return []

    embedding_model = get_embedding_model()

    return embedding_model.embed_documents(
        texts,
        task_type="RETRIEVAL_DOCUMENT",
    )


def embed_question(question):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    embedding_model = get_embedding_model()

    return embedding_model.embed_query(
        question,
        task_type="RETRIEVAL_QUERY",
    )


if __name__ == "__main__":
    model = get_embedding_model()

    sample_code = [
        "def login_user(email, password):\n"
        "    return authenticate(email, password)",
        "def calculate_total(price, quantity):\n"
        "    return price * quantity",
    ]

    document_vectors = embed_code_chunks(sample_code)

    query_vector = embed_question(
        "Where is user authentication handled?"
    )

    print("Number of code chunks:", len(document_vectors))
    print("Code vector dimensions:", len(document_vectors[0]))
    print("Question vector dimensions:", len(query_vector))
    print("Embedding test successful!")
