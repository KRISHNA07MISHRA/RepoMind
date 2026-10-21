
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from embeddings import get_embedding_model

from pathlib import Path

DB_DIRECTORY = Path(__file__).parent / "qdrant_db"


def main():
    embedding_model = get_embedding_model()

    documents = [
        Document(
            page_content=(
                "def login_user(email, password):\n"
                "    return authenticate(email, password)"
            ),
            metadata={"file_path": "src/auth.py"},
        ),
        Document(
            page_content=(
                "def calculate_total(price, quantity):\n"
                "    return price * quantity"
            ),
            metadata={"file_path": "src/cart.py"},
        ),
    ]

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        path=str(DB_DIRECTORY),
        collection_name="repomind_test",
    )

    results = vector_store.similarity_search(
        "Where is user authentication handled?",
        k=2,
    )

    print("Retrieved documents:", len(results))

    for document in results:
        print("\nFile:", document.metadata["file_path"])
        print(document.page_content)


if __name__ == "__main__":
    main()
