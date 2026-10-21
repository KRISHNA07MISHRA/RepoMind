
from vector_store import index_repository

if __name__ == "__main__":
    result = index_repository(
        "https://github.com/psf/requests",
        max_files=10,
    )

    print("Repository:", result["repository"])
    print("Branch:", result["branch"])
    print("Files fetched:", result["files_fetched"])
    print("Chunks indexed:", result["chunks_indexed"])
    print("Fetch errors:", result["fetch_errors"])
    print("Qdrant indexing completed!")
