
from vector_store import create_vector_store


def search_code(question, k=3):
    vector_store = create_vector_store()

    results = vector_store.similarity_search(
        query=question,
        k=k,
    )

    return results


if __name__ == "__main__":
    question = "How does the library handle HTTP requests?"

    results = search_code(question)

    print("Question:", question)
    print("Retrieved chunks:", len(results))

    for index, document in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print("File:", document.metadata.get("file_path"))
        print("Lines:",
              document.metadata.get("start_line"),
              "-",
              document.metadata.get("end_line"))
        print(document.page_content[:500])
