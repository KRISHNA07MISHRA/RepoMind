
from langchain_core.prompts import ChatPromptTemplate

from llm import get_chat_model
from vector_store import create_vector_store


PROMPT = ChatPromptTemplate.from_template("""
You are RepoMind, a repository-aware coding assistant.

Answer the user's question using only the repository context below.

Rules:
- Explain the implementation clearly.
- Do not invent behavior that is not supported by the context.
- If the context is insufficient, say what information is missing.
- Cite supporting files using this format:
  [file_path: start_line-end_line]
- Distinguish facts from reasonable inferences.

Repository context:
{context}

User question:
{question}
""")


def answer_question(question, k=5,vector_store = None):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if vector_store is None:
        vector_store = create_vector_store()

    documents = vector_store.similarity_search(
        query=question,
        k=k,
    )

    if not documents:
        return "I couldn't find relevant code in the indexed repository."

    context_parts = []

    for document in documents:
        file_path = document.metadata.get("file_path", "unknown")
        start_line = document.metadata.get("start_line", "?")
        end_line = document.metadata.get("end_line", "?")

        context_parts.append(
            f"Source: {file_path}, lines {start_line}-{end_line}\n"
            f"{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    model = get_chat_model()
    chain = PROMPT | model

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response.text



if __name__ == "__main__":
    from vector_store import index_repository

    github_url = input("Enter GitHub repository URL: ").strip()
    question = input("Enter your question: ").strip()

    if not github_url or not question:
        raise ValueError("Repository URL and question are required.")

    vector_store = None

    try:
        print("\nIndexing repository...")
        result = index_repository(github_url)
        vector_store = result["vector_store"]

        print(f"Repository: {result['repository']}")
        print(f"Files fetched: {result['files_fetched']}")
        print(f"Chunks indexed: {result['chunks_indexed']}")

        print("\nGenerating answer...")
        answer = answer_question(
            question,
            vector_store=vector_store,
        )

        print("\nRepoMind Answer:\n")
        print(answer)

    except Exception as error:
        print(f"\nError: {error}")

    finally:
        if vector_store is not None:
            vector_store.client.close()
