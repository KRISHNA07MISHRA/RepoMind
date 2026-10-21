
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_chat_model():
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "GEMINI_API_KEY missing. Check your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )


if __name__ == "__main__":
    model = get_chat_model()

    response = model.invoke(
        "Explain in one sentence what an HTTP request is."
    )

    print(response.text)
