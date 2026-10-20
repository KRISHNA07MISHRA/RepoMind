
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY was not found in .env")
        return

    client = genai.Client(api_key=api_key)

    print("Connecting to Gemini...")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="In two short sentences, explain what a Python repository is.",
    )

    print("\nRepoMind AI says:")
    print(response.text)


if __name__ == "__main__":
    main()
