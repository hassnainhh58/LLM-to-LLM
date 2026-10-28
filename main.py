# main.py
import os
from dotenv import load_dotenv
from agent import run_agent

# Load variables from .env
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY is missing from .env")
if not LLM_MODEL:
    raise ValueError("LLM_MODEL is missing from .env")


def main():
    print("================================")
    print("    AI STUDY ASSISTANT")
    print("================================")
    print()

    message_history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            print("Thinking...")
            output_data, message_history = run_agent(question, message_history)

            print()
            print("AI:")
            print(f"Answer: {output_data.get('answer')}")
            print(f"Route: {output_data.get('route')}")
            print(f"Topic: {output_data.get('topic')}")
            print(f"Difficulty: {output_data.get('difficulty')}")
            print(f"Used calculator: {output_data.get('used_calculator')}")
            print(f"Used weather: {output_data.get('used_weather')}")
            print(f"Used web search: {output_data.get('used_web_search')}")
            print()

        except Exception as error:
            print()
            print("Something went wrong:")
            print(repr(error))
            print()


if __name__ == "__main__":
    main()
    