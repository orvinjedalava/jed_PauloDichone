# load env variables from .env file
from dotenv import load_dotenv

load_dotenv()

# create an API client
from anthropic import Anthropic


def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})


def chat(messages):

    system_prompt = """ 
    You are an expert Systems Design Mentor sepecializing in software architecture and distributed systems.
    Do not directly answer the user's questions. Instead, guide them through the process of designing systems by asking probing questions, providing hints, and encouraging critical thinking.
    Guide them to a solution step by step.
    """
    client = Anthropic()
    response = client.messages.create(
        system=system_prompt,
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


def display_messages(messages):
    """Display all messages being sent to the LLM"""
    print("\n" + "=" * 60)
    print("MESSAGES BEING SENT TO LLM:")
    print("=" * 60)
    for i, msg in enumerate(messages, 1):
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[Message {i} - {role}]")
        print(f"{content}")
    print("=" * 60 + "\n")


def main():
    messages = []

    print("🤖 Simple Console Chatbot")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Check if user wants to exit
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n👋 Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Add user message to the conversation
        add_user_message(messages, user_input)

        # Display all messages being sent to the LLM
        display_messages(messages)

        # Get response from the chatbot
        print("🤔 Thinking...\n")
        response = chat(messages)

        # Display assistant's response
        print(f"Assistant: {response}\n")

        # Add assistant's response to the conversation history
        add_assistant_message(messages, response)


if __name__ == "__main__":
    main()
