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
    """Stream the response from the LLM and return the complete text"""
    client = Anthropic()
    full_response = ""

    print("Assistant: ", end="", flush=True)
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(
                text, end="", flush=True
            )  # Stream each chunk immediately on same line
            full_response += text

    print("\n")  # Add newline after streaming completes
    return full_response


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

        # Get response from the chatbot (streaming)
        print("🤔 Thinking...\n")
        response = chat(messages)

        # Add assistant's response to the conversation history
        add_assistant_message(messages, response)


if __name__ == "__main__":
    main()
