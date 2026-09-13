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
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


def main():
    messages = []
    add_user_message(messages, "Write a short poem about the sea.")
    print("Messages;;;;;", messages)

    response = chat(messages)
    print("Assistant:", response)

    # Take the answer from the assistant and add it to the assistant messages
    add_assistant_message(messages, response)

    # Now, ask a follow-up question
    add_user_message(messages, "Can you make it rhyme?")
    print("Messages;;;;;", messages)

    # Call the chat function again with the updated messages
    response = chat(messages)
    print("Final Assistant Response:", response)


if __name__ == "__main__":
    main()