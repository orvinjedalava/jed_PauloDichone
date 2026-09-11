from dotenv import load_dotenv
load_dotenv()

# create an API client
from anthropic import Anthropic
from anthropic.types import Message, MessageParam
from typing import Iterable

def add_user_message(messages: list[MessageParam], content: str):
    add_message(messages, content, "user")

def add_assistant_message(messages: list[MessageParam], content: str):
    add_message(messages, content, "assistant")

def add_message(messages: list[MessageParam], content: str, role: str):
    new_message: MessageParam = {
        "role": role,
        "content": content
    }
    messages.append(new_message)

def chat(messages: Iterable[MessageParam])-> str:
    client = Anthropic()
    response: Message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text    


def main() -> None:
    messages = []
    add_user_message(messages, "Write a short poem about the sea.")
    print("Messages;;;;;", messages)

    response : str = chat(messages)
    print("Assistant:", response)

    # Take the answer from the assistant and add it to the assistant message
    add_assistant_message(messages, response)

    # Now, ask a follow-up question
    add_user_message(messages, "Can you make it ryhme?")
    print("Messages;;;;;", messages)

    # call the chat function again with the updated messages
    response : str = chat(messages)
    print("Assistant:", response)
