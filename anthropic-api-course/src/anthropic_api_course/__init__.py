from dotenv import load_dotenv
load_dotenv()

# create an API client
from anthropic import Anthropic
from anthropic.types import Message


def main() -> None:
    client = Anthropic()

    message: Message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Write a short poem about the sea."
            }
        ]
    )
    print(message.content)
