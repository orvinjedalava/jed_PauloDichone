from dotenv import load_dotenv
from anthropic import Anthropic
import voyageai

load_dotenv()


def generate_embedding(client, text, model="voyage-3-large", input_type="query"):
    result = client.embed([text], model=model, input_type=input_type)
    return result.embeddings[0]


def main():
    client = Anthropic()
    voyage_client = voyageai.Client()

    text = "The quick brown fox jumps over the lazy dog."
    embedding = generate_embedding(voyage_client, text)
    print("Generated Embedding:", embedding)


if __name__ == "__main__":
    main()
