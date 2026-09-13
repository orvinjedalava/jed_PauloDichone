from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()


def main():
    client = Anthropic()
    print("\n🌐 Claude Web Search Demo")
    print("=" * 60)
    print("Ask Claude a question that requires current information!\n")

    # Get user's question
    question = input("Your question: ").strip()

    if not question:
        question = "What's the weather in NYC?"
        print(f"Using default question: {question}")

    print("\n⏳ Searching the web...\n")

    # Make the API call with web search enabled (matching docs example)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
    )

    # Display the full response for debugging
    print("=" * 60)
    print("📋 Full Response:")
    print("=" * 60)
    print(response)
    print("\n" + "=" * 60)

    # Display Claude's formatted answer
    print("📝 Claude's Answer:\n")

    for content in response.content:
        if content.type == "text":
            print(content.text)

            # Show citations if available
            if hasattr(content, "citations") and content.citations:
                print("\n📚 Sources:")
                for citation in content.citations:
                    print(f"  • {citation.title}")
                    print(f"    {citation.url}\n")

    # Show usage stats
    print("\n" + "=" * 60)
    print("💰 Usage:")
    print(f"  Input tokens: {response.usage.input_tokens}")
    print(f"  Output tokens: {response.usage.output_tokens}")
    if hasattr(response.usage, "server_tool_use") and response.usage.server_tool_use:
        stu = response.usage.server_tool_use
        # Access as attribute, not dictionary
        searches = getattr(stu, "web_search_requests", 0)
        print(f"  Web searches: {searches}")
        print(f"  Search cost: ${searches * 0.01:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
