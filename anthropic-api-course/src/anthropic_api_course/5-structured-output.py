from dotenv import load_dotenv
from anthropic import Anthropic
import json

load_dotenv()


def main():
    client = Anthropic()

    print("🚀 Generating CloudFormation Template for EC2 Instance\n")
    print("=" * 70)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": "Generate a CloudFormation JSON template that provisions an EC2 instance. Return ONLY the raw JSON without any markdown formatting, code blocks, or explanations.",
            },
        ],
        stop_sequences=["\n\nNote:", "\n\nThis", "\n\nHere"],
    )

    cloudformation_json = response.content[0].text

    print("CloudFormation Template (Clean JSON):")
    print("=" * 70)
    print(cloudformation_json)
    print("=" * 70)

    # Validate the JSON
    try:
        parsed = json.loads(cloudformation_json)
        print("\n✅ Valid JSON! Ready to copy and paste.")
        print(f"\nResources in template: {list(parsed.get('Resources', {}).keys())}")
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")


if __name__ == "__main__":
    main()
