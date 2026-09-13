from dotenv import load_dotenv
from anthropic import Anthropic
import json

load_dotenv()

from datetime import datetime, timezone


get_current_datetime_schema = {
    "name": "get_current_datetime",
    "description": "Get the current date and time in a specified timezone with configurable output format. Returns datetime string, timezone, and unix timestamp.",
    "input_schema": {
        "type": "object",
        "properties": {
            "timezone_str": {
                "type": "string",
                "enum": ["UTC", "EST", "PST"],
                "description": "The timezone to return the current time in. Supported values: UTC (Coordinated Universal Time), EST (Eastern Standard Time, UTC-5), PST (Pacific Standard Time, UTC-8)",
                "default": "UTC",
            },
            "format": {
                "type": "string",
                "enum": ["iso", "readable"],
                "description": "Output format for the datetime string. 'iso' returns ISO 8601 format (e.g., 2024-01-15T14:30:00), 'readable' returns human-friendly format (e.g., January 15, 2024 at 2:30 PM)",
                "default": "iso",
            },
        },
        "required": [],
    },
}


def get_current_datetime(timezone_str="UTC", format="iso"):
    """
    Get the current date and time in specified timezone and format
    """
    # Get current time in UTC
    now_utc = datetime.now(timezone.utc)

    # For simplicity, we'll handle just a few timezones
    # In production, use pytz library
    if timezone_str == "UTC":
        current_time = now_utc
    elif timezone_str == "EST":
        current_time = now_utc.replace(hour=(now_utc.hour - 5) % 24)
    elif timezone_str == "PST":
        current_time = now_utc.replace(hour=(now_utc.hour - 8) % 24)
    else:
        current_time = now_utc

    # Format the output
    if format == "iso":
        return {
            "datetime": current_time.isoformat(),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }
    elif format == "readable":
        return {
            "datetime": current_time.strftime("%B %d, %Y at %I:%M %p"),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }
    else:
        return {
            "datetime": str(current_time),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }


def process_tool_calls(client, messages, response):
    """Process tool calls and return the final response"""

    if response.stop_reason == "tool_use":
        # Process tool calls and collect results
        tool_results = []
        for content in response.content:
            if content.type == "text":
                print(f"Claude says: {content.text}")
            elif content.type == "tool_use":
                print(f"\n🔧 Tool call detected: {content.name}")
                print(f"Parameters: {content.input}")

                # Execute the tool
                if content.name == "get_current_datetime":
                    result = get_current_datetime(**content.input)
                    print(f"Result: {result}\n")

                    # Prepare tool result for response
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # Send tool results back to the model (OUTSIDE the loop)
        if tool_results:
            # Add assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Add tool results
            messages.append({"role": "user", "content": tool_results})

            followup_response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=messages,
                tools=[get_current_datetime_schema],
            )

            # If there are more tool calls, recursively process them
            if followup_response.stop_reason == "tool_use":
                return process_tool_calls(client, messages, followup_response)

            return followup_response

    return response


def main():
    client = Anthropic()
    messages = []

    print("🕐 Interactive Time Zone Chatbot")
    print("Ask me about the current time in different places!")
    print("Supported timezones: UTC, EST (Eastern), PST (Pacific/California)")
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

        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})

        print(f"\n{'='*60}")

        # Get initial response from Claude
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=messages,
            tools=[get_current_datetime_schema],
        )

        # Process any tool calls and get final response
        final_response = process_tool_calls(client, messages, response)

        # Display the final answer
        print(f"{'='*60}")
        print("Assistant: ", end="")
        for content in final_response.content:
            if content.type == "text":
                print(content.text)
        print(f"{'='*60}\n")

        # Add assistant's final response to conversation history
        if final_response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": final_response.content})


if __name__ == "__main__":
    main()
