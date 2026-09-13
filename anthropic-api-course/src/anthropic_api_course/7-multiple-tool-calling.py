from dotenv import load_dotenv
from anthropic import Anthropic
import json
import random
from datetime import datetime, timezone

load_dotenv()

# Tool 1: Your existing datetime tool
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

# Tool 2: New dice rolling tool
roll_dice_schema = {
    "name": "roll_dice",
    "description": "Roll one or more dice with a specified number of sides. Returns individual rolls and total.",
    "input_schema": {
        "type": "object",
        "properties": {
            "num_dice": {
                "type": "integer",
                "description": "Number of dice to roll",
                "default": 1,
                "minimum": 1,
                "maximum": 10,
            },
            "sides": {
                "type": "integer",
                "description": "Number of sides on each die",
                "default": 6,
                "enum": [4, 6, 8, 10, 12, 20, 100],
            },
        },
        "required": [],
    },
}


def get_current_datetime(timezone_str="UTC", format="iso"):
    """Get the current date and time in specified timezone and format"""
    now_utc = datetime.now(timezone.utc)

    if timezone_str == "UTC":
        current_time = now_utc
    elif timezone_str == "EST":
        current_time = now_utc.replace(hour=(now_utc.hour - 5) % 24)
    elif timezone_str == "PST":
        current_time = now_utc.replace(hour=(now_utc.hour - 8) % 24)
    else:
        current_time = now_utc

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


def roll_dice(num_dice=1, sides=6):
    """Roll dice and return results"""
    rolls = [random.randint(1, sides) for _ in range(num_dice)]

    return {
        "dice_count": num_dice,
        "sides": sides,
        "rolls": rolls,
        "total": sum(rolls),
        "average": round(sum(rolls) / len(rolls), 2),
        "max_possible": num_dice * sides,
    }


def process_tool_calls(client, messages, response, tools):
    """Process tool calls and return the final response"""

    if response.stop_reason == "tool_use":
        tool_results = []

        for content in response.content:
            if content.type == "text":
                print(f"Claude says: {content.text}")
            elif content.type == "tool_use":
                print(f"\nTool call detected: {content.name}")
                print(f"Parameters: {content.input}")

                # Execute the appropriate tool
                if content.name == "get_current_datetime":
                    result = get_current_datetime(**content.input)
                elif content.name == "roll_dice":
                    result = roll_dice(**content.input)
                else:
                    result = {"error": f"Unknown tool: {content.name}"}

                print(f"Result: {result}\n")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result),
                    }
                )

        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            followup_response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=messages,
                tools=tools,
            )

            # Handle recursive tool calls
            if followup_response.stop_reason == "tool_use":
                return process_tool_calls(client, messages, followup_response, tools)

            return followup_response

    return response


def main():
    client = Anthropic()
    messages = []
    tools = [get_current_datetime_schema, roll_dice_schema]

    print("Interactive Assistant with Multiple Tools")
    print("=" * 50)
    print("I can help you with:")
    print("• Current time in UTC, EST, or PST")
    print("• Rolling dice (d4, d6, d8, d10, d12, d20, d100)")
    print("\nExample queries:")
    print("- 'What time is it in California and roll 2 six-sided dice'")
    print("- 'Roll 3d20 and tell me the time in EST'")
    print("- 'What's the current UTC time?'")
    print("\nType 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print(f"\n{'='*60}")

        # Get initial response
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=messages,
            tools=tools,
        )

        # Process tool calls
        final_response = process_tool_calls(client, messages, response, tools)

        # Display final answer
        print(f"{'='*60}")
        print("Assistant: ", end="")
        for content in final_response.content:
            if content.type == "text":
                print(content.text)
        print(f"{'='*60}\n")

        # Add to conversation history
        if final_response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": final_response.content})


if __name__ == "__main__":
    main()
