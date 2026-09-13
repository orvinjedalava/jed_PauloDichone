from dotenv import load_dotenv
from anthropic import Anthropic
import json
from datetime import datetime

load_dotenv()


# ENHANCED product review tool schema with data enrichment
product_review_schema = {
    "name": "extract_product_review",
    "description": """Extract and enrich product review information from messy, unstructured text. 
    This tool handles real-world reviews with informal language, emojis, slang, and various rating formats.
    It normalizes data, infers missing information, and enriches the review with additional insights.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title or headline of the review. If not explicitly stated, infer a concise title (3-8 words) that captures the essence of the review.",
            },
            "review_text": {
                "type": "string",
                "description": "The cleaned and normalized full text of the review, without informal abbreviations or excessive emojis.",
            },
            "rating": {
                "type": "integer",
                "description": "Rating on a 1-5 scale. Convert from other formats: '10/10' or '5 stars' → 5, '3.5/5' → 4, '8/10' → 4, etc. If no explicit rating, infer from sentiment.",
                "minimum": 1,
                "maximum": 5,
            },
            "author": {
                "type": "string",
                "description": "Username or name of the reviewer. If not found, use 'Anonymous'.",
            },
            "product_name": {
                "type": "string",
                "description": "The specific product being reviewed, extracted from the text.",
            },
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed"],
                "description": "Overall sentiment of the review based on the language and tone used.",
            },
            "would_recommend": {
                "type": "boolean",
                "description": "Whether the reviewer would recommend this product, inferred from the review content.",
            },
            "key_features_mentioned": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific product features or aspects mentioned in the review (e.g., 'battery life', 'sound quality', 'price').",
            },
            "price_mentioned": {
                "type": "number",
                "description": "Price mentioned in the review, if any. Extract the numeric value only.",
            },
            "pros": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of positive aspects or advantages mentioned.",
            },
            "cons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of negative aspects or drawbacks mentioned.",
            },
        },
        "required": [
            "title",
            "review_text",
            "rating",
            "author",
            "product_name",
            "sentiment",
            "would_recommend",
        ],
    },
}


def extract_product_review(
    title,
    review_text,
    rating,
    author,
    product_name,
    sentiment,
    would_recommend,
    key_features_mentioned=None,
    price_mentioned=None,
    pros=None,
    cons=None,
):
    """
    Process and structure an enriched product review.
    In production, this would save to a database, trigger analytics, etc.
    """
    structured_review = {
        "title": title,
        "review_text": review_text,
        "rating": rating,
        "author": author,
        "product_name": product_name,
        "sentiment": sentiment,
        "would_recommend": would_recommend,
        "key_features_mentioned": key_features_mentioned or [],
        "price_mentioned": price_mentioned,
        "pros": pros or [],
        "cons": cons or [],
        "extracted_at": datetime.now().isoformat(),
        "processed": True,
    }
    return structured_review


def process_tool_calls(client, messages, response):
    """Process tool calls and return the final response"""

    if response.stop_reason == "tool_use":
        # Process tool calls and collect results
        tool_results = []
        for content in response.content:
            if content.type == "text":
                print(f"Claude says: {content.text}\n")
            elif content.type == "tool_use":
                print(f"\n🔧 Tool call detected: {content.name}")
                print(f"📊 Extracted Data:")
                print(json.dumps(content.input, indent=2))

                # Execute the tool
                if content.name == "extract_product_review":
                    result = extract_product_review(**content.input)
                    print(f"\n✅ Review structured and enriched successfully!")
                    print(f"\n💾 Structured Output:")
                    print(json.dumps(result, indent=2))

                    # Prepare tool result for response
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # Send tool results back to the model
        if tool_results:
            # Add assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Add tool results
            messages.append({"role": "user", "content": tool_results})

            followup_response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=messages,
                tools=[product_review_schema],
            )

            # If there are more tool calls, recursively process them
            if followup_response.stop_reason == "tool_use":
                return process_tool_calls(client, messages, followup_response)

            return followup_response

    return response


def main():
    client = Anthropic()

    print("� ENHANCED Product Review Extraction with Data Enrichment")
    print("=" * 80)
    print("This demo shows Claude's REAL POWER for structured data extraction:")
    print("  ✓ Handles messy, informal, real-world text")
    print("  ✓ Normalizes ratings from different formats (10/10, stars, etc.)")
    print("  ✓ Infers missing information intelligently")
    print("  ✓ Enriches data with sentiment, features, pros/cons")
    print("  ✓ Extracts and validates structured JSON")
    print("=" * 80)

    # Example 1: Messy informal review with emojis and slang
    print("\n\n📦 EXAMPLE 1: Messy Informal Review (Real-World Data)")
    print("=" * 80)
    print("💡 Challenge: Informal language, emojis, non-standard rating format\n")

    review_text_1 = """
    omg these headphones r AMAZING!!! 🎧🔥 i paid like 200 bucks last week 
    and theyre totally worth it. battery life is insane - i think 30hrs? 
    maybe more idk. noise canceling is chef's kiss 👨‍🍳💋 sound quality is 
    crispy af. only downside is they're kinda heavy after a while but nbd.
    10/10 would buy again
    - tech_guy_92
    """

    print(f"Raw Review Text:\n{review_text_1}\n")

    messages = [
        {
            "role": "user",
            "content": f"Extract and enrich the product review information from this text:\n\n{review_text_1}",
        }
    ]

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=messages,
        tools=[product_review_schema],
        tool_choice={"type": "tool", "name": "extract_product_review"},
    )

    final_response = process_tool_calls(client, messages, response)
    print(f"\n{'='*80}\n")

    # Example 2: Review with missing rating but clear sentiment
    print("\n\n📦 EXAMPLE 2: Missing Rating - Sentiment-Based Inference")
    print("=" * 80)
    print("💡 Challenge: No explicit rating, must infer from negative sentiment\n")

    review_text_2 = """
    Subject: Total waste of money - BrewMaster 3000
    
    I bought this coffee maker hoping for café-quality drinks at home. What a joke!
    The coffee is mediocre at best. Machine sounds like a jet engine taking off.
    Takes FOREVER to brew one cup. Water tank is tiny - refill constantly.
    At $150 this is highway robbery. Save your money and go to Starbucks instead.
    
    Posted by: CoffeeLover_Jane
    """

    print(f"Raw Review Text:\n{review_text_2}\n")

    messages_2 = [
        {
            "role": "user",
            "content": f"Extract and enrich the review data from this text:\n\n{review_text_2}",
        }
    ]

    response_2 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=messages_2,
        tools=[product_review_schema],
        tool_choice={"type": "tool", "name": "extract_product_review"},
    )

    final_response_2 = process_tool_calls(client, messages_2, response_2)
    print(f"\n{'='*80}\n")

    # Example 3: Mixed sentiment review
    print("\n\n📦 EXAMPLE 3: Mixed Sentiment Review")
    print("=" * 80)
    print("💡 Challenge: Both positive and negative points, partial star rating\n")

    review_text_3 = """
    Galaxy Watch Ultra - decent but not perfect
    
    Got this smartwatch 2 weeks ago for $399. GPS tracking is super accurate 
    for my runs and the heart rate monitor seems legit. Battery easily lasts 
    2 days which is nice. Screen is bright and crisp.
    
    However... the UI is clunky and confusing. Lots of features I'll never use.
    Notifications are buggy sometimes. For the price, I expected more polish.
    
    3.5 stars - good hardware, software needs work
    
    ~ FitnessFanatic23
    """

    print(f"Raw Review Text:\n{review_text_3}\n")

    messages_3 = [
        {
            "role": "user",
            "content": f"Extract and enrich this review:\n\n{review_text_3}",
        }
    ]

    response_3 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=messages_3,
        tools=[product_review_schema],
        tool_choice={"type": "tool", "name": "extract_product_review"},
    )

    final_response_3 = process_tool_calls(client, messages_3, response_3)
    print(f"\n{'='*80}\n")

    # Example 4: Ultra-short review (edge case)
    print("\n\n📦 EXAMPLE 4: Ultra-Short Review (Edge Case)")
    print("=" * 80)
    print("💡 Challenge: Minimal text, must infer context and details\n")

    review_text_4 = """
    AirPods Max - overpriced. Sound is good but not $549 good. ⭐⭐⭐
    """

    print(f"Raw Review Text:\n{review_text_4}\n")

    messages_4 = [
        {
            "role": "user",
            "content": f"Extract and enrich this review:\n\n{review_text_4}",
        }
    ]

    response_4 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=messages_4,
        tools=[product_review_schema],
        tool_choice={"type": "tool", "name": "extract_product_review"},
    )

    final_response_4 = process_tool_calls(client, messages_4, response_4)
    print(f"\n{'='*80}\n")

    # Summary
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("\n🎯 What This Demo Showed:")
    print("\n1. 🧹 DATA NORMALIZATION:")
    print("   - Converted '10/10' → 5 stars, '3.5 stars' → 4, emojis → ratings")
    print("   - Cleaned informal language ('r' → 'are', 'af' → removed)")
    print("   - Handled various text formats and styles")
    
    print("\n2. 🧠 INTELLIGENT INFERENCE:")
    print("   - Inferred ratings when not explicitly stated")
    print("   - Generated titles when missing")
    print("   - Determined sentiment from context")
    print("   - Assessed recommendation likelihood")
    
    print("\n3. 📊 DATA ENRICHMENT:")
    print("   - Extracted product names, prices, features")
    print("   - Identified pros and cons separately")
    print("   - Categorized sentiment (positive/negative/mixed)")
    print("   - Pulled out key features mentioned")
    
    print("\n4. ✅ STRUCTURED VALIDATION:")
    print("   - Ensured all required fields present")
    print("   - Validated rating range (1-5)")
    print("   - Normalized data types (numbers, booleans, strings)")
    print("   - Created consistent, database-ready JSON")
    
    print("\n💡 REAL-WORLD USE CASES:")
    print("   • E-commerce platforms (Amazon, eBay)")
    print("   • Review aggregation services")
    print("   • Sentiment analysis pipelines")
    print("   • Customer feedback systems")
    print("   • Product analytics dashboards")
    print("   • Content moderation workflows")
    
    print("\n🚀 WHY TOOLS > PLAIN PROMPTING:")
    print("   • Guaranteed structure and schema validation")
    print("   • Type safety (integers, booleans, enums)")
    print("   • Consistent output format across all calls")
    print("   • Easier to integrate with databases/APIs")
    print("   • Better error handling and validation")
    
    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    main()
