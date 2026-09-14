from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import ToolParam, ToolResultBlockParam, MessageParam, Message
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json
from datetime import datetime

load_dotenv()

class ProductReview(BaseModel):
    title: str = Field(description="Concise 3-8 word title capturing the review's essence")
    review_text: str = Field(description="Cleaned, normalized review text")
    rating: int = Field(ge=1, le=5, description="Rating 1-5, converted from other formats")
    author: int = Field(default="Anonymous")
    product_name: str = Field(description="Specific product being reviewed, extracted from the text")
    sentiment: Literal["positive", "negative","neutral","mixed"]
    would_recommend: bool = Field(description="Whether the reviewer would recommend this product, inferred from the review content")
    key_features_mentioned: list[str] = Field(default_factory=list, description="List of specific product features or aspects mentioned in the review (e.g., 'battery life', 'sound quality', 'price')")
    price_mentioned: Optional[float] = Field(default=None, description="Price mentioned in the review, if any. Extract the numeric value only")
    pros: list[str] = Field(default_factory=list, description="List of positive aspects or advantages mentioned.")
    cons: list[str] = Field(default_factory=list, description="List of negative aspects or drawbacks mentioned")

product_review_schema_prompt: str = """
    Extract and enrich product review information from messy, unstructured text. 
    This tool handles real-world reviews with informal language, emojis, slang, and various rating formats.
    It normalizes data, infers missing information, and enriches the review with additional insights.
"""

# ENHANCED product review tool schema with data enrichment
product_review_schema: ToolParam = ToolParam (
    name="extract_product_review",
    description= product_review_schema_prompt,
    input_schema=ProductReview.model_json_schema()
)

def process_tool_calls(client: Anthropic, messages: list[MessageParam], response: Message):
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
                    review: ProductReview = ProductReview.model_validate(content.input)
                    structured_review: dict[str, any] = review.model_dump()
                    review_json_dump: str = json.dumps(structured_review)

                    print(f"\n✅ Review structured and enriched successfully!")
                    print(f"\n💾 Structured Output:")
                    print(review_json_dump)

                    # Prepare tool result for response
                    tool_results.append(ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=content.id,
                        content=review_json_dump)
                    )

        # Send tool results back to the model
        if tool_results:
            # Add assistant's response to messages
            messages.append(MessageParam(
                role="assistant",
                content=response.content
            ))

            # Add tool results
            messages.append(MessageParam(
                role="user",
                content=tool_results
            ))

            followup_response: Message = client.messages.create(
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

    response: Message = client.messages.create(
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
