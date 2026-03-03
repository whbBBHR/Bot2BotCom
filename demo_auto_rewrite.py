#!/usr/bin/env python3
"""
Demo of the AUTO-REWRITE feature.
Shows how AI suggestions are automatically extracted and offered as an option.
"""

import anthropic
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def extract_suggested_question(ai_feedback):
    """Extract the suggested improved question from AI feedback."""
    if not ai_feedback:
        return None

    # Look for common patterns for suggestions
    patterns = [
        r'Suggested improved version[:\s]+["\']([^"\']+)["\']',
        r'Suggested improved version[:\s]+([^\n]+)',
        r'[Bb]etter version[:\s]+["\']([^"\']+)["\']',
        r'[Rr]ewrite as[:\s]+["\']([^"\']+)["\']',
        r'[Tt]ry[:\s]+["\']([^"\']+)["\']',
        r'"([^"]{20,})"',  # Any quoted text longer than 20 chars
    ]

    for pattern in patterns:
        match = re.search(pattern, ai_feedback, re.IGNORECASE)
        if match:
            suggestion = match.group(1).strip()
            suggestion = suggestion.rstrip('.')
            if len(suggestion) > 10 and len(suggestion) < 500:
                return suggestion

    return None

def demo_auto_rewrite(question):
    """Demo the auto-rewrite feature."""

    print("="*80)
    print("AUTO-REWRITE FEATURE DEMO")
    print("="*80)
    print()
    print(f"Original question: \"{question}\"")
    print()

    # Get AI review
    print("🤖 Getting AI clarity review...")

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Review this question for clarity and effectiveness:

"{question}"

Provide:
1. Clarity score (1-10) with brief explanation
2. Is it specific enough? (yes/no with explanation)
3. Suggested improved version (ONLY if needed - if the question is already good, say "No improvement needed")

If you suggest an improvement, format it clearly in quotes like:
Suggested improved version: "Your improved question here"

Keep your response concise and actionable."""
            }]
        )

        # Extract text from response
        ai_feedback = ""
        for block in response.content:
            if block.type == 'text':
                ai_feedback += block.text

        print()
        print("📊 AI Clarity Review:")
        print("-"*80)
        print(ai_feedback)
        print("-"*80)
        print()

        # Extract suggested improvement
        suggested_question = extract_suggested_question(ai_feedback)

        # Show options
        print("="*80)
        print("OPTIONS:")
        print("  [1] Use original question")

        if suggested_question:
            print("  [2] Use AI's suggested improvement (auto-rewrite) ✨")
            print("  [3] Edit manually and recheck")
            print("  [4] Cancel (return to main menu)")
            print("="*80)
            print()
            print(f"💡 AI Suggestion: \"{suggested_question}\"")
            print()
            print("✅ AUTO-REWRITE AVAILABLE!")
            print("   → Simply choose option [2] to use the AI's rewritten version")
            print("   → No need to manually copy/paste!")
        else:
            print("  [2] Edit manually and recheck")
            print("  [3] Cancel (return to main menu)")
            print("="*80)
            print()
            print("ℹ️  No rewrite suggested (question is already good)")

        return suggested_question

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print()
    print("="*80)
    print("DEMONSTRATING AUTO-REWRITE FEATURE")
    print("="*80)
    print()
    print("This shows how the AI's suggested improvement is automatically")
    print("extracted and offered as a one-click option!")
    print()

    # Test with a poor question that will get a suggestion
    test_questions = [
        "What is AI",  # Too vague - should get suggestion
        "What are the key differences between Claude and GPT-4o?",  # Good - may not need suggestion
    ]

    for i, question in enumerate(test_questions, 1):
        print()
        print("="*80)
        print(f"TEST {i}/2")
        print("="*80)
        print()

        suggested = demo_auto_rewrite(question)

        print()
        print("─"*80)
        print("RESULT:")
        if suggested:
            print(f"  Original:   \"{question}\"")
            print(f"  Suggested:  \"{suggested}\"")
            print()
            print("  ✅ Auto-rewrite would apply this with option [2]")
        else:
            print(f"  Question:   \"{question}\"")
            print("  ℹ️  No rewrite needed (question is good)")
        print("─"*80)

        if i < len(test_questions):
            print()
            input("Press Enter to continue to next test...")

    print()
    print("="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print()
    print("✨ NEW FEATURE: Auto-Rewrite")
    print()
    print("How it works:")
    print("  1. AI reviews your question")
    print("  2. If improvement needed, AI suggests a rewrite")
    print("  3. Suggestion is automatically extracted")
    print("  4. You choose [2] to instantly use it - no copying!")
    print()
    print("Benefits:")
    print("  ✅ No manual copying/pasting")
    print("  ✅ One-click to apply AI's suggestion")
    print("  ✅ Preserves original if you prefer it")
    print("  ✅ Can still manually edit if you want")
    print()
    print("To use in the main script:")
    print("  python API_2_Api_com.py")
    print()
