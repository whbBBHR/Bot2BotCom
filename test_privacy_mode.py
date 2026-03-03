#!/usr/bin/env python3
"""
Test the privacy-focused mode (no grammar check, only Claude AI review).
This simulates what happens when grammar check is disabled.
"""

import anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def test_privacy_mode(question):
    """Test the privacy-focused review mode."""

    print("="*80)
    print("PRIVACY MODE TEST")
    print("="*80)
    print()
    print("Testing with question:")
    print(f'  "{question}"')
    print()

    # Simulate privacy choice
    print("="*80)
    print("PRIVACY NOTICE: Grammar Checking")
    print("="*80)
    print("Grammar checking sends your question to LanguageTool's public API.")
    print("⚠️  Your question text will be transmitted to a third-party service.")
    print("✓  Alternative: Use only local validation + AI review (more private)")
    print("="*80)
    print("\nEnable external grammar check? (y/n, default: n): n")
    print("✓ Grammar checking DISABLED (more private - recommended)")
    print()

    # Local validation (100% private)
    print("="*80)
    print("🔍 LOCAL VALIDATION (100% PRIVATE)")
    print("="*80)
    char_count = len(question)
    word_count = len(question.split())

    print(f"✓ Character count: {char_count}/1000")
    print(f"✓ Word count: {word_count}")

    if char_count < 5:
        print("⚠️  Warning: Question too short")
    elif char_count > 1000:
        print("⚠️  Warning: Question too long")
    else:
        print("✓ Length: Good")

    if word_count < 3:
        print("⚠️  Warning: Consider adding more detail")
    elif word_count > 100:
        print("⚠️  Warning: Consider being more concise")
    else:
        print("✓ Detail level: Good")

    if not question.strip().endswith('?'):
        print("⚠️  Suggestion: Consider ending with ?")
    else:
        print("✓ Formatting: Good")

    print("\n📊 Local validation complete - no data transmitted!")
    print()

    # Grammar check (skipped for privacy)
    print("="*80)
    print("🔍 QUESTION REVIEW")
    print("="*80)
    print()
    print("⚠️  Grammar check disabled for privacy (using local validation only)")
    print("   → ✅ Your question stays on your computer")
    print("   → ✅ No data sent to LanguageTool")
    print()

    # Claude AI review
    print("🤖 Getting AI clarity review...")
    print("   → ⚠️  Question will be sent to Claude API (your API key)")
    print()

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

Keep your response concise and actionable."""
            }]
        )

        ai_feedback = ""
        for block in response.content:
            if block.type == 'text':
                ai_feedback += block.text

        print("📊 AI Clarity Review:")
        print("-"*80)
        print(ai_feedback)
        print("-"*80)

        print("\n✅ AI review complete!")

    except Exception as e:
        print(f"❌ AI review failed: {e}")
        print("   (Check your ANTHROPIC_API_KEY in .env file)")

    print()
    print("="*80)
    print("PRIVACY SUMMARY")
    print("="*80)
    print("Data transmitted:")
    print("  ❌ LanguageTool API: Nothing (disabled for privacy)")
    print("  ⚠️  Claude API: Your question (for clarity review)")
    print()
    print("Privacy level: 🟢 HIGH")
    print("  • Grammar check skipped (no third-party)")
    print("  • Only Claude API used (your own API key)")
    print("  • Local validation only (100% private)")
    print("="*80)

if __name__ == "__main__":
    print()
    print("="*80)
    print("TESTING PRIVACY-FOCUSED MODE")
    print("="*80)
    print()
    print("This demonstrates the flow when grammar checking is DISABLED")
    print("(most private option - default setting)")
    print()

    # Test with a sample question
    test_question = "What are the key differences between Claude and GPT-4o?"

    test_privacy_mode(test_question)

    print()
    print("="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print()
    print("✅ This is what happens with privacy mode enabled:")
    print("   1. Local validation only (100% private)")
    print("   2. Grammar check skipped (no LanguageTool)")
    print("   3. Claude AI review (your API key)")
    print()
    print("To run the full interactive version:")
    print("   python API_2_Api_com.py")
    print()
