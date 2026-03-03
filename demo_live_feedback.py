#!/usr/bin/env python3
"""
Demo script for live feedback with privacy-focused grammar checking.
This shows the question input flow without running the full bot conversation.
"""

import anthropic
import os
from dotenv import load_dotenv
from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
import requests

# Load environment variables
load_dotenv()
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def check_grammar_languagetool(text):
    """Check grammar using LanguageTool API."""
    try:
        print("\n🔍 Checking grammar with LanguageTool...")

        response = requests.post(
            'https://api.languagetool.org/v2/check',
            data={'text': text, 'language': 'en-US'},
            timeout=5
        )
        result = response.json()

        if result.get('matches'):
            print(f"\n⚠️  Found {len(result['matches'])} potential grammar/clarity issues:\n")
            for i, match in enumerate(result['matches'][:5], 1):
                context = match.get('context', {})
                offset = context.get('offset', 0)
                length = context.get('length', 0)
                text_excerpt = context.get('text', '')

                issue_text = text_excerpt[offset:offset+length] if text_excerpt else ''
                print(f"{i}. {match['message']}")
                if issue_text:
                    print(f"   Issue: '{issue_text}'")

                if match.get('replacements'):
                    suggestions = [r['value'] for r in match['replacements'][:3]]
                    print(f"   Suggestions: {', '.join(suggestions)}")
                print()
        else:
            print("\n✓ No grammar issues detected")

    except Exception as e:
        print(f"⚠️  Grammar check unavailable: {e}")

class QuestionValidator(Validator):
    """Validator for question input with real-time feedback."""

    def validate(self, document):
        text = document.text

        if len(text.strip()) > 0 and len(text.strip()) < 5:
            raise ValidationError(
                message='Question too short (minimum 5 characters)',
                cursor_position=len(text)
            )

        if len(text) > 1000:
            raise ValidationError(
                message='Question too long (maximum 1000 characters)',
                cursor_position=len(text)
            )

def get_question_with_live_feedback():
    """Get question with live character count, word count, and validation."""

    style = Style.from_dict({
        'bottom-toolbar': 'bg:#222222 #aaaaaa',
        'bottom-toolbar.good': 'bg:#006600 #ffffff bold',
        'bottom-toolbar.warning': 'bg:#aa6600 #ffffff',
        'bottom-toolbar.error': 'bg:#aa0000 #ffffff bold',
    })

    def get_bottom_toolbar():
        """Display live stats in bottom toolbar."""
        from prompt_toolkit.application import get_app
        app = get_app()
        text = app.current_buffer.text

        word_count = len(text.split()) if text.strip() else 0
        char_count = len(text)

        if char_count == 0:
            status = ''
        elif char_count < 5:
            status = ' | <error>Too short</error>'
        elif char_count > 1000:
            status = ' | <error>Too long</error>'
        elif word_count < 3:
            status = ' | <warning>Consider adding more detail</warning>'
        elif word_count > 100:
            status = ' | <warning>Consider being more concise</warning>'
        else:
            status = ' | <good>✓ Good length</good>'

        question_mark = ''
        if char_count > 0 and not text.strip().endswith('?'):
            question_mark = ' | <warning>Consider ending with ?</warning>'

        return HTML(
            f'<b>Characters:</b> {char_count}/1000 | '
            f'<b>Words:</b> {word_count}'
            f'{status}{question_mark} | '
            f'<style bg="#444444">Press Enter to submit | Ctrl+C to cancel</style>'
        )

    print("\n" + "="*80)
    print("📝 QUESTION INPUT WITH LIVE FEEDBACK")
    print("="*80)
    print("Type your question below. The bottom bar shows live stats and suggestions.")
    print()

    try:
        question = prompt(
            '❯ ',
            multiline=False,
            validator=QuestionValidator(),
            validate_while_typing=True,
            bottom_toolbar=get_bottom_toolbar,
            style=style,
            complete_while_typing=False
        )
        return question.strip()

    except KeyboardInterrupt:
        print("\n\n❌ Question input cancelled.")
        return None

def review_and_improve_question(question, enable_grammar_check=False):
    """Review question and offer improvements using Claude."""
    print("\n" + "="*80)
    print("🔍 QUESTION REVIEW")
    print("="*80)

    # Grammar check (if enabled)
    if enable_grammar_check:
        check_grammar_languagetool(question)
    else:
        print("\n⚠️  Grammar check disabled for privacy (using local validation only)")

    # AI review from Claude
    print("\n🤖 Getting AI clarity review...")

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

        print("\n📊 AI Clarity Review:")
        print("-"*80)
        print(ai_feedback)
        print("-"*80)

    except Exception as e:
        print(f"⚠️  AI review unavailable: {e}")

    print("\n" + "="*80)
    print("Review complete!")
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("LIVE FEEDBACK DEMO - Privacy-Focused Mode")
    print("="*80)
    print()

    # Privacy notice
    print("="*80)
    print("PRIVACY NOTICE: Grammar Checking")
    print("="*80)
    print("Grammar checking sends your question to LanguageTool's public API.")
    print("⚠️  Your question text will be transmitted to a third-party service.")
    print("✓  Alternative: Use only local validation + AI review (more private)")
    print("="*80)
    grammar_choice = input("\nEnable external grammar check? (y/n, default: n): ").strip().lower()
    enable_grammar = grammar_choice in ['y', 'yes']

    if enable_grammar:
        print("✓ Grammar checking ENABLED (less private)")
    else:
        print("✓ Grammar checking DISABLED (more private - recommended)")

    # Get question with live feedback
    question = get_question_with_live_feedback()

    if question:
        print(f"\n✅ Question received: \"{question}\"")

        # Review the question
        review_and_improve_question(question, enable_grammar_check=enable_grammar)

        print("\n" + "="*80)
        print("✅ DEMO COMPLETE!")
        print("="*80)
        print(f"\nYour final question: \"{question}\"")
        print("\nIn the full script, this would now start the bot conversation.")
    else:
        print("\n❌ Demo cancelled.")
