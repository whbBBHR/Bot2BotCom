#!/usr/bin/env python3
"""
Test script for live feedback question input feature.
This demonstrates the live feedback interface without running the full bot conversation.
"""

from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

class QuestionValidator(Validator):
    """Validator for question input with real-time feedback."""

    def validate(self, document):
        text = document.text

        # Minimum length check
        if len(text.strip()) > 0 and len(text.strip()) < 5:
            raise ValidationError(
                message='Question too short (minimum 5 characters)',
                cursor_position=len(text)
            )

        # Maximum length check
        if len(text) > 1000:
            raise ValidationError(
                message='Question too long (maximum 1000 characters)',
                cursor_position=len(text)
            )

def test_live_feedback():
    """Test the live feedback input interface."""

    # Custom style for the prompt
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

        # Provide feedback based on stats
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

        # Check for question mark
        question_mark = ''
        if char_count > 0 and not text.strip().endswith('?'):
            question_mark = ' | <warning>Consider ending with ?</warning>'

        return HTML(
            f'<b>Characters:</b> {char_count}/1000 | '
            f'<b>Words:</b> {word_count}'
            f'{status}{question_mark} | '
            f'<style bg="#444444">Press Enter to submit | Ctrl+C to cancel</style>'
        )

    print("="*80)
    print("LIVE FEEDBACK QUESTION INPUT TEST")
    print("="*80)
    print("\nThis demonstrates the live feedback feature.")
    print("Watch the bottom toolbar update as you type!")
    print("\nTry:")
    print("  • Typing less than 5 characters (error)")
    print("  • Typing 3-10 words (warning: too short)")
    print("  • Typing without a question mark (warning)")
    print("  • Typing 10-50 words with a '?' (good)")
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

        print("\n" + "="*80)
        print("✓ QUESTION SUBMITTED SUCCESSFULLY!")
        print("="*80)
        print(f"\nYour question: {question}")
        print(f"Length: {len(question)} characters, {len(question.split())} words")
        print("\n" + "="*80)

        return question.strip()

    except KeyboardInterrupt:
        print("\n\n❌ Input cancelled.")
        return None

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PROMPT_TOOLKIT LIVE FEEDBACK TEST")
    print("="*80)
    print("\nThis test requires 'prompt_toolkit' to be installed.")
    print("Install with: pip install prompt_toolkit")
    print("\n" + "="*80 + "\n")

    try:
        result = test_live_feedback()

        if result:
            print("\n✓ Test completed successfully!")
            print(f"\nFinal question: '{result}'")
        else:
            print("\n⚠ Test cancelled by user.")

    except ImportError as e:
        print(f"\n❌ ERROR: Missing required package")
        print(f"   {e}")
        print("\n   Install with: pip install prompt_toolkit")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
