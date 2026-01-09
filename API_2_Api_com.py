import anthropic
import openai
import os
import re
from dotenv import load_dotenv
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import base64
from pathlib import Path
from PIL import Image
import io
from shazamio import Shazam
import asyncio
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from document_processor import (
    extract_document_text,
    extract_multiple_documents,
    format_document_context,
    get_supported_extensions,
    is_supported_file
)

# Load environment variables from .env file (override any existing ones)
load_dotenv(override=True)

# Bot 1 (Claude) receives a message
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Bot 2 (OpenAI) responds
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Centralized model config with environment override and safe defaults
class ModelConfig:
    CLAUDE_DEFAULT = os.getenv("CLAUDE_MODEL_DEFAULT", "claude-sonnet-4-5-20250929")
    CLAUDE_QUICK_CHECK = os.getenv("CLAUDE_MODEL_QUICK_CHECK", "claude-3-5-haiku-20241022")

# Knowledge base for RAG
KNOWLEDGE_BASE = {
    "Claude": "Large language model by Anthropic, trained on diverse data up to April 2024",
    "GPT-4o": "Multimodal AI model by OpenAI with vision and text capabilities",
    "Bot2Bot": "Dual-agent conversation system combining Claude and GPT-4o for diverse perspectives"
}

def get_context_prompt():
    """Add current date context to conversation."""
    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")
    return f"[Current Date]\n{today}\n\n"

def anthropic_message(model: str, max_tokens: int, messages: list):
    """
    Wrapper to call Anthropic API.
    """
    try:
        return claude_client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
    except anthropic.NotFoundError as e:
        print(f"\n❌ Model not found: {model}")
        print(f"   Error: {e}")
        print(f"   Please check your API key and available models.")
        raise
    except anthropic.AuthenticationError as e:
        print(f"\n❌ Authentication Error - Invalid API key")
        print(f"   Model: {model}")
        print(f"   Error: {e}")
        print(f"\n   Please check your ANTHROPIC_API_KEY in .env file")
        raise

def web_search(query, num_results=3):
    """Perform a simple web search and return summarized results."""
    try:
        # Using DuckDuckGo HTML search (no API key required)
        search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=5)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Extract search results
            for result in soup.find_all('div', class_='result')[:num_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')

                if title_elem and snippet_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'snippet': snippet_elem.get_text(strip=True)
                    })

            if results:
                search_summary = f"\n[Web Search Results for '{query}']\n"
                for i, res in enumerate(results, 1):
                    search_summary += f"{i}. {res['title']}\n   {res['snippet']}\n"
                return search_summary

        return ""
    except Exception as e:
        print(f"\n[Web search failed: {str(e)}]")
        return ""

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
            issues = []
            for i, match in enumerate(result['matches']):

                context = match.get('context', {})
                offset = context.get('offset', 0)
                length = context.get('length', 0)
                text_excerpt = context.get('text', '')

                issue_text = text_excerpt[offset:offset+length] if text_excerpt else ''
                print(f"{i+1}. {match['message']}")
                if issue_text:
                    print(f"   Issue: '{issue_text}'")

                if match.get('replacements'):
                    suggestions = [r['value'] for r in match['replacements'][:3]]
                    print(f"   Suggestions: {', '.join(suggestions)}")
                    issues.append({
                        'message': match['message'],
                        'issue': issue_text,
                        'suggestions': suggestions
                    })
                print()

            return issues
        else:
            print("\n✓ No grammar issues detected")
            return []

    except Exception as e:
        print(f"⚠️  Grammar check unavailable: {e}")
        return None

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
        if len(text) > 5000:
            raise ValidationError(
                message='Question too long (maximum 5000 characters)',
                cursor_position=len(text)
            )

def get_question_with_live_feedback():
    """Get question with live character count, word count, and validation."""

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
        elif char_count > 5000:
            status = ' | <error>Too long</error>'
        elif word_count < 3:
            status = ' | <warning>Consider adding more detail</warning>'
        elif word_count > 500:
            status = ' | <warning>Very long - consider splitting</warning>'
        else:
            status = ' | <good>✓ Good length</good>'

        # Check for question mark
        question_mark = ''
        if char_count > 0 and not text.strip().endswith('?'):
            question_mark = ' | <warning>Consider ending with ?</warning>'

        return HTML(
            f'<b>Characters:</b> {char_count}/5000 | '
            f'<b>Words:</b> {word_count}'
            f'{status}{question_mark} | '
            f'<style bg="#444444">Press Enter to submit | Ctrl+C to cancel</style>'
        )

    print("\n" + "="*80)
    print("📝 QUESTION INPUT WITH LIVE FEEDBACK")
    print("="*80)
    print("Type your question below. The bottom bar shows live stats and suggestions.")
    print("💡 Use mouse to click and position cursor, or select text to edit.")
    print()

    try:
        question = prompt(
            '❯ ',
            multiline=False,
            validator=QuestionValidator(),
            validate_while_typing=True,
            bottom_toolbar=get_bottom_toolbar,
            style=style,
            complete_while_typing=False,
            mouse_support=True
        )

        return question.strip()

    except KeyboardInterrupt:
        print("\n\n❌ Question input cancelled.")
        return None

def extract_suggested_question(ai_feedback):
    """Extract the suggested improved question from AI feedback."""
    if not ai_feedback:
        return None

    import re

    # Check if AI said no improvement needed
    if re.search(r'no improvement needed|already good|no change|question is (good|fine|clear)', ai_feedback, re.IGNORECASE):
        return None

    # Look for common patterns for suggestions (more specific patterns first)
    patterns = [
        r'Suggested improved version:\s*["\']([^"\']+)["\']',  # With quotes
        r'Suggested improved version:\s*\n\s*["\']([^"\']+)["\']',  # Multi-line with quotes
        r'Suggested:\s*["\']([^"\']+)["\']',
        r'Better:\s*["\']([^"\']+)["\']',
        r'Try:\s*["\']([^"\']+)["\']',
        r'Improved:\s*["\']([^"\']+)["\']',
        r'Rewrite:\s*["\']([^"\']+)["\']',
        # Look for quoted questions (ending with ?)
        r'"([^"]{15,200}\?)"',  # Quoted text 15-200 chars ending with ?
    ]

    for pattern in patterns:
        match = re.search(pattern, ai_feedback, re.IGNORECASE)
        if match:
            suggestion = match.group(1).strip()
            # Clean up
            suggestion = suggestion.strip('.,;:')

            # Validate it's a reasonable question
            if (len(suggestion) >= 15 and
                len(suggestion) <= 300 and
                suggestion.count(' ') >= 2 and  # At least 3 words
                not suggestion.startswith('For ')):  # Skip "For basic definition:" etc
                return suggestion

    return None

def review_and_improve_question(question, enable_grammar_check=False):
    """Automatically review and improve a question (no prompts)."""
    print("\n" + "="*80)
    print("🔍 QUESTION REVIEW (automatic)")
    print("="*80)

    # Optional external grammar check (silent)
    if enable_grammar_check:
        _ = check_grammar_languagetool(question)
    else:
        print("Grammar check disabled (privacy).")

    # Use the non-interactive refinement
    result = auto_refine_question(question, enable_grammar_check=enable_grammar_check)

    if result["suggestion"]:
        print(f'✓ Applied AI improvement: "{result["final"]}"')
    else:
        print("✓ No improvement needed. Using original.")

    return result["final"]

def chatbot_conversation(initial_prompt, turns=3, enable_web_search=False, audio_file=None, document_context=None):
    conversation_history = []
    
    # Add context with current date
    context = get_context_prompt()

    # Add knowledge base info
    kb_context = f"\n[Knowledge Base]\n"
    for key, value in KNOWLEDGE_BASE.items():
        kb_context += f"- {key}: {value}\n"

    # Identify music if audio file is provided
    music_info = ""
    if audio_file and Path(audio_file).exists():
        print("Identifying music...", end=" ", flush=True)
        music_result = identify_music(audio_file)
        if music_result.get('found'):
            music_info = f"\n[Music Identified]\n"
            music_info += f"Title: {music_result['title']}\n"
            music_info += f"Artist: {music_result['artist']}\n"
            music_info += f"Album: {music_result['album']}\n"
            music_info += f"Genre: {music_result['genres']}\n"
            if music_result.get('shazam_url'):
                music_info += f"More info: {music_result['shazam_url']}\n"
            print("✓", end=" ")
        else:
            print(f"✗ ({music_result.get('error', 'Unknown error')})", end=" ")

    # Perform web search if enabled
    search_results = ""
    if enable_web_search:
        print("Searching web...", end=" ", flush=True)
        search_results = web_search(initial_prompt)
        print("✓", end=" ")

    # Add document context if provided
    doc_context = ""
    if document_context:
        print("Adding document context...", end=" ", flush=True)
        doc_context = format_document_context(document_context)
        print("✓", end=" ")

    # Combine all context with the initial prompt
    enhanced_prompt = context + kb_context
    if music_info:
        enhanced_prompt += music_info
    if search_results:
        enhanced_prompt += search_results
    if doc_context:
        enhanced_prompt += doc_context
    enhanced_prompt += f"\n[User Question]\n{initial_prompt}"

    current_message = enhanced_prompt

    try:
        for i in range(turns):
            print(f"Turn {i+1}/{turns}...", end=" ", flush=True)

            # Claude responds
            claude_response = anthropic_message(
                ModelConfig.CLAUDE_DEFAULT,
                1024,
                [{"role": "user", "content": current_message}]
            )

            # Extract text and images from Claude's response
            claude_text = ""
            claude_images = []

            for block in claude_response.content:
                if block.type == 'text':
                    claude_text += block.text

            # Extract any images from Claude's response
            claude_images = extract_images_from_response(claude_response.content, i+1, "Claude")

            # Store Claude's response
            conversation_history.append({
                "speaker": "Claude",
                "message": claude_text,
                "images": claude_images
            })
            print("Claude ✓", end=" ", flush=True)

            # GPT responds to Claude (with vision support)
            # Build GPT message content with text
            message_content = [{"type": "text", "text": claude_text}]

            # Add images if Claude provided any
            if claude_images:
                for img in claude_images:
                    image_data = prepare_image_for_gpt(img['path'])
                    if image_data:
                        message_content.append(image_data)

            gpt_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": message_content}]
            )

            gpt_text = gpt_response.choices[0].message.content or ""
            conversation_history.append({
                "speaker": "GPT",
                "message": gpt_text,
                "images": []
            })
            print("GPT ✓")

            current_message = gpt_text
    except KeyboardInterrupt:
        print("\n\nConversation interrupted! Returning partial results...\n")

    return conversation_history

# Create images directory if it doesn't exist
IMAGES_DIR = Path("conversation_images")
IMAGES_DIR.mkdir(exist_ok=True)

# System audio capture
def list_audio_devices():
    """List all available audio devices with their capabilities."""
    devices = sd.query_devices()
    print("\n=== Available Audio Devices ===\n")

    for idx, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append(f"INPUT ({device['max_input_channels']} ch)")
        if device['max_output_channels'] > 0:
            device_type.append(f"OUTPUT ({device['max_output_channels']} ch)")

        marker = ">" if idx == sd.default.device[0] else " "
        print(f"{marker} [{idx}] {device['name']}")
        print(f"     Type: {' + '.join(device_type)}")
        print(f"     Sample Rate: {int(device['default_samplerate'])} Hz")
        print()

    return devices

def capture_system_audio(duration=10, sample_rate=44100, output_file="temp_recording.wav", device=None):
    """
    Capture audio directly from system output (what's playing through speakers).

    On macOS, this requires a virtual device OR using Aggregate Device setup.
    On Windows, this can use WASAPI loopback.
    On Linux, this can use PulseAudio monitor.
    """
    try:
        print(f"\n🔊 Capturing system audio for {duration} seconds...")

        if device is None:
            # Try to auto-detect loopback/monitor device
            devices = sd.query_devices()
            found_devices = []

            for idx, dev in enumerate(devices):
                dev_name = dev['name'].lower()

                # Look for virtual audio devices (BlackHole, Soundflower, etc.)
                if any(keyword in dev_name for keyword in ['blackhole', 'soundflower', 'loopback', 'stereo mix', 'monitor']):
                    if dev['max_input_channels'] > 0:
                        found_devices.append((idx, dev['name'], 'virtual'))

                # Look for aggregate devices
                elif 'aggregate' in dev_name and dev['max_input_channels'] > 0:
                    found_devices.append((idx, dev['name'], 'aggregate'))

                # Check for MacBook Pro speakers with input capability (rare but possible)
                elif 'macbook' in dev_name and 'speaker' in dev_name and dev['max_input_channels'] > 0:
                    found_devices.append((idx, dev['name'], 'builtin'))

            if found_devices:
                # Prioritize: virtual > aggregate > builtin
                found_devices.sort(key=lambda x: {'virtual': 0, 'aggregate': 1, 'builtin': 2}[x[2]])
                device = found_devices[0][0]
                device_name = found_devices[0][1]
                device_type = found_devices[0][2]

                print(f"✓ Found audio capture device: {device_name}")
                if device_type == 'builtin':
                    print("  ⚠️  Using built-in device - this may capture microphone audio")
                    print("  For true system audio, set up BlackHole or Aggregate Device")
            else:
                print("⚠️  No loopback audio device found!")
                print("\n📌 macOS System Audio Capture Options:")
                print("\n  Option 1: BlackHole (Recommended - Easiest)")
                print("    1. Install: brew install blackhole-2ch")
                print("    2. Open Audio MIDI Setup (Cmd+Space, search 'Audio MIDI')")
                print("    3. Click '+' → Create Multi-Output Device")
                print("    4. Check: MacBook Pro Speakers + BlackHole 2ch")
                print("    5. Set Multi-Output as system output")
                print("    6. Run this script - it will auto-detect BlackHole!")
                print("\n  Option 2: Soundflower (Alternative)")
                print("    brew install soundflower")
                print("\n  Option 3: Use Microphone (Current Fallback)")
                print("    Record via microphone picking up speaker audio")
                print("\nFalling back to default input device (microphone)...")
                return None

        print("Recording in: 3...", end=" ", flush=True)
        time.sleep(1)
        print("2...", end=" ", flush=True)
        time.sleep(1)
        print("1...", end=" ", flush=True)
        time.sleep(1)
        print("Capturing NOW! 🔴\n")

        # Capture audio from specified device
        recording = sd.rec(int(duration * sample_rate),
                          samplerate=sample_rate,
                          channels=2,
                          dtype='float64',
                          device=device)

        # Monitor capture levels in real-time
        start_time = time.time()
        frames_collected = 0

        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            remaining = duration - elapsed

            current_samples = int(elapsed * sample_rate)
            if current_samples > frames_collected and current_samples < len(recording):
                recent_audio = recording[frames_collected:current_samples]
                if len(recent_audio) > 0:
                    level = np.abs(recent_audio).mean()

                    bar_length = int(min(level * 50, 50))
                    bar = "█" * bar_length + "░" * (50 - bar_length)

                    if level > 0.1:
                        status = "🟢"
                    elif level > 0.01:
                        status = "🟡"
                    else:
                        status = "🔴"

                    print(f"\r{status} Capturing... {remaining:.1f}s | {bar} | Level: {level:.3f}", end="", flush=True)
                    frames_collected = current_samples

            time.sleep(0.1)

        sd.wait()
        print("\n")

        # Analyze capture quality
        avg_level = np.abs(recording).mean()

        if avg_level < 0.001:
            print("⚠️  Warning: Very low audio captured!")
            print("   Make sure audio is playing through the selected device.\n")
        elif avg_level < 0.01:
            print("⚠️  Low audio levels detected.\n")
        else:
            print("✓ Good audio levels captured!\n")

        # Convert to int16 format for better Shazam compatibility
        recording_int16 = np.int16(recording * 32767)

        # Save to file
        sf.write(output_file, recording_int16, sample_rate, subtype='PCM_16')
        print(f"✓ System audio saved to {output_file}")
        print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
        print(f"  Duration: {duration}s")
        print(f"  Format: 16-bit PCM, {sample_rate} Hz\n")

        return output_file
    except Exception as e:
        print(f"\n✗ System audio capture failed: {str(e)}")
        return None

# Audio recording
def test_microphone(duration=3, sample_rate=44100):
    """Test microphone and show audio levels."""
    try:
        print("\n🎤 Testing microphone...")
        print("Speak or play music to test audio levels\n")

        # Record a short test
        test_recording = sd.rec(int(duration * sample_rate),
                               samplerate=sample_rate,
                               channels=2,
                               dtype='float64')

        # Show live audio levels
        start_time = time.time()
        while time.time() - start_time < duration:
            # Calculate current audio level
            if sd.get_stream().active:
                current_frame = sd.get_stream().read(sample_rate // 10)[0]
                level = np.abs(current_frame).mean()

                # Create visual bar
                bar_length = int(level * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)

                # Print level with color indicator
                if level > 0.1:
                    status = "🟢 GOOD"
                elif level > 0.01:
                    status = "🟡 LOW "
                else:
                    status = "🔴 NONE"

                print(f"\r{status} | {bar} | {level:.3f}", end="", flush=True)
            time.sleep(0.1)

        sd.wait()
        print("\n✓ Microphone test complete!\n")

        # Analyze the test recording
        max_level = np.abs(test_recording).max()
        avg_level = np.abs(test_recording).mean()

        print(f"Audio Statistics:")
        print(f"  Max Level: {max_level:.3f}")
        print(f"  Avg Level: {avg_level:.3f}")

        if avg_level < 0.01:
            print("  ⚠️  Warning: Audio levels are very low!")
            print("     - Check microphone connection")
            print("     - Increase volume/gain")
            print("     - Move closer to audio source")
        elif avg_level > 0.1:
            print("  ✓ Audio levels look good!")
        else:
            print("  ⚠️  Audio levels are acceptable but could be higher")

        return avg_level > 0.001

    except Exception as e:
        print(f"\n✗ Microphone test failed: {str(e)}")
        return False

def record_audio_with_monitoring(duration=10, sample_rate=44100, output_file="temp_recording.wav"):
    """Record audio from the system microphone with real-time monitoring."""
    try:
        print(f"\n🎤 Recording audio for {duration} seconds...")
        print("Make sure music is playing and your microphone can hear it!")
        print("Recording in: 3...", end=" ", flush=True)
        time.sleep(1)
        print("2...", end=" ", flush=True)
        time.sleep(1)
        print("1...", end=" ", flush=True)
        time.sleep(1)
        print("Recording NOW! 🔴\n")

        # Record audio
        recording = sd.rec(int(duration * sample_rate),
                          samplerate=sample_rate,
                          channels=2,
                          dtype='float64')

        # Monitor recording levels in real-time
        start_time = time.time()
        frames_collected = 0

        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            remaining = duration - elapsed

            # Calculate current audio level from recorded data
            current_samples = int(elapsed * sample_rate)
            if current_samples > frames_collected and current_samples < len(recording):
                recent_audio = recording[frames_collected:current_samples]
                if len(recent_audio) > 0:
                    level = np.abs(recent_audio).mean()

                    # Create visual bar
                    bar_length = int(min(level * 50, 50))
                    bar = "█" * bar_length + "░" * (50 - bar_length)

                    # Status indicator
                    if level > 0.1:
                        status = "🟢"
                    elif level > 0.01:
                        status = "🟡"
                    else:
                        status = "🔴"

                    print(f"\r{status} Recording... {remaining:.1f}s | {bar} | Level: {level:.3f}", end="", flush=True)
                    frames_collected = current_samples

            time.sleep(0.1)

        sd.wait()  # Wait until recording is finished
        print("\n")

        # Analyze recording quality
        max_level = np.abs(recording).max()
        avg_level = np.abs(recording).mean()

        if avg_level < 0.001:
            print("⚠️  Warning: Very low audio detected! Recording may not identify properly.")
            print("   Consider re-recording with higher volume.\n")
        elif avg_level < 0.01:
            print("⚠️  Low audio levels detected. May affect identification accuracy.\n")
        else:
            print("✓ Good audio levels detected!\n")

        # Convert to int16 format for better Shazam compatibility
        recording_int16 = np.int16(recording * 32767)

        # Save to file
        sf.write(output_file, recording_int16, sample_rate, subtype='PCM_16')
        print(f"✓ Recording saved to {output_file}")
        print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
        print(f"  Duration: {duration}s")
        print(f"  Format: 16-bit PCM, {sample_rate} Hz\n")

        return output_file
    except Exception as e:
        print(f"\n✗ Recording failed: {str(e)}")
        return None

def record_audio(duration=10, sample_rate=44100, output_file="temp_recording.wav"):
    """Wrapper for backward compatibility - uses monitoring version."""
    return record_audio_with_monitoring(duration, sample_rate, output_file)

# Music identification
async def identify_music_async(audio_file_path):
    """Identify music from an audio file using Shazam."""
    import json
    from pathlib import Path

    print("\n=== Music Identification Debug ===")

    # Validate audio file
    try:
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            print(f"❌ ERROR: Audio file does not exist: {audio_file_path}")
            return {'found': False, 'error': f'File not found: {audio_file_path}'}

        file_size = audio_path.stat().st_size
        print(f"✓ Audio file exists: {audio_file_path}")
        print(f"  File size: {file_size / 1024:.1f} KB")

        if file_size < 1000:
            print(f"⚠ WARNING: File is very small ({file_size} bytes) - may be empty or corrupted")

        # Check audio file format
        import soundfile as sf
        try:
            audio_info = sf.info(audio_file_path)
            print(f"  Sample rate: {audio_info.samplerate} Hz")
            print(f"  Channels: {audio_info.channels}")
            print(f"  Duration: {audio_info.duration:.2f}s")
            print(f"  Format: {audio_info.format} / {audio_info.subtype}")
        except Exception as info_err:
            print(f"⚠ WARNING: Could not read audio info: {info_err}")

    except Exception as validation_err:
        print(f"❌ ERROR during file validation: {validation_err}")
        import traceback
        traceback.print_exc()
        return {'found': False, 'error': f'Validation error: {validation_err}'}

    # Attempt Shazam identification
    try:
        print("\n🔍 Calling Shazam API...")
        shazam = Shazam()
        result = await shazam.recognize(audio_file_path)

        print("✓ Shazam API call completed")

        # Log the full response structure
        print(f"\n--- Full Shazam Response ---")
        print(json.dumps(result, indent=2, default=str)[:2000])  # First 2000 chars
        print("--- End Response ---\n")

        if result and 'track' in result:
            track = result['track']
            print("✓ Match found!")
            print(f"  Title: {track.get('title', 'Unknown')}")
            print(f"  Artist: {track.get('subtitle', 'Unknown Artist')}")

            return {
                'title': track.get('title', 'Unknown'),
                'artist': track.get('subtitle', 'Unknown Artist'),
                'album': track.get('sections', [{}])[0].get('metadata', [{}])[0].get('text', 'Unknown Album') if track.get('sections') else 'Unknown Album',
                'genres': track.get('genres', {}).get('primary', 'Unknown Genre'),
                'shazam_url': track.get('url', ''),
                'cover_art': track.get('images', {}).get('coverart', ''),
                'found': True
            }
        else:
            print("❌ No match found in Shazam response")
            print(f"Response keys: {list(result.keys()) if result else 'None'}")
            return {'found': False, 'error': 'No match found'}

    except Exception as e:
        print(f"❌ ERROR during Shazam identification: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return {'found': False, 'error': str(e)}

def identify_music(audio_file_path):
    """Synchronous wrapper for music identification."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(identify_music_async(audio_file_path))

def test_audio_playback(audio_file_path):
    """Test audio file by playing it back and showing waveform analysis."""
    from pathlib import Path
    import soundfile as sf
    import numpy as np

    print("\n=== Audio Playback Test ===")

    try:
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            print(f"❌ ERROR: Audio file does not exist: {audio_file_path}")
            return False

        # Read audio file
        audio_data, sample_rate = sf.read(audio_file_path)
        print(f"✓ Audio file loaded: {audio_file_path}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(audio_data) / sample_rate:.2f}s")
        print(f"  Channels: {audio_data.shape[1] if len(audio_data.shape) > 1 else 1}")

        # Analyze audio content
        if len(audio_data.shape) > 1:
            # Stereo - take mean of channels for analysis
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data

        max_amplitude = np.max(np.abs(audio_mono))
        rms_level = np.sqrt(np.mean(audio_mono**2))

        print(f"\n📊 Audio Analysis:")
        print(f"  Max amplitude: {max_amplitude:.4f}")
        print(f"  RMS level: {rms_level:.4f}")
        print(f"  Dynamic range: {20 * np.log10(max_amplitude / (rms_level + 1e-10)):.1f} dB")

        if max_amplitude < 0.01:
            print(f"  ⚠ WARNING: Audio is very quiet (max: {max_amplitude:.4f})")
            print(f"            Recording may be too quiet for Shazam to identify")
        elif max_amplitude > 0.95:
            print(f"  ⚠ WARNING: Audio may be clipping (max: {max_amplitude:.4f})")
        else:
            print(f"  ✓ Audio levels look good")

        # Check for silence
        silence_threshold = 0.001
        non_silent_samples = np.sum(np.abs(audio_mono) > silence_threshold)
        silence_percentage = 100 * (1 - non_silent_samples / len(audio_mono))

        print(f"  Silence: {silence_percentage:.1f}%")
        if silence_percentage > 50:
            print(f"  ⚠ WARNING: Audio is mostly silence ({silence_percentage:.1f}%)")

        # Simple waveform visualization (first 100 samples)
        print(f"\n📈 Waveform (first 0.1s):")
        samples_to_show = min(100, len(audio_mono))
        step = max(1, samples_to_show // 50)

        for i in range(0, samples_to_show, step):
            amplitude = audio_mono[i]
            bar_length = int(abs(amplitude) * 40)
            bar = '█' * bar_length
            print(f"  {amplitude:+.3f} |{bar}")

        # Offer to play audio
        play_choice = input("\n🔊 Play audio file? (y/n): ").strip().lower()
        if play_choice in ['y', 'yes']:
            print("Playing audio...")
            sd.play(audio_data, sample_rate)
            sd.wait()
            print("✓ Playback complete")

        return True

    except Exception as e:
        print(f"❌ ERROR during audio test: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_image_from_base64(image_data, image_name):
    """Save a base64 encoded image to a file."""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)

        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Save image
        image_path = IMAGES_DIR / image_name
        image.save(image_path)

        return str(image_path)
    except Exception as e:
        print(f"\n[Error saving image: {str(e)}]")
        return None

def extract_images_from_response(response_content, turn_num, speaker):
    """Extract images from Claude's response content blocks."""
    images = []
    image_counter = 1

    for block in response_content:
        # Check if block is an image
        if hasattr(block, 'type') and block.type == 'image':
            image_name = f"turn{turn_num}_{speaker}_image{image_counter}.png"

            # Save the image
            if hasattr(block, 'source') and hasattr(block.source, 'data'):
                image_path = save_image_from_base64(block.source.data, image_name)
                if image_path:
                    images.append({
                        'path': image_path,
                        'media_type': getattr(block.source, 'media_type', 'image/png')
                    })
                    image_counter += 1

    return images

def prepare_image_for_gpt(image_path):
    """Prepare an image for GPT-4o vision by encoding it to base64."""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Determine image type from extension
        extension = Path(image_path).suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(extension, 'image/png')

        return {
            'type': 'image_url',
            'image_url': {
                'url': f"data:{media_type};base64,{image_data}"
            }
        }
    except Exception as e:
        print(f"\n[Error preparing image for GPT: {str(e)}]")
        return None

# Document Import Functions
def import_documents_menu():
    """Interactive menu for importing documents."""
    print("\n" + "="*80)
    print("📄 DOCUMENT IMPORT")
    print("="*80)
    print("\nSupported formats: PDF, DOCX, DOC, TXT, MD")
    print()

    # Get file paths
    print("Enter document paths (one per line, empty line to finish):")
    print("Example: /path/to/document.pdf")
    print()

    file_paths = []
    while True:
        path = input(f"  Document {len(file_paths)+1} (or press Enter to finish): ").strip()

        if not path:
            if len(file_paths) == 0:
                print("⚠️  No documents provided.")
                return None
            break

        if path.lower() == 'back':
            return None

        # Expand user path
        path = os.path.expanduser(path)

        # Check if file exists
        if not os.path.exists(path):
            print(f"    ❌ File not found: {path}")
            continue

        # Check if supported
        if not is_supported_file(path):
            ext = Path(path).suffix
            print(f"    ❌ Unsupported format: {ext}")
            print(f"    Supported: {', '.join(get_supported_extensions())}")
            continue

        file_paths.append(path)
        print(f"    ✓ Added: {Path(path).name}")

    if not file_paths:
        return None

    # Extract documents
    print(f"\n📖 Processing {len(file_paths)} document(s)...")
    result = extract_multiple_documents(file_paths)

    # Show summary
    summary = result['summary']
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total files: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total words: {summary['total_words']:,}")
    print(f"Total characters: {summary['total_chars']:,}")

    # Show individual results
    print("\nDocuments:")
    for doc in result['documents']:
        if doc['status'] == 'success':
            print(f"  ✓ {doc['filename']} ({doc['word_count']:,} words)")
        else:
            print(f"  ❌ {doc['filename']} - {doc.get('error', 'Unknown error')}")

    if summary['errors']:
        print("\nErrors:")
        for error in summary['errors']:
            print(f"  • {error}")

    print("="*80)

    return result

def get_document_qa_question():
    """Get a question about the loaded documents."""
    print("\n" + "="*80)
    print("📝 DOCUMENT Q&A MODE")
    print("="*80)
    print("\nYour question will be answered based on the loaded documents.")
    print("The bots will have access to all document content as context.")
    print()

    question = input("Ask a question about the documents (or 'back' to cancel): ").strip()

    if question.lower() == 'back' or not question:
        return None

    return question

def auto_refine_question(question: str, enable_grammar_check: bool = False) -> dict:
    """
    Non-interactive refinement:
    - Optionally runs grammar check (no prompts)
    - Gets AI clarity feedback and a single suggested improved question
    Returns dict: {'final': str, 'suggestion': str|None, 'clarity_feedback': str|None}
    """
    if enable_grammar_check:
        # Silent grammar check (prints minimal info if issues)
        _ = check_grammar_languagetool(question)

    ai_feedback = None
    suggestion = None
    try:
        response = anthropic_message(
            ModelConfig.CLAUDE_DEFAULT,
            500,
            [{
                "role": "user",
                "content": f"""Review this question for clarity and effectiveness and provide ONE improved version if needed:

"{question}"

Provide:
1. Clarity score (1-10) with brief explanation
2. Is it specific enough? (yes/no with explanation)
3. Suggested improved version (exactly ONE), or "No improvement needed"

Format:
Clarity: X/10 - ...
Specific: yes/no - ...
Suggested improved version: "Your single best improved question"
"""
            }]
        )
        ai_feedback = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                ai_feedback += block.text
        suggestion = extract_suggested_question(ai_feedback)
    except Exception as e:
        ai_feedback = f"AI review unavailable: {e}"
        suggestion = None

    final = suggestion or question
    return {"final": final, "suggestion": suggestion, "clarity_feedback": ai_feedback}

if __name__ == "__main__":
    try:
        print("=" * 80)
        print("Bot2Bot Conversation - Claude vs GPT-4o")
        print("=" * 80)
        print("Press Ctrl+C at any time to quit\n")

        document_context = None

        while True:
            print("\n" + "="*80)
            print("STEP 1: Ask Your Question")
            print("="*80)
            print("Choose input method:")
            print("  [1] Quick Ask (automatic quality and refinement)")
            print("  [2] Advanced input (live editor, then auto refine)")
            print("  [3] Import documents (PDF, DOCX, TXT, MD)")
            print("  [0] Exit")
            print("="*80)

            input_method = input("Choice (0/1/2/3, default: 1): ").strip() or "1"

            if input_method == "0":
                print("\n👋 Goodbye!")
                break

            if input_method == "3":
                result = import_documents_menu()
                if result:
                    document_context = result
                    print("\n✓ Documents loaded successfully!")
                    qa_choice = input("Use Document Q&A mode? (y/n, default: y): ").strip().lower()
                    if qa_choice != 'n':
                        qa_question = get_document_qa_question()
                        if qa_question:
                            initial_prompt = qa_question
                            turns = 3
                            enable_web_search = False
                            audio_file = None

                            print(f"\n🎯 Starting Document Q&A conversation...")
                            conversation = chatbot_conversation(
                                initial_prompt,
                                turns=turns,
                                enable_web_search=enable_web_search,
                                audio_file=audio_file,
                                document_context=document_context
                            )

                            print("\n" + "="*80)
                            print("=== Bot2Bot Conversation (Document Q&A) ===")
                            print("="*80 + "\n")
                            for entry in conversation:
                                print(f"{entry['speaker']}:")
                                print(f"{entry['message']}\n")
                                if entry.get('images'):
                                    for img in entry['images']:
                                        print(f"  📷 {img['path']}")
                                print("-" * 80 + "\n")

                            continue_choice = input("\nStart another conversation? (y/n): ").strip().lower()
                            if continue_choice not in ['y', 'yes']:
                                break
                continue

            if input_method == "2":
                # Advanced input with live feedback
                initial_prompt = get_question_with_live_feedback()
                if not initial_prompt:
                    print("↩️  Cancelled. Returning to menu.")
                    continue

            else:
                # Quick Ask: single input
                initial_prompt = input("\nEnter your question: ").strip()
                if not initial_prompt:
                    print("❌ Please enter a valid question.")
                    continue

            # Turns
            try:
                turns = int(input("\nHow many turns? (default: 3): ").strip() or "3")
            except ValueError:
                turns = 3

            # Optional web search
            enable_web_search = input("Enable web search? [y/N]: ").strip().lower() in ("y", "yes")

            audio_file = None

            print(f"\n🚀 Starting {turns}-turn conversation")
            conversation = chatbot_conversation(
                initial_prompt,
                turns=turns,
                enable_web_search=enable_web_search,
                audio_file=audio_file,
                document_context=document_context
            )

            print("\n" + "="*80)
            print("=== Bot2Bot Conversation ===")
            print("="*80 + "\n")
            for entry in conversation:
                print(f"{entry['speaker']}:")
                print(f"{entry['message']}\n")
                if entry.get('images'):
                    for img in entry['images']:
                        print(f"  📷 {img['path']}")
                print("-" * 80 + "\n")

            # Loop
            if input("Start another conversation? [y/N]: ").strip().lower() not in ("y", "yes"):
                print("\n👋 Goodbye!")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting...")
        exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)