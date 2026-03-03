# Live Feedback Question Input Feature

## Overview

The Bot2Bot conversation script now includes an advanced question input system with **real-time live feedback** to help you write better, clearer questions.

## Features

### 1. Live Feedback Interface
As you type your question, the bottom toolbar displays:
- **Character count** (0-1000 max)
- **Word count**
- **Real-time validation**:
  - ❌ **Too short** (< 5 characters)
  - ⚠️ **Consider adding more detail** (< 3 words)
  - ⚠️ **Consider being more concise** (> 100 words)
  - ✓ **Good length** (optimal range)
  - ⚠️ **Consider ending with ?** (no question mark detected)

### 2. Grammar Checking (LanguageTool API)
After submitting your question, the system:
- Checks grammar using the free LanguageTool API
- Displays up to 5 grammar/clarity issues
- Shows specific suggestions for each issue
- Highlights the problematic text

### 3. AI Clarity Review (Claude)
The system uses Claude AI to review:
- **Clarity score** (1-10) with explanation
- **Specificity** - Is the question specific enough?
- **Suggested improvements** - Only if needed

### 4. Interactive Review Process
After analysis, you can:
1. **Use original question** - Proceed with your original
2. **Edit and recheck** - Revise and analyze again
3. **Cancel** - Return to main menu

## Installation

### Install Required Packages

```bash
pip install prompt_toolkit
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Verify Installation

Run the test script:

```bash
python test_live_feedback.py
```

This will let you try the live feedback interface without running the full bot conversation.

## Usage

### Method 1: Advanced Input (Recommended)

When you run the main script:

```bash
python API_2_Api_com.py
```

1. Choose option **[1] Advanced input with live feedback**
2. Start typing your question
3. Watch the bottom toolbar for live feedback
4. Press **Enter** to submit
5. Review grammar and clarity feedback
6. Choose to use, edit, or cancel

### Method 2: Simple Input

1. Choose option **[2] Simple text input**
2. Type your question normally
3. Optionally request review

## Examples

### Example 1: Too Short
```
❯ AI?
```
**Feedback:**
- Characters: 3/1000 | Words: 1 | ⚠️ Too short | ⚠️ Consider adding more detail

### Example 2: Missing Question Mark
```
❯ What is artificial intelligence
```
**Feedback:**
- Characters: 32/1000 | Words: 4 | ✓ Good length | ⚠️ Consider ending with ?

### Example 3: Optimal Question
```
❯ What are the key differences between Claude and GPT-4o?
```
**Feedback:**
- Characters: 58/1000 | Words: 10 | ✓ Good length

## Grammar Checking Example

**Input:**
```
What is the differense between AI and machine lerning?
```

**LanguageTool Output:**
```
⚠️  Found 2 potential grammar/clarity issues:

1. Possible spelling mistake found.
   Issue: 'differense'
   Suggestions: difference, different, differing

2. Possible spelling mistake found.
   Issue: 'lerning'
   Suggestions: learning, earning, warning
```

## AI Clarity Review Example

**Input:**
```
Tell me about AI
```

**Claude Review:**
```
Clarity score: 4/10
This question is too vague. "AI" covers a vast field, and "tell me about"
doesn't specify what aspect interests you.

Is it specific enough? No
The question should specify whether you want to know about AI capabilities,
history, applications, ethics, or technical implementation.

Suggested improved version:
"What are the main capabilities and limitations of modern AI systems like
Claude and GPT-4o?"
```

## Keyboard Shortcuts

- **Enter** - Submit question
- **Ctrl+C** - Cancel input
- **Backspace** - Delete characters
- **Arrow keys** - Navigate (in multi-line mode)

## Technical Details

### Live Feedback Implementation

The live feedback uses `prompt_toolkit`, a powerful Python library for building interactive command-line applications.

**Key components:**

1. **QuestionValidator** - Validates input in real-time
2. **get_bottom_toolbar()** - Updates status bar dynamically
3. **Custom styling** - Color-coded feedback (green=good, yellow=warning, red=error)

### Grammar API

Uses the free **LanguageTool API** (https://api.languagetool.org):
- No API key required for basic usage
- Supports multiple languages (default: en-US)
- Rate limits apply (use responsibly)

### AI Review

Uses **Claude Sonnet 4.5** for clarity analysis:
- Analyzes question structure and specificity
- Provides actionable suggestions
- Only suggests improvements when needed

## Troubleshooting

### Issue: "Module not found: prompt_toolkit"

**Solution:**
```bash
pip install prompt_toolkit
```

### Issue: Bottom toolbar not updating

**Solution:**
- Ensure your terminal supports ANSI escape codes
- Try a different terminal (iTerm2, Windows Terminal, etc.)
- Update prompt_toolkit: `pip install --upgrade prompt_toolkit`

### Issue: Grammar check fails

**Solution:**
- This is expected if offline or if LanguageTool API is down
- The script will continue without grammar checking
- Check internet connection
- Try again later

### Issue: AI review unavailable

**Solution:**
- Check `ANTHROPIC_API_KEY` in your `.env` file
- Verify API key is valid
- Check API quota/limits

## Customization

### Adjust Character Limits

Edit in [API_2_Api_com.py](API_2_Api_com.py:125):

```python
# Maximum length check
if len(text) > 1000:  # Change 1000 to your preferred limit
    raise ValidationError(...)
```

### Change Word Count Thresholds

Edit in [API_2_Api_com.py](API_2_Api_com.py:159):

```python
elif word_count < 3:  # Adjust threshold
    status = ' | <warning>Consider adding more detail</warning>'
elif word_count > 100:  # Adjust threshold
    status = ' | <warning>Consider being more concise</warning>'
```

### Customize Toolbar Colors

Edit in [API_2_Api_com.py](API_2_Api_com.py:136):

```python
style = Style.from_dict({
    'bottom-toolbar': 'bg:#222222 #aaaaaa',  # Default
    'bottom-toolbar.good': 'bg:#006600 #ffffff bold',  # Success
    'bottom-toolbar.warning': 'bg:#aa6600 #ffffff',  # Warning
    'bottom-toolbar.error': 'bg:#aa0000 #ffffff bold',  # Error
})
```

## Performance

- **Live feedback:** Instant (no API calls while typing)
- **Grammar check:** ~1-2 seconds (LanguageTool API call)
- **AI review:** ~2-5 seconds (Claude API call)

## Privacy & Data

- **Live feedback:** All processing is local, no data sent while typing
- **Grammar check:** Question text sent to LanguageTool API (public service)
- **AI review:** Question text sent to Anthropic's Claude API

## Future Enhancements

Potential improvements:
- [ ] Multi-line question support
- [ ] Save/load question templates
- [ ] Question history with search
- [ ] Custom grammar rules
- [ ] Offline grammar checking
- [ ] Speech-to-text input
- [ ] Auto-suggest completions

## Contributing

Feel free to suggest improvements or report issues!

## License

This feature is part of the Bot2Bot conversation system.
