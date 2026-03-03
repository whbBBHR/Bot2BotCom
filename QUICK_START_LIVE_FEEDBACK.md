# Quick Start: Live Feedback Question Input

## 🚀 Installation (One-Time Setup)

```bash
# Install the required package
pip install prompt_toolkit

# Or install all dependencies
pip install -r requirements.txt
```

## 🧪 Test the Feature

Before using it in the main script, test it standalone:

```bash
python test_live_feedback.py
```

Try typing different things to see the live feedback in action!

## 🎯 Using in Main Script

### Step 1: Run the script
```bash
python API_2_Api_com.py
```

### Step 2: Choose input method
```
STEP 1: Enter Your Question
================================================================================
Choose input method:
  [1] Advanced input with live feedback (recommended)
  [2] Simple text input
================================================================================

Choice (1/2, default: 1): 1
```

### Step 3: Type your question
```
================================================================================
📝 QUESTION INPUT WITH LIVE FEEDBACK
================================================================================
Type your question below. The bottom bar shows live stats and suggestions.

❯ What are the main differences between Claude and GPT-4o?
```

**Watch the bottom toolbar update as you type:**
```
Characters: 58/1000 | Words: 10 | ✓ Good length | Press Enter to submit | Ctrl+C to cancel
```

### Step 4: Review feedback
After pressing Enter, you'll see:

```
================================================================================
🔍 QUESTION REVIEW
================================================================================

🔍 Checking grammar with LanguageTool...
✓ No grammar issues detected

🤖 Getting AI clarity review...

📊 AI Clarity Review:
--------------------------------------------------------------------------------
Clarity score: 9/10
This is a well-formed, specific question that clearly asks for a comparison
between two AI models.

Is it specific enough? Yes
The question is specific and answerable.

Suggested improved version: No improvement needed
--------------------------------------------------------------------------------
```

### Step 5: Choose action
```
================================================================================
OPTIONS:
  [1] Use original question
  [2] Edit and recheck
  [3] Cancel (return to main menu)
================================================================================

Your choice (1/2/3): 1
```

## 📊 Live Feedback Indicators

### Bottom Toolbar Shows:

| Indicator | Meaning |
|-----------|---------|
| `Characters: 58/1000` | Current character count and limit |
| `Words: 10` | Current word count |
| `✓ Good length` | **Green** - Optimal length (3-100 words) |
| `⚠️ Too short` | **Red** - Less than 5 characters |
| `⚠️ Consider adding more detail` | **Yellow** - Less than 3 words |
| `⚠️ Consider being more concise` | **Yellow** - More than 100 words |
| `⚠️ Consider ending with ?` | **Yellow** - No question mark |

## 💡 Tips for Best Results

### ✅ Good Questions:
- **Specific**: "What are the key differences between Claude and GPT-4o?"
- **Clear**: "How does reinforcement learning work in AI?"
- **Focused**: "What ethical concerns exist around AI surveillance?"

### ❌ Avoid:
- **Too vague**: "Tell me about AI"
- **Too broad**: "Explain everything about machine learning algorithms"
- **Multiple questions**: "What is AI and how does it work and what are the risks?"
- **Too short**: "AI?"

## 🎨 Visual Example

```
Before typing:
❯ |
Characters: 0/1000 | Words: 0 | Press Enter to submit | Ctrl+C to cancel

While typing (5 chars):
❯ What|
Characters: 4/1000 | Words: 1 | Too short | Press Enter to submit | Ctrl+C to cancel

Good question:
❯ What are the ethical implications of AI?|
Characters: 44/1000 | Words: 7 | ✓ Good length | Press Enter to submit | Ctrl+C to cancel
```

## 🔧 Troubleshooting

### Q: The bottom toolbar isn't showing

**A:** Your terminal may not support it. Try:
- iTerm2 (macOS)
- Windows Terminal (Windows)
- GNOME Terminal (Linux)

### Q: "ModuleNotFoundError: No module named 'prompt_toolkit'"

**A:** Install it:
```bash
pip install prompt_toolkit
```

### Q: Grammar check says "unavailable"

**A:** This is normal if:
- You're offline
- LanguageTool API is temporarily down
- The script will continue without it

### Q: Can I skip the review step?

**A:** Yes! Choose option `[2] Simple text input` instead, and answer "n" when asked about review.

## 🎯 Advanced Usage

### Quick Mode (Skip Advanced Input)
If you want to skip the live feedback:
1. Choose option `[2] Simple text input`
2. Type your question normally
3. Skip review by answering "n"

### Review Without Live Input
If you just want the grammar/clarity review:
1. Choose option `[2] Simple text input`
2. Type your question
3. Answer "y" to review

## 📝 Example Session

```bash
$ python API_2_Api_com.py

================================================================================
Bot2Bot Conversation - Claude vs GPT-4o
================================================================================
Press Ctrl+C at any time to quit

STEP 1: Enter Your Question
================================================================================
Choose input method:
  [1] Advanced input with live feedback (recommended)
  [2] Simple text input
================================================================================

Choice (1/2, default: 1): 1

================================================================================
📝 QUESTION INPUT WITH LIVE FEEDBACK
================================================================================
Type your question below. The bottom bar shows live stats and suggestions.

❯ What are the main advantages of Claude over other AI models?

[... grammar and clarity review ...]

✓ Using original question

How many conversation turns? (default: 3): 3

[... conversation proceeds ...]
```

## 🚀 Next Steps

1. ✅ Install `prompt_toolkit`
2. ✅ Run `test_live_feedback.py` to try it
3. ✅ Use it in the main script
4. 📚 Read the full documentation in `LIVE_FEEDBACK_FEATURE.md`

---

**Need help?** Check the full documentation or create an issue!
