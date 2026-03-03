# Privacy Guide: Question Input System

## Overview

The live feedback question input system has **different privacy levels** depending on which features you enable. This guide explains what data goes where.

## Privacy Comparison

### 🔒 Most Private: Local Validation Only

**What happens:**
- ✅ Live feedback (character/word count, validation) - **100% local**
- ✅ Color-coded suggestions - **100% local**
- ❌ Grammar check - **DISABLED**
- ⚠️ AI clarity review - **Sent to Anthropic Claude API**

**Data transmitted:**
- **None** to LanguageTool
- **Your question** to Anthropic's Claude API (for clarity review)

**To enable:**
```
Enable external grammar check? (y/n, default: n): n
```

---

### ⚠️ Less Private: With Grammar Check

**What happens:**
- ✅ Live feedback - **100% local**
- ✅ Color-coded suggestions - **100% local**
- ⚠️ Grammar check - **Sent to LanguageTool public API**
- ⚠️ AI clarity review - **Sent to Anthropic Claude API**

**Data transmitted:**
- **Your question** to LanguageTool API (`https://api.languagetool.org`)
- **Your question** to Anthropic's Claude API

**To enable:**
```
Enable external grammar check? (y/n, default: n): y
```

---

## Detailed Privacy Analysis

### What Stays Local (Never Transmitted)

✅ **Live Feedback Features:**
- Character counting
- Word counting
- Length validation (too short/long)
- Question mark detection
- Color-coded status indicators

**100% Private** - Processed entirely on your computer

---

### What Gets Transmitted

#### 1. Grammar Check (LanguageTool API) - OPTIONAL

**When enabled:**
```python
response = requests.post(
    'https://api.languagetool.org/v2/check',
    data={'text': text, 'language': 'en-US'},
    timeout=5
)
```

**What's sent:**
- Your complete question text
- Language code ('en-US')

**Where it goes:**
- LanguageTool's public API servers
- Location: Unknown (likely EU-based)

**Privacy concerns:**
- ⚠️ Third-party service (not controlled by you)
- ⚠️ May log requests (subject to their privacy policy)
- ⚠️ Data retention policy unclear
- ⚠️ No guarantee of deletion
- ✓ HTTPS encrypted in transit
- ✓ Free service (no API key = less tracking)

**LanguageTool Privacy Policy:**
- https://languagetool.org/legal/privacy

---

#### 2. AI Clarity Review (Claude API) - ALWAYS ENABLED

**What's sent:**
```python
response = claude_client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": f"""Review this question for clarity and effectiveness:

"{question}"
"""
    }]
)
```

**What's sent:**
- Your complete question text
- Prompt asking for clarity review

**Where it goes:**
- Anthropic's Claude API servers
- Location: US-based

**Privacy concerns:**
- ⚠️ Your question is processed by Anthropic
- ✓ Subject to Anthropic's privacy policy
- ✓ HTTPS encrypted in transit
- ✓ More reputable company (Anthropic vs LanguageTool)
- ✓ You control the API key
- ⚠️ Logged for abuse prevention (per Anthropic policy)

**Anthropic Privacy Policy:**
- https://www.anthropic.com/privacy

---

## Privacy Recommendations

### 🔒 Maximum Privacy Setup

**Disable grammar checking:**
```
Enable external grammar check? (y/n, default: n): n
```

**Result:**
- ✅ Local validation only
- ✅ AI review via Claude (your API key)
- ❌ No third-party grammar service

**Trade-off:**
- You lose spelling/grammar suggestions
- You keep clarity review (Claude)
- More private overall

---

### ⚖️ Balanced Privacy Setup

**Enable grammar checking for non-sensitive questions:**
```
Enable external grammar check? (y/n, default: n): y
```

**When to use:**
- General questions (not sensitive)
- Public information queries
- Learning/educational questions

**When NOT to use:**
- Questions containing personal info
- Questions about private projects
- Sensitive topics
- Proprietary information

---

### 🚫 Don't Use For Sensitive Data

**Never use live feedback for questions containing:**
- ❌ Personal information (names, addresses, etc.)
- ❌ Passwords or credentials
- ❌ Proprietary business information
- ❌ Medical/health information
- ❌ Financial data
- ❌ Legal matters
- ❌ Anything you wouldn't post publicly

**Why:**
- Your question goes to LanguageTool (if enabled)
- Your question goes to Claude API (always)
- Both services may log requests
- No guarantee of data deletion

---

## Comparison Table

| Feature | Local Only | With Grammar Check |
|---------|------------|-------------------|
| **Live character count** | ✅ Local | ✅ Local |
| **Live word count** | ✅ Local | ✅ Local |
| **Validation errors** | ✅ Local | ✅ Local |
| **Grammar checking** | ❌ Disabled | ⚠️ LanguageTool API |
| **Spelling suggestions** | ❌ Disabled | ⚠️ LanguageTool API |
| **AI clarity review** | ⚠️ Claude API | ⚠️ Claude API |
| **Data to 3rd parties** | Claude only | LanguageTool + Claude |
| **Privacy level** | 🟢 Better | 🟡 Moderate |

---

## Data Retention

### LanguageTool API
- **Unknown** - Free public API
- May log requests for abuse prevention
- No clear data deletion policy
- Subject to GDPR (if EU-based)

### Anthropic Claude API
Per Anthropic's policy:
- API requests may be logged
- Used for abuse prevention
- Not used for training (per Anthropic policy)
- Subject to their data retention policy
- You can request deletion (contact Anthropic)

---

## Alternative: Fully Offline Grammar Check

If you need **complete privacy**, you can replace LanguageTool with an offline library:

### Option: Use `language-tool-python` (Local)

```bash
pip install language-tool-python
```

```python
import language_tool_python

def check_grammar_offline(text):
    """Check grammar using local LanguageTool."""
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)

    for match in matches[:5]:
        print(f"{match.message}")
        print(f"Suggestions: {', '.join(match.replacements[:3])}")

    return matches
```

**Advantages:**
- ✅ 100% offline
- ✅ No data transmitted
- ✅ Completely private
- ✅ No API rate limits

**Disadvantages:**
- ❌ Slower (first-time download is large ~200MB)
- ❌ Uses more RAM
- ❌ Requires Java installation

---

## Privacy Best Practices

### 1. Be Aware of What You Type
- Assume anything you type could be logged
- Don't include sensitive information
- Keep questions general

### 2. Use Grammar Check Selectively
- Disable for sensitive questions
- Enable for general questions
- Default is OFF (more private)

### 3. Review Privacy Policies
- LanguageTool: https://languagetool.org/legal/privacy
- Anthropic: https://www.anthropic.com/privacy

### 4. Consider Offline Alternatives
- Use `language-tool-python` for offline grammar checking
- Skip AI review for very sensitive questions

### 5. Use Environment Variables
- Store API keys in `.env` file
- Never commit `.env` to version control
- Keep API keys secret

---

## Summary: Privacy Levels

### 🟢 High Privacy (Recommended Default)
```
Enable external grammar check? n
```
- Local validation only
- Claude API for clarity review (your API key)
- No third-party grammar service

### 🟡 Moderate Privacy
```
Enable external grammar check? y
```
- Local validation
- LanguageTool API for grammar (third-party)
- Claude API for clarity review

### 🔴 Low Privacy (Not Recommended)
Using this for sensitive data:
- Never type sensitive information
- Both services receive your question
- May be logged/retained

---

## Your Control

You have control over:
- ✅ Whether to enable grammar checking
- ✅ What questions to ask
- ✅ Your Anthropic API key
- ✅ Using the feature at all (can skip)

You DON'T have control over:
- ❌ LanguageTool's logging practices
- ❌ Anthropic's data retention (subject to their policy)
- ❌ Third-party service uptime
- ❌ API terms of service changes

---

## Recommendation

**Default setting (most private):**
```
Enable external grammar check? (y/n, default: n): n
```

This gives you:
- ✅ Live feedback (100% local)
- ✅ AI clarity review (Claude API, your key)
- ❌ No third-party grammar service

**Only enable grammar checking when:**
- Question contains no sensitive data
- You accept LanguageTool's privacy policy
- You need spelling/grammar suggestions
- Question is general/educational

---

## Questions?

**Q: Is Claude API review private?**
A: More private than LanguageTool, but still transmitted to Anthropic. Subject to their privacy policy.

**Q: Can I disable all external services?**
A: Not completely - Claude API is always used for clarity review. You can skip the review step by choosing simple input and declining review.

**Q: Is HTTPS enough?**
A: HTTPS encrypts in transit, but doesn't prevent the service from logging your request.

**Q: Should I trust LanguageTool?**
A: They're a legitimate service, but privacy policies can change. Use at your own discretion.

**Q: How do I go fully offline?**
A: Use `language-tool-python` (offline) and skip the Claude AI review entirely.

---

**Bottom line:** Grammar checking uses a third-party service. It's now **disabled by default** for your privacy. Enable it only when you're comfortable with your question being transmitted to LanguageTool's API.
