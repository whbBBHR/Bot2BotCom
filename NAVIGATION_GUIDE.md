# Navigation & Menu Guide

## ✅ What's Improved

The script now has **better navigation** with:
- ✅ Option to exit at any time
- ✅ Option to go back to previous menu
- ✅ View all features before choosing
- ✅ No getting locked into choices

## 🗺️ Navigation Flow

### Main Menu (Step 1)

```
================================================================================
STEP 1: Enter Your Question
================================================================================
Choose input method:
  [1] Advanced input with live feedback (recommended)
  [2] Simple text input
  [3] Show all script features & options  ← NEW!
  [0] Exit script                          ← NEW!
================================================================================

Choice (0/1/2/3, default: 1):
```

**Options:**
- **[0]** - Exit the script entirely
- **[1]** - Advanced input (live feedback, auto-rewrite)
- **[2]** - Simple text input (traditional)
- **[3]** - See what the script can do

---

### Option [3]: Features Menu

```
================================================================================
BOT2BOT SCRIPT FEATURES
================================================================================

📝 Question Input Features:
  • Live feedback (real-time character/word count)
  • Grammar checking (optional, LanguageTool API)
  • AI clarity review (Claude Sonnet 4.5)
  • Auto-rewrite suggestions

🤖 Bot Conversation Features:
  • Claude vs GPT-4o conversation
  • Configurable number of turns
  • Web search integration (optional)
  • Current date context

🎵 Music Identification Features:
  • Capture system audio (BlackHole required)
  • Record from microphone
  • Shazam integration
  • Audio level monitoring

🖼️  Image Support:
  • Claude can generate images
  • GPT-4o Vision support
  • Images saved to conversation_images/

🔒 Privacy Features:
  • Grammar check disabled by default
  • Web search optional
  • Local validation (100% private)
  • Control what data goes where

================================================================================

Press Enter to continue...
```

Returns to main menu after viewing.

---

### Privacy Notice (With Back Option)

```
================================================================================
PRIVACY NOTICE: Grammar Checking
================================================================================
Grammar checking sends your question to LanguageTool's public API.
⚠️  Your question text will be transmitted to a third-party service.
✓  Alternative: Use only local validation + AI review (more private)
================================================================================

Enable external grammar check? (y/n/back, default: n):
```

**Options:**
- **y** - Enable grammar check (less private)
- **n** - Disable grammar check (more private, default)
- **back** - Return to input method selection ← NEW!

---

### Simple Input (Option 2)

```
Enter your discussion question (or 'back' to return):
```

**Can type:**
- Your question
- **back** - Return to input method selection ← NEW!

Then:

```
Review and improve question? (y/n/back, default: n):
```

**Options:**
- **y** - Review with AI
- **n** - Skip review
- **back** - Return to input method selection ← NEW!

---

### Advanced Input (Option 1)

Uses the live feedback prompt. Press **Ctrl+C** to cancel and return.

---

## 🔄 Navigation Paths

### Path 1: Exit Immediately

```
Main Menu → [0] → Exit ✓
```

### Path 2: View Features First

```
Main Menu → [3] → View Features → Press Enter → Main Menu
         ↓
     Choose [1] or [2]
```

### Path 3: Change Mind on Privacy

```
Main Menu → [1] or [2]
         ↓
    Privacy Notice → [back] → Main Menu
```

### Path 4: Change Mind After Typing

```
Main Menu → [2] Simple Input
         ↓
    "Enter question" → type 'back' → Main Menu
```

### Path 5: Change Mind on Review

```
Main Menu → [2] Simple Input
         ↓
    Type question
         ↓
    "Review?" → type 'back' → Main Menu
```

---

## 📋 Complete Menu Tree

```
┌─────────────────────────────────────┐
│         MAIN MENU                   │
│  [0] Exit                           │
│  [1] Advanced input                 │
│  [2] Simple input                   │
│  [3] Show features                  │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┬──────────┬─────────┐
    │             │          │         │
   [0]           [1]        [2]       [3]
   Exit       Advanced   Simple    Features
    │            │          │         │
    ↓            │          │         │
   END           │          │         ↓
                 │          │      Display
                 │          │      Features
                 │          │         │
                 │          │         ↓
                 │          │      Return
                 │          │      to Menu
                 │          │
                 ↓          ↓
           ┌─────────────────────┐
           │  Privacy Notice     │
           │  [y/n/back]        │
           └──────┬──────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
       [y]       [n]     [back]
        │         │         │
        ↓         ↓         ↓
    Grammar   No Grammar  Return
    Enabled   Checking    to Menu
        │         │
        └────┬────┘
             │
             ↓
    ┌────────────────────┐
    │  Question Input    │
    │  (can type 'back') │
    └────────┬───────────┘
             │
             ↓
    ┌────────────────────┐
    │  Review Question?  │
    │  [y/n/back]       │
    └────────┬───────────┘
             │
        ┌────┼────┬──────┐
        │    │    │      │
       [y]  [n] [back]  Ctrl+C
        │    │    │      │
        ↓    ↓    ↓      ↓
     Review Skip Return Return
        │    │    Menu   Menu
        └────┴────┘
             │
             ↓
    ┌────────────────────┐
    │  Continue to Bot   │
    │  Conversation      │
    └────────────────────┘
```

---

## ⌨️ Keyboard Commands

| Key/Command | Action |
|-------------|--------|
| **0** | Exit script (main menu) |
| **1** | Advanced input with live feedback |
| **2** | Simple text input |
| **3** | Show all features |
| **back** | Return to previous menu |
| **Ctrl+C** | Cancel/return to menu |
| **Enter** | Use default choice |

---

## 💡 Usage Examples

### Example 1: Just Want to Exit

```
Choice (0/1/2/3, default: 1): 0

👋 Exiting Bot2Bot script. Goodbye!
```

### Example 2: Check Features First

```
Choice (0/1/2/3, default: 1): 3

[Features displayed]

Press Enter to continue...

[Returns to main menu]

Choice (0/1/2/3, default: 1): 1
```

### Example 3: Change Mind on Privacy

```
Choice (0/1/2/3, default: 1): 1

Enable external grammar check? (y/n/back, default: n): back

↩️  Returning to input method selection...

Choice (0/1/2/3, default: 1): 2
```

### Example 4: Change Mind After Typing

```
Choice (0/1/2/3, default: 1): 2

Enable external grammar check? (y/n/back, default: n): n

Enter your discussion question (or 'back' to return): back

↩️  Returning to input method selection...

Choice (0/1/2/3, default: 1): 3
```

---

## ✨ Benefits

### Before (Old)

❌ No way to exit gracefully
❌ Couldn't see what script offers
❌ Locked into choice once made
❌ Had to Ctrl+C to go back

### After (New)

✅ Clean exit option [0]
✅ Features menu [3]
✅ Can go 'back' at any step
✅ Better user experience

---

## 🎯 Best Practices

### First Time Using?

1. Choose **[3]** to see all features
2. Read through capabilities
3. Press Enter to return
4. Choose **[1]** or **[2]** based on your needs

### Quick Start?

1. Press **Enter** (uses default [1])
2. Answer **n** to privacy (default)
3. Type your question
4. Continue

### Want to Leave?

- At main menu: Choose **[0]**
- After choosing input: Type **'back'**
- While typing: Press **Ctrl+C**
- Anytime: Press **Ctrl+C** twice

---

## 🔧 Technical Details

### Implementation

**New options added:**
- `[0]` Exit - Calls `exit(0)`
- `[3]` Features - Displays menu, then `continue` loop
- `back` - Returns to top of while loop

**Back functionality:**
- Grammar check: `if grammar_check_choice == 'back': continue`
- Question input: `if initial_prompt.lower() == 'back': continue`
- Review choice: `if review_choice == 'back': continue`

---

## 📝 Summary

The script now offers:

1. **Main Menu**
   - [0] Exit
   - [1] Advanced input
   - [2] Simple input
   - [3] Show features

2. **Back Navigation**
   - Type 'back' at prompts
   - Returns to main menu
   - No need to restart

3. **Features Display**
   - See all capabilities
   - Make informed choice
   - Return to menu

**Result:** Better user experience with full control over navigation!

---

**Updated**: 2026-01-07
**Status**: ✅ Implemented and Ready
