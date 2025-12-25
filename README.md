# Bot2Bot Communication

A Python project that enables conversation between two AI models: Claude (Anthropic) and GPT-4o (OpenAI). Watch as they discuss topics back and forth!

## Features

- 🤖 Two-way AI conversation (Claude ↔ GPT-4o)
- 💬 Interactive CLI for custom discussion topics
- 🔄 Configurable conversation turns
- 🔐 Secure API key management via environment variables
- 🌐 FastAPI web interface for image analysis (FastAPI_implem.py)

## Scripts

### API_2_Api_com.py
Command-line interface for bot-to-bot conversations. Simple, interactive way to watch Claude and GPT-4o discuss any topic.

### FastAPI_implem.py
FastAPI web service providing Claude-powered image analysis endpoints including:
- Image analysis and description
- Image comparison
- Text extraction from images
- General image processing

## Setup

### 1. Clone the Repository

```bash
git clone git@github.com:whbBBHR/Bot2BotCom.git
cd Bot2BotCom
```

### 2. Create Virtual Environment

```bash
python3 -m venv Botvenv
source Botvenv/bin/activate  # On macOS/Linux
# or
Botvenv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

**Get API Keys:**
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys

## Usage

### Bot2Bot CLI (API_2_Api_com.py)

Run the interactive conversation script:

```bash
python API_2_Api_com.py
```

You'll be prompted to:
1. Enter a discussion question
2. Choose the number of conversation turns (default: 3)

**Example:**

```
================================================================================
Bot2Bot Conversation - Claude vs GPT-4o
================================================================================

Enter your discussion question: What is the future of artificial intelligence?
How many conversation turns? (default: 3): 3

Starting 3-turn conversation...

=== Bot2Bot Conversation ===

Claude:
[Claude's response...]
--------------------------------------------------------------------------------

GPT:
[GPT-4o's response...]
--------------------------------------------------------------------------------
...
```

### FastAPI Image Analysis (FastAPI_implem.py)

Run the FastAPI server:

```bash
python FastAPI_implem.py
```

Server runs on `http://localhost:8000`

**API Endpoints:**
- `POST /analyze-image/` - Analyze and describe an image
- `POST /compare-images/` - Compare two images
- `POST /extract-text/` - Extract text from an image
- `POST /process-image/` - General image processing

View interactive API docs at `http://localhost:8000/docs`

## Requirements

- Python 3.9+
- anthropic
- openai
- python-dotenv
- fastapi (for FastAPI_implem.py)
- uvicorn (for FastAPI_implem.py)
- pillow (for FastAPI_implem.py)

=== Bot2Bot Conversation ===

Claude:
[Claude's response...]
--------------------------------------------------------------------------------

GPT:
[GPT-4o's response...]
--------------------------------------------------------------------------------
...
```

## Requirements

- Python 3.9+
- anthropic
- openai
- python-dotenv

## Security

⚠️ **Never commit your `.env` file to version control!** It contains sensitive API keys.

The `.gitignore` file is configured to exclude:
- `.env`
- `Botvenv/`
- `__pycache__/`

## License

MIT

## Contributing

Pull requests welcome! Feel free to open issues for bugs or feature requests.
