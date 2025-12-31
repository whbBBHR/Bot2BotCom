# Bot2Bot AI Coding Agent Instructions

## Project Overview
Bot2Bot is a dual-purpose Python project for multi-provider AI interaction:
1. **CLI Interface** (`API_2_Api_com.py`) - Command-line bot-to-bot conversations between Claude and GPT-4o
2. **FastAPI Web Service** (`FastAPI_implem.py`) - Multi-provider vision API with image analysis endpoints

## Architecture & Key Components

### Multi-Provider Pattern
- **Separate Client Initialization**: Claude (`anthropic.Anthropic`) and OpenAI (`openai.OpenAI`) clients initialized separately with env vars
- **Model Configuration Class**: `ModelConfig` centralizes all model names (Claude: SONNET/HAIKU/OPUS, OpenAI: GPT4_VISION/GPT4_TURBO/GPT4O_MINI)
- **Response Models**: Pydantic BaseModels with `model_config = {"protected_namespaces": ()}` to suppress warnings on fields like `model_used`

### FastAPI Endpoints Structure
**Claude Endpoints** (prefix `/analyze/`):
- `/analyze/single` - Single image analysis (default: claude-sonnet-4-5-20250929)
- `/analyze/multiple` - Batch image comparison
- `/analyze/url` - Download and analyze from URL
- `/ocr/extract-text` - OCR extraction (uses HAIKU for speed)

**OpenAI Endpoints** (prefix `/openai/analyze/`):
- `/openai/analyze/single` - Single image with gpt-4o (default: gpt-4o)
- `/openai/analyze/multiple` - Batch image comparison

**Comparison Endpoint** (NEW):
- `/compare/analyze/single` - Concurrent Claude vs OpenAI analysis with timing comparison

## Critical Implementation Patterns

### Image Processing Pipeline
```python
# Standard flow: read → resize → encode → API call
image_bytes = await file.read()
image_bytes = await resize_image_if_needed(image_bytes, max_size=1568)
base64_image = encode_image(image_bytes)
media_type = get_image_media_type(image_bytes)
```

### Concurrent Execution
- Use `asyncio.gather()` for parallel Claude + OpenAI calls (see `/compare/analyze/single`)
- Enables performance comparison and faster dual-provider responses

### Error Handling Convention
- Catch exceptions, log traceback to stderr with `print(traceback.format_exc(), file=sys.stderr)`
- Return HTTPException with 500 status for API errors, 400 for validation errors

### Text Extraction from Claude Responses
```python
def extract_text_from_response(response_content) -> str:
    for block in response_content:
        if isinstance(block, TextBlock):
            return block.text
    return ""
```
Claude returns `MessageParam` objects, not plain strings.

## CLI Conversation Flow (API_2_Api_com.py)

### Overview
Implements turn-based bot-to-bot conversation where Claude and GPT-4o take turns responding to each other.

### Key Function: `chatbot_conversation(initial_prompt, turns=3)`
- **Round structure**: Each turn = Claude responds, then GPT responds
- **History tracking**: Full conversation stored in list with speaker/message pairs
- **Message passing**: Each bot receives the previous bot's output as new prompt
- **Text extraction**: Both clients return different structures:
  - Claude: Extract from `.content[i].text` (MessageParam objects)
  - GPT: Direct from `.choices[0].message.content`

### Typical Flow Example
```
User Input: "Discuss the future of AI"
│
├─ Turn 1:
│  ├─ Claude responds to "Discuss the future of AI"
│  └─ GPT responds to Claude's message
│
├─ Turn 2:
│  ├─ Claude responds to GPT's message
│  └─ GPT responds to Claude's message
│
└─ Turn 3: Same pattern repeats
```

### Customization Options
- **turns parameter**: Control conversation length (default 3 turns = 6 total messages)
- **initial_prompt**: Set discussion topic (interactive input via `input()`)
- **Model selection**: Currently hardcoded to Sonnet + gpt-4o, easily swappable via `ModelConfig`

## Performance Tuning Strategies

### Model Selection by Use Case
| Use Case | Claude Model | OpenAI Model | Why |
|----------|-------------|-------------|-----|
| **Speed Priority** | HAIKU | gpt-4o-mini | Fastest response times |
| **Balanced (Default)** | SONNET | gpt-4o | Best latency/quality trade-off |
| **Quality Priority** | OPUS | gpt-4o | Highest accuracy responses |
| **OCR Tasks** | HAIKU | - | Fast text extraction |

### Image Optimization
```python
# Resize large images BEFORE API calls (max 1568px)
# This reduces token consumption and processing time
image_bytes = await resize_image_if_needed(image_bytes, max_size=1568)

# For batch operations, resize all images upfront in a loop
for file in files:
    image_bytes = await file.read()
    image_bytes = await resize_image_if_needed(image_bytes)  # Reduces API latency by ~20-30%
```

### Concurrent Processing Patterns
```python
# Bad: Sequential calls (slow)
claude_result = await get_claude_analysis()
openai_result = await get_openai_analysis()
# Total time: ~8 seconds (both serial)

# Good: Parallel calls with asyncio.gather() (fast)
results = await asyncio.gather(
    get_claude_analysis(),
    get_openai_analysis()
)
# Total time: ~5 seconds (concurrent, limited by slower provider)
```

### Cost-Performance Trade-offs
- **Use gpt-4o-mini**: 60% cheaper than gpt-4o, only 10-15% slower
- **Use Claude HAIKU**: 90% cheaper than SONNET, suitable for simple OCR/text extraction
- **Batch operations**: Process multiple images in single API call vs individual calls reduces overhead

### Response Caching Opportunity
- Store `processing_time` values for identical prompts/images
- Cache `/health` endpoint responses (rarely changes)
- Consider implementing Redis for frequently analyzed images

## Development Workflow

### Setup
```bash
source Botvenv/bin/activate
pip install -r requirements.txt
```

### Running Services
- **CLI conversation**: `python API_2_Api_com.py`
- **FastAPI server**: `python FastAPI_implem.py` (runs on http://0.0.0.0:8000)
- **API docs**: http://localhost:8000/docs (Swagger UI)

### Testing Endpoints
```bash
# Create test image
python /tmp/create_test_image.py

# Test Claude
curl -X POST http://localhost:8000/analyze/single \
  -F "file=@/tmp/test_image.jpg" \
  -F "prompt=Describe this"

# Test Comparison (concurrent)
curl -X POST http://localhost:8000/compare/analyze/single \
  -F "file=@/tmp/test_image.jpg" \
  -F "claude_model=claude-sonnet-4-5-20250929" \
  -F "openai_model=gpt-4o"
```

## Configuration & Dependencies

### Environment Variables (`.env`)
- `ANTHROPIC_API_KEY` - Claude API authentication
- `OPENAI_API_KEY` - OpenAI API authentication
- Loaded via `dotenv.load_dotenv()` at startup

### Key Dependencies
- **anthropic** - Claude SDK (types: TextBlock, MessageParam)
- **openai** - OpenAI SDK
- **fastapi/uvicorn** - Web framework and server
- **pydantic** - Request/response validation
- **pillow** - Image processing (resize, format detection)
- **httpx** - Async HTTP client for URL downloads

## Project-Specific Conventions

1. **Protected Namespaces**: All Pydantic models include `model_config = {"protected_namespaces": ()}` to allow fields starting with "model_"
2. **Async Functions**: All FastAPI handlers use `async def` with `await` for I/O operations
3. **File Handling**: Use `BytesIO` for in-memory image buffers; always call `await file.read()` for uploads
4. **Timing**: Include `processing_time` in all response models using `(datetime.now() - start_time).total_seconds()`
5. **Model Parameters**: Pass model strings directly (not enum names) - FastAPI Form defaults inject model names as strings

## Common Pitfalls to Avoid

- ❌ Using `ModelConfig.GPT4_VISION` as default in OpenAI endpoints (causes wrong model in Swagger UI) → Use string `"gpt-4o"`
- ❌ Calling Claude with gpt-4o model string → Results in 404 error; ensure correct client/model pairing
- ❌ Synchronous image I/O in async functions → Use `await file.read()`, not `file.read()`
- ❌ Forgetting to resize images before sending to Claude → Max size ~1568px enforced by `resize_image_if_needed()`

## Debugging Tips

- Port 8000 conflicts? Kill with: `lsof -i :8000 | grep -v COMMAND | awk '{print $2}' | xargs kill -9`
- Check API key availability: `curl http://localhost:8000/health`
- Inspect server logs for stderr output containing debug traces
- Use `jq` for pretty-printing JSON responses: `curl ... | jq .`
