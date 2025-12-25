from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import anthropic
import base64
from io import BytesIO
from PIL import Image
import httpx
from datetime import datetime
import asyncio

app = FastAPI(title="Claude Vision API", version="1.0.0")

# Initialize Claude client
claude_client = anthropic.Anthropic(api_key="your-api-key-here")

# Response models
class ImageAnalysisResponse(BaseModel):
    analysis: str
    model_used: str
    processing_time: float
    image_count: int
    timestamp: str

class MultiImageResponse(BaseModel):
    comparison: str
    model_used: str
    processing_time: float
    images_processed: int

# Configuration
class ModelConfig:
    SONNET = "claude-sonnet-4-5-20250929"  # Best balance
    HAIKU = "claude-haiku-4-5-20251001"     # Fastest
    OPUS = "claude-opus-4-20250514"         # Highest quality

def encode_image(image_bytes: bytes) -> str:
    """Convert image bytes to base64"""
    return base64.b64encode(image_bytes).decode('utf-8')

def get_image_media_type(image_bytes: bytes) -> str:
    """Detect image format"""
    img = Image.open(BytesIO(image_bytes))
    format_map = {
        'JPEG': 'image/jpeg',
        'PNG': 'image/png',
        'GIF': 'image/gif',
        'WEBP': 'image/webp'
    }
    return format_map.get(img.format, 'image/jpeg')

async def resize_image_if_needed(image_bytes: bytes, max_size: int = 1568) -> bytes:
    """Resize image if too large (Claude has size limits)"""
    img = Image.open(BytesIO(image_bytes))
    
    # Check if resize needed
    if max(img.size) > max_size:
        # Calculate new size maintaining aspect ratio
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        buffer = BytesIO()
        img.save(buffer, format=img.format or 'PNG')
        return buffer.getvalue()
    
    return image_bytes

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/analyze/single", response_model=ImageAnalysisResponse)
async def analyze_single_image(
    file: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail"),
    model: str = Form(ModelConfig.SONNET),
    max_tokens: int = Form(1024)
):
    """
    Analyze a single image
    
    - **file**: Image file (JPEG, PNG, GIF, WEBP)
    - **prompt**: Analysis instructions
    - **model**: sonnet (balanced), haiku (fast), or opus (quality)
    - **max_tokens**: Maximum response length
    """
    start_time = datetime.now()
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image_bytes = await resize_image_if_needed(image_bytes)
        
        # Encode image
        base64_image = encode_image(image_bytes)
        media_type = get_image_media_type(image_bytes)
        
        # Call Claude API
        response = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ImageAnalysisResponse(
            analysis=response.content[0].text,
            model_used=model,
            processing_time=processing_time,
            image_count=1,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze/multiple", response_model=MultiImageResponse)
async def analyze_multiple_images(
    files: List[UploadFile] = File(...),
    prompt: str = Form("Compare and analyze these images"),
    model: str = Form(ModelConfig.SONNET),
    max_tokens: int = Form(2048)
):
    """
    Analyze and compare multiple images (up to 20)
    
    - **files**: List of image files
    - **prompt**: Analysis instructions
    - **model**: Claude model to use
    """
    start_time = datetime.now()
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images allowed")
    
    try:
        # Process all images
        image_contents = []
        
        for file in files:
            image_bytes = await file.read()
            image_bytes = await resize_image_if_needed(image_bytes)
            base64_image = encode_image(image_bytes)
            media_type = get_image_media_type(image_bytes)
            
            image_contents.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image
                }
            })
        
        # Add text prompt
        image_contents.append({
            "type": "text",
            "text": prompt
        })
        
        # Call Claude
        response = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": image_contents
            }]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return MultiImageResponse(
            comparison=response.content[0].text,
            model_used=model,
            processing_time=processing_time,
            images_processed=len(files)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze/url")
async def analyze_image_from_url(
    image_url: str = Form(...),
    prompt: str = Form("Describe this image"),
    model: str = Form(ModelConfig.SONNET)
):
    """
    Analyze an image from a URL
    """
    start_time = datetime.now()
    
    try:
        # Download image
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
        
        # Process image
        image_bytes = await resize_image_if_needed(image_bytes)
        base64_image = encode_image(image_bytes)
        media_type = get_image_media_type(image_bytes)
        
        # Call Claude
        claude_response = claude_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "analysis": claude_response.content[0].text,
            "processing_time": processing_time,
            "model_used": model
        }
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/ocr/extract-text")
async def extract_text_from_image(
    file: UploadFile = File(...),
    model: str = Form(ModelConfig.HAIKU)  # Haiku is fast for OCR
):
    """
    Extract text from image (OCR)
    """
    try:
        image_bytes = await file.read()
        image_bytes = await resize_image_if_needed(image_bytes)
        base64_image = encode_image(image_bytes)
        media_type = get_image_media_type(image_bytes)
        
        response = claude_client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract all text from this image. Preserve formatting and structure."
                    }
                ]
            }]
        )
        
        return {
            "extracted_text": response.content[0].text,
            "model_used": model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vision/medical-analysis")
async def medical_image_analysis(
    file: UploadFile = File(...),
    clinical_context: Optional[str] = Form(None),
    model: str = Form(ModelConfig.OPUS)  # Use Opus for medical accuracy
):
    """
    Analyze medical images (X-rays, MRI, CT scans)
    Note: Always use Opus for medical applications
    """
    try:
        image_bytes = await file.read()
        base64_image = encode_image(image_bytes)
        media_type = get_image_media_type(image_bytes)
        
        prompt = "Analyze this medical image. Identify key features, potential abnormalities, and relevant anatomical structures."
        if clinical_context:
            prompt += f"\n\nClinical context: {clinical_context}"
        
        prompt += "\n\nIMPORTANT: This is for educational purposes only. Always consult qualified healthcare professionals for medical diagnosis."
        
        response = claude_client.messages.create(
            model=ModelConfig.OPUS,  # Force Opus for medical
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        return {
            "analysis": response.content[0].text,
            "model_used": ModelConfig.OPUS,
            "disclaimer": "For educational purposes only. Not a substitute for professional medical advice."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/process-images")
async def batch_process_images(
    files: List[UploadFile] = File(...),
    operation: str = Form("describe"),
    model: str = Form(ModelConfig.HAIKU)  # Haiku for batch speed
):
    """
    Process multiple images independently (batch operation)
    Operations: describe, ocr, classify, detect-objects
    """
    results = []
    
    prompts = {
        "describe": "Provide a brief description of this image.",
        "ocr": "Extract any text from this image.",
        "classify": "Classify the main subject/category of this image.",
        "detect-objects": "List all objects visible in this image."
    }
    
    prompt = prompts.get(operation, prompts["describe"])
    
    # Process images concurrently
    async def process_single(file: UploadFile):
        try:
            image_bytes = await file.read()
            image_bytes = await resize_image_if_needed(image_bytes)
            base64_image = encode_image(image_bytes)
            media_type = get_image_media_type(image_bytes)
            
            response = claude_client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            return {
                "filename": file.filename,
                "result": response.content[0].text,
                "status": "success"
            }
        except Exception as e:
            return {
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            }
    
    # Process all images concurrently
    tasks = [process_single(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    return {
        "total_processed": len(results),
        "operation": operation,
        "model_used": model,
        "results": results
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "available_models": {
            "sonnet": ModelConfig.SONNET,
            "haiku": ModelConfig.HAIKU,
            "opus": ModelConfig.OPUS
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Claude Vision API",
        "endpoints": {
            "single_image": "/analyze/single",
            "multiple_images": "/analyze/multiple",
            "image_url": "/analyze/url",
            "ocr": "/ocr/extract-text",
            "medical": "/vision/medical-analysis",
            "batch": "/batch/process-images"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)