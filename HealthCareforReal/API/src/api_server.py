#!/usr/bin/env python3
"""
Qwen LoRA API Server with Gemini API-compatible format
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .inference_engine import QwenInferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference engine
inference_engine: Optional[QwenInferenceEngine] = None

# Pydantic models matching Gemini API format
class GenerationConfig(BaseModel):
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=1)
    max_output_tokens: Optional[int] = Field(default=1024, ge=1, le=8192)
    stop_sequences: Optional[List[str]] = Field(default=None)

class Content(BaseModel):
    text: str

class Part(BaseModel):
    text: str

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|model)$")
    parts: List[Part]

class GenerateContentRequest(BaseModel):
    contents: List[Message]
    generation_config: Optional[GenerationConfig] = None
    stream: Optional[bool] = False

class Candidate(BaseModel):
    content: Message
    finish_reason: str = "STOP"
    index: int = 0

class UsageMetadata(BaseModel):
    prompt_tokens: int
    candidates_tokens: int
    total_tokens: int

class GenerateContentResponse(BaseModel):
    candidates: List[Candidate]
    usage_metadata: Optional[UsageMetadata] = None
    model_version: str = "qwen-lora-v1"

class StreamChunk(BaseModel):
    candidates: List[Candidate]
    usage_metadata: Optional[UsageMetadata] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the inference engine"""
    global inference_engine
    
    logger.info("Starting Qwen LoRA API Server...")
    
    # Initialize inference engine
    try:
        inference_engine = QwenInferenceEngine()
        await inference_engine.initialize()
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise
    
    yield
    
    # Cleanup
    if inference_engine:
        await inference_engine.cleanup()
        logger.info("Inference engine cleaned up")

# Create FastAPI app
app = FastAPI(
    title="Qwen LoRA API",
    description="Gemini API-compatible interface for Qwen LoRA models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "qwen-lora",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/v1/models")
async def list_models():
    """List available models (Gemini API compatible)"""
    return {
        "models": [
            {
                "name": "models/qwen-lora",
                "version": "1.0.0",
                "display_name": "Qwen LoRA Fine-tuned Model",
                "description": "Fine-tuned Qwen model with LoRA adapters",
                "input_token_limit": 2048,
                "output_token_limit": 1024,
                "supported_generation_methods": ["generateContent"],
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        ]
    }

@app.post("/v1/models/qwen-lora:generateContent")
async def generate_content(request: GenerateContentRequest):
    """Generate content using Qwen LoRA model (Gemini API compatible)"""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        # Convert request to internal format
        messages = []
        for content in request.contents:
            role = "user" if content.role == "user" else "assistant"
            text = " ".join([part.text for part in content.parts])
            messages.append({"role": role, "content": text})
        
        # Set generation config
        config = request.generation_config or GenerationConfig()
        
        if request.stream:
            return StreamingResponse(
                stream_generate_content(messages, config),
                media_type="application/json"
            )
        else:
            return await generate_non_stream(messages, config)
            
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_non_stream(messages: List[Dict], config: GenerationConfig) -> GenerateContentResponse:
    """Generate non-streaming response"""
    start_time = time.time()
    
    # Generate response
    response_text, token_counts = await inference_engine.generate(
        messages=messages,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_output_tokens,
        stop_sequences=config.stop_sequences
    )
    
    # Create response
    candidate = Candidate(
        content=Message(
            role="model",
            parts=[Part(text=response_text)]
        ),
        finish_reason="STOP",
        index=0
    )
    
    usage = UsageMetadata(
        prompt_tokens=token_counts.get("prompt_tokens", 0),
        candidates_tokens=token_counts.get("completion_tokens", 0),
        total_tokens=token_counts.get("total_tokens", 0)
    )
    
    return GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=usage
    )

async def stream_generate_content(messages: List[Dict], config: GenerationConfig) -> AsyncGenerator[str, None]:
    """Stream generation response"""
    try:
        async for chunk_text, is_final, token_counts in inference_engine.stream_generate(
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_output_tokens,
            stop_sequences=config.stop_sequences
        ):
            candidate = Candidate(
                content=Message(
                    role="model",
                    parts=[Part(text=chunk_text)]
                ),
                finish_reason="STOP" if is_final else "PARTIAL",
                index=0
            )
            
            chunk = StreamChunk(candidates=[candidate])
            
            if is_final and token_counts:
                chunk.usage_metadata = UsageMetadata(
                    prompt_tokens=token_counts.get("prompt_tokens", 0),
                    candidates_tokens=token_counts.get("completion_tokens", 0),
                    total_tokens=token_counts.get("total_tokens", 0)
                )
            
            yield f"data: {chunk.model_dump_json()}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_response = {"error": {"message": str(e), "type": "generation_error"}}
        yield f"data: {json.dumps(error_response)}\n\n"

@app.post("/v1/models/qwen-lora:streamGenerateContent")
async def stream_generate_content_endpoint(request: GenerateContentRequest):
    """Stream generate content endpoint (Gemini API compatible)"""
    request.stream = True
    return await generate_content(request)

# Simplified endpoint for easier backend integration
class SimpleRequest(BaseModel):
    message: str
    
class SimpleResponse(BaseModel):
    response: str

class ConversationMessage(BaseModel):
    role: str
    content: str
    
class ConversationRequest(BaseModel):
    messages: List[ConversationMessage]
    
@app.post("/chat")
async def simple_chat(request: SimpleRequest):
    """Simple chat endpoint - easier integration"""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        # Convert to message format
        messages = [{"role": "user", "content": request.message}]
        
        # Generate response with faster settings
        response_text, _ = await inference_engine.generate(
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            top_k=20,  # Reduced for speed
            max_tokens=256  # Reduced for speed
        )
        
        return SimpleResponse(response=response_text)
        
    except Exception as e:
        logger.error(f"Simple chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/conversation")
async def conversation_chat(request: ConversationRequest):
    """Chat endpoint with conversation history"""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        # Convert to message format expected by inference engine
        messages = []
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Generate response with faster settings
        response_text, _ = await inference_engine.generate(
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            top_k=20,  # Reduced for speed
            max_tokens=256  # Reduced for speed
        )
        
        return SimpleResponse(response=response_text)
        
    except Exception as e:
        logger.error(f"Conversation chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models/{model_name}")
async def get_model_info(model_name: str):
    """Get model information"""
    if model_name != "qwen-lora":
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "name": f"models/{model_name}",
        "version": "1.0.0",
        "display_name": "Qwen LoRA Fine-tuned Model",
        "description": "Fine-tuned Qwen model with LoRA adapters",
        "input_token_limit": 2048,
        "output_token_limit": 1024,
        "supported_generation_methods": ["generateContent"],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "details": str(exc)
            }
        }
    )

def main():
    """Main function to run the API server"""
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=3001,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()