"""
FastAPI application for Medical LLM Assistant.
Provides REST API endpoints for model interaction.
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import logging
import time

# Disable HuggingFace warnings about authentication
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow downloads but use cache
os.environ['HF_HUB_OFFLINE'] = '0'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from HuggingFace
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
logger.info(f"Project root: {PROJECT_ROOT}")

# Initialize FastAPI app
app = FastAPI(
    title="Medical LLM Assistant API",
    description="REST API for Medical Question-Answering using fine-tuned LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for medical queries"""
    query: str = Field(..., description="Medical question to answer", min_length=5)
    max_tokens: int = Field(150, description="Maximum tokens to generate", ge=30, le=512)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)
    top_p: float = Field(0.95, description="Nucleus sampling parameter", ge=0.1, le=1.0)


class QueryResponse(BaseModel):
    """Response model for queries"""
    query: str
    response: str
    processing_time: float
    tokens_generated: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str


class ModelInfo(BaseModel):
    """Model information response"""
    base_model: str
    fine_tuned: bool
    parameter_count: Optional[str]
    device: str


# Global model variables
model = None
tokenizer = None
model_config = {
    "base_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "lora_adapter_path": str(PROJECT_ROOT / "notebooks" / "medical_llm_final"),
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer
    
    if model is not None:
        logger.info("Model already loaded")
        return
    
    try:
        base_model = model_config['base_model_name']
        logger.info(f"Loading model: {base_model}")
        logger.info("Using HuggingFace cache (no re-download needed)")
        
        # Load tokenizer (uses default HuggingFace cache)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model
            # cache_dir not specified = uses ~/.cache/huggingface (where model already exists)
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type='nf8',
            bnb_8bit_compute_dtype=torch.float16
        )
        
        # Load base model with quantization (uses default HuggingFace cache)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
            # cache_dir not specified = uses ~/.cache/huggingface (already has 2.1GB model)
        )
        logger.info("✓ Base model loaded from cache (no download)")
        
        # Try to load LoRA adapter
        try:
            adapter_path = model_config['lora_adapter_path']
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            
            # Verify adapter files exist
            adapter_config = Path(adapter_path) / "adapter_config.json"
            if not adapter_config.exists():
                raise FileNotFoundError(f"adapter_config.json not found at {adapter_path}")
                
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info(" LoRA adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LoRA adapter: {e}")
            logger.warning("Using base model only (not fine-tuned)")
        
        model.eval()
        
        # Enable faster attention if available
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True
            
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """API startup - model loads on first request"""
    logger.info("Starting up Medical LLM Assistant API...")
    logger.info("Model will be loaded on first query request")
    logger.info("API ready to accept requests")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Medical LLM Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=model_config['base_model_name']
    )


@app.get("/info", response_model=ModelInfo)
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_info = f"{total_params:,} total, {trainable_params:,} trainable"
    except:
        param_info = "Unknown"
    
    return ModelInfo(
        base_model=model_config['base_model_name'],
        fine_tuned=True,  # Assuming fine-tuned version
        parameter_count=param_info,
        device=model_config['device']
    )


@app.post("/load")
async def load_model_endpoint():
    """Manually trigger model loading"""
    if model is not None:
        return {"status": "already_loaded", "message": "Model is already loaded"}
    
    try:
        logger.info("Manual model load triggered via /load endpoint")
        load_model()
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/performance")
async def get_performance_tips():
    """Get performance optimization info"""
    device = "CPU" if not torch.cuda.is_available() else f"GPU ({torch.cuda.get_device_name(0)})"
    
    tips = {
        "current_device": device,
        "model_loaded": model is not None,
        "tips": []
    }
    
    if not torch.cuda.is_available():
        tips["tips"].extend([
            "⚠ Running on CPU - responses will be slower (30-90 seconds)",
            "💡 Tip: Use shorter questions for faster responses",
            "💡 Tip: Reduce max_tokens (default 150) for speed",
            "🚀 Best: Install CUDA-enabled GPU for 10x faster inference"
        ])
    else:
        tips["tips"].extend([
            "✓ GPU acceleration enabled",
            "⚡ Expected response time: 5-15 seconds"
        ])
    
    return tips


@app.post("/query", response_model=QueryResponse)
async def query_model(request: QueryRequest):
    """
    Query the medical assistant model.
    
    Args:
        request: QueryRequest with query and generation parameters
        
    Returns:
        QueryResponse with generated answer
    """
    # Load model on first request if not already loaded
    if model is None or tokenizer is None:
        logger.info("Model not loaded yet, loading now...")
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {str(e)}")
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing query (max_tokens={request.max_tokens}, temp={request.temperature})")
        
        # Format prompt
        prompt = f"### Instruction:\n{request.query}\n\n### Response:\n"
        
        # Tokenize with optimized length for CPU speed
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256  # Reduced from 384 for faster CPU processing
        ).to(model.device)
        
        # Generate with CPU-optimized parameters
        with torch.inference_mode():  # Faster than no_grad()
            generation_config = GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True if request.temperature > 0.1 else False,  # Greedy if temp low
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,  # Stronger penalty for faster stopping
                num_beams=1,  # Greedy/sampling only (no beam search)
                use_cache=True,  # Enable KV cache
                early_stopping=True,  # Stop when EOS generated
                no_repeat_ngram_size=3  # Prevent repetition loops
            )
            
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response.strip()
        
        processing_time = time.time() - start_time
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        
        logger.info(f"Query completed in {processing_time:.2f}s, generated {tokens_generated} tokens ({tokens_generated/processing_time:.1f} tokens/s)")
        
        return QueryResponse(
            query=request.query,
            response=response,
            processing_time=round(processing_time, 3),
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_query", response_model=List[QueryResponse])
async def batch_query(requests: List[QueryRequest]):
    """
    Process multiple queries in batch.
    
    Args:
        requests: List of QueryRequest objects
        
    Returns:
        List of QueryResponse objects
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 queries per batch")
    
    responses = []
    for req in requests:
        try:
            response = await query_model(req)
            responses.append(response)
        except Exception as e:
            logger.error(f"Error in batch query: {e}")
            # Add error response
            responses.append(QueryResponse(
                query=req.query,
                response=f"Error: {str(e)}",
                processing_time=0.0,
                tokens_generated=0
            ))
    
    return responses


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
