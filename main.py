"""
FastAPI-based Image Background Removal API
Optimized for performance, security, and scalability
"""

import os
import io
import uuid
import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Header, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from rembg import remove, new_session
import redis
from cachetools import TTLCache
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import onnxruntime as ort
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class for better organization
class Config:
    """Centralized configuration management"""
    ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "JahidHasan3455jskfjlfd")
    raw = os.getenv("MAX_FILE_SIZE", "10")
    value = raw.strip().split()[0]
    MAX_FILE_SIZE = int(value)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(BASE_DIR, "outputs"))
    raw1 = os.getenv("CACHE_TTL", "3600")
    first1 = raw1.strip().split()[0]
    CACHE_TTL = int(first1)
    raw_model = os.getenv("BG_REMOVAL_MODEL", "u2net")
    MODEL_NAME = raw_model.split("#", 1)[0].strip()
    raw_workers = os.getenv("MAX_WORKERS", "4")
    first_token = raw_workers.strip().split()[0]
    MAX_WORKERS = int(first_token)

    

# Initialize configuration
config = Config()

# Global variables for model and cache
# BG_REMOVAL_MODEL = "models/u2net.onnx"
bg_removal_session: Any = None
_onnx_u2net_session: Optional[ort.InferenceSession] = None
# bg_removal_session = None
memory_cache = TTLCache(maxsize=100, ttl=config.CACHE_TTL)
thread_pool = None
redis_client = None

# Pydantic models for request/response validation
class ProcessingStatus(BaseModel):
    """Model for processing status responses"""
    status: str = Field(description="Processing status")
    message: str = Field(description="Status message")
    task_id: Optional[str] = Field(None, description="Task ID for async processing")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

# Custom exceptions
class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks"""
    # Startup
    global bg_removal_session, thread_pool, redis_client
    
    logger.info("Initializing background removal API...")
    
    # Create temp directory
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    
    # Initialize thread pool for CPU-intensive tasks
    thread_pool = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    
    # Initialize Redis client (optional for caching)
    try:
        redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
        await asyncio.get_event_loop().run_in_executor(None, redis_client.ping)
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Using in-memory cache only.")
        redis_client = None
    
    # Initialize background removal model
    try:
        bg_removal_session = new_session("u2net")  #config.MODEL_NAME
        logger.info(f"Background removal model '{config.MODEL_NAME}' loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load background removal model: {e}")
        raise RuntimeError("Failed to initialize background removal model")
    
    yield
    
    # Shutdown
    logger.info("Shutting down background removal API...")
    if thread_pool:
        thread_pool.shutdown(wait=True)
    if redis_client:
        redis_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Background Removal API",
    description="High-performance API for removing backgrounds from images using AI",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this properly for production
)

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify access token for API authentication"""
    if credentials.credentials != config.ACCESS_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    # Check file extension
    file_ext = os.path.splitext(file.filename.lower())[1] if file.filename else ""
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
    
    # Check content type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Must be an image file."
        )

def generate_cache_key(image_data: bytes, model_name: str) -> str:
    """Generate cache key based on image content and model"""
    image_hash = hashlib.md5(image_data).hexdigest()
    return f"bg_removal:{model_name}:{image_hash}"

async def get_cached_result(cache_key: str) -> Optional[bytes]:
    """Get cached result from Redis or memory cache"""
    try:
        # Try Redis first
        if redis_client:
            cached = await asyncio.get_event_loop().run_in_executor(
                None, redis_client.get, cache_key
            )
            if cached:
                return bytes.fromhex(str(cached))
        
        # Fall back to memory cache
        return memory_cache.get(cache_key)
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")
        return None

async def set_cached_result(cache_key: str, result: bytes) -> None:
    """Store result in cache"""
    try:
        # Store in Redis
        if redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None, redis_client.setex, cache_key, config.CACHE_TTL, result.hex()
            )
        
        # Store in memory cache
        memory_cache[cache_key] = result
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Apply quality enhancements to the image"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply subtle enhancements
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # Slight sharpening
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)  # Slight contrast boost
        
        return image
    except Exception as e:
        logger.warning(f"Image enhancement failed: {e}")
        return image

def optimize_image_for_processing(image_data: bytes) -> Image.Image:
    """Optimize image for background removal processing"""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        # Resize if too large (optimization for processing speed)
        max_dimension = 2048
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            # Ensure new_size is exactly (width, height)
            if len(new_size) != 2:
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Image resized to {new_size} for optimization")
        
        return image
    except Exception as e:
        raise ImageProcessingError(f"Failed to optimize image: {str(e)}")

def process_background_removal(image_data: bytes, enhance: bool = True) -> bytes:
    """Process background removal with optimizations"""
    try:
        # Optimize image for processing
        image = optimize_image_for_processing(image_data)
        
        # Convert PIL image back to bytes for rembg
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        optimized_image_data = img_byte_arr.getvalue()
        
        # Remove background using the global session
        result_data = remove(optimized_image_data, session=bg_removal_session)
        
        # Post-process if enhancement is requested
        if enhance:
            # Ensure result_data is bytes before using BytesIO
            if isinstance(result_data, bytes):
                result_image = Image.open(io.BytesIO(result_data))
            elif isinstance(result_data, Image.Image):
                result_image = result_data
            else:
                raise ImageProcessingError("Unexpected result_data type for enhancement")
            result_image = enhance_image_quality(result_image)
            
            # Convert back to bytes
            enhanced_byte_arr = io.BytesIO()
            result_image.save(enhanced_byte_arr, format='PNG', optimize=True)
            result_data = enhanced_byte_arr.getvalue()
        
        # Force garbage collection to free memory
        gc.collect()

        # Ensure result_data is bytes before returning
        if isinstance(result_data, bytes):
            return result_data
        elif isinstance(result_data, Image.Image):
            buf = io.BytesIO()
            result_data.save(buf, format='PNG')
            return buf.getvalue()
        elif isinstance(result_data, np.ndarray):
            # Convert ndarray to PIL Image, then to bytes
            img = Image.fromarray(result_data)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
        else:
            raise ImageProcessingError("Unexpected result_data type in background removal")

    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}")
        raise ImageProcessingError(f"Background removal failed: {str(e)}")




# def create_fresh_onnx_session() -> ort.InferenceSession:
#     """
#     Create a brand new ONNX Runtime session, completely separate from any rembg code.
    
#     This function is designed to be completely isolated from any other background
#     removal libraries to prevent the conflicts you've been experiencing.
#     """
#     try:
#         logger.info(f"Creating new ONNX Runtime session for U2Net model at: {config.MODEL_NAME}")
        
#         # Verify the model file exists before trying to load it
#         import os
#         if not os.path.exists(config.MODEL_NAME):
#             raise FileNotFoundError(f"U2Net model file not found at {config.MODEL_NAME}")
        
#         # Configure ONNX Runtime session with performance optimizations
#         session_options = ort.SessionOptions()
#         session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
#         # Set up execution providers (GPU if available, otherwise CPU)
#         providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
#         # Create the session - this is a pure ONNX Runtime session
#         fresh_session = ort.InferenceSession(
#             config.MODEL_NAME, 
#             sess_options=session_options,
#             providers=providers
#         )
        
#         # Verify this is actually an ONNX Runtime session by checking its type
#         logger.info(f"Session type: {type(fresh_session)}")
#         logger.info(f"Session providers: {fresh_session.get_providers()}")
        
#         # Double-check that our session has the methods we need
#         if not hasattr(fresh_session, 'get_inputs'):
#             raise RuntimeError("Session doesn't have get_inputs method - this shouldn't happen with ONNX Runtime")
        
#         if not hasattr(fresh_session, 'run'):
#             raise RuntimeError("Session doesn't have run method - this shouldn't happen with ONNX Runtime")
        
#         # Get input information to verify the model loaded correctly
#         input_info = fresh_session.get_inputs()[0]
#         logger.info(f"Model input name: {input_info.name}")
#         logger.info(f"Model input shape: {input_info.shape}")
        
#         return fresh_session
        
#     except FileNotFoundError as e:
#         error_msg = f"Model file not found: {str(e)}"
#         logger.error(error_msg)
#         raise ImageProcessingError(error_msg) from e
#     except Exception as e:
#         error_msg = f"Failed to create ONNX Runtime session: {str(e)}"
#         logger.error(error_msg)
#         raise ImageProcessingError(error_msg) from e

# def get_isolated_onnx_session() -> ort.InferenceSession:
#     """
#     Get or create our isolated ONNX Runtime session.
    
#     This function ensures we always work with a pure ONNX Runtime session,
#     never any session from rembg or other libraries.
#     """
#     global _onnx_u2net_session
    
#     # If we don't have a session yet, or if somehow we got the wrong type, create a new one
#     if _onnx_u2net_session is None:
#         logger.info("Creating new ONNX session for the first time")
#         _onnx_u2net_session = create_fresh_onnx_session()
#     else:
#         # Double-check that we still have the right type of session
#         # This prevents issues if some other code overwrote our session
#         if not isinstance(_onnx_u2net_session, ort.InferenceSession):
#             logger.warning(f"Session was overwritten with wrong type: {type(_onnx_u2net_session)}")
#             logger.info("Recreating ONNX session")
#             _onnx_u2net_session = create_fresh_onnx_session()
    
#     return _onnx_u2net_session

# def preprocess_image_for_u2net(image: Image.Image) -> tuple[np.ndarray, tuple]:
#     """
#     Preprocess image for U2Net model input.
    
#     U2Net models typically expect:
#     - Input shape: (batch_size, channels, height, width) = (1, 3, 320, 320)
#     - Pixel values normalized to [0, 1] range
#     - RGB color format
#     - Channels-first ordering (CHW instead of HWC)
    
#     Returns:
#         Tuple containing:
#         - Preprocessed image array ready for model input
#         - Original image size for restoring output dimensions
#     """
#     # Convert to RGB if the image is in a different format
#     # This handles RGBA, grayscale, CMYK, etc.
#     if image.mode != 'RGB':
#         logger.debug(f"Converting image from {image.mode} to RGB")
#         image = image.convert('RGB')
    
#     # Store original dimensions for later restoration
#     original_size = image.size  # PIL uses (width, height) format
#     logger.debug(f"Original image size: {original_size}")
    
#     # Resize to the size expected by U2Net (typically 320x320)
#     target_size = (320, 320)
#     image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    
#     # Convert PIL image to numpy array
#     # This gives us shape (height, width, channels) with values in [0, 255]
#     image_array = np.array(image_resized, dtype=np.float32)
    
#     # Normalize pixel values to [0, 1] range as expected by the model
#     image_array = image_array / 255.0
    
#     # Transpose from HWC to CHW format (channels first)
#     # Deep learning models typically expect channels first
#     image_array = np.transpose(image_array, (2, 0, 1))
    
#     # Add batch dimension: (C, H, W) -> (1, C, H, W)
#     # Models expect batch processing even for single images
#     image_array = np.expand_dims(image_array, axis=0)
    
#     logger.debug(f"Preprocessed image shape: {image_array.shape}")
    
#     return image_array, original_size

# def postprocess_u2net_output(mask: np.ndarray, original_size: tuple) -> np.ndarray:
#     """
#     Convert U2Net model output back to a usable image mask.
    
#     U2Net outputs are typically probability maps that need to be:
#     - Resized back to original image dimensions
#     - Converted from float probabilities to uint8 pixel values
#     - Cleaned up to create sharp alpha channel boundaries
    
#     Args:
#         mask: Raw model output (probability map)
#         original_size: Target size for the final mask (width, height)
    
#     Returns:
#         Processed mask ready to use as an alpha channel
#     """
#     logger.debug(f"Processing mask with shape: {mask.shape}")
    
#     # Remove batch dimension if present
#     # Model output might be (1, H, W) or (1, 1, H, W)
#     while mask.ndim > 2 and mask.shape[0] == 1:
#         mask = mask[0]
    
#     # If mask still has multiple channels, take the first one
#     # Some models output multiple probability maps
#     if mask.ndim == 3:
#         mask = mask[0]
    
#     # Ensure mask values are in valid range [0, 1]
#     mask = np.clip(mask, 0, 1)
    
#     # Convert to 8-bit values for standard image processing
#     mask_uint8 = (mask * 255).astype(np.uint8)
    
#     # Resize back to original image dimensions
#     # OpenCV resize expects (width, height) which matches PIL's format
#     mask_resized = cv2.resize(mask_uint8, original_size, interpolation=cv2.INTER_LANCZOS4)
    
#     logger.debug(f"Final mask shape: {mask_resized.shape}")
    
#     return mask_resized

# def remove_background_with_isolated_u2net(image_data: bytes) -> bytes:
#     """
#     Remove background using our isolated U2Net ONNX implementation.
    
#     This function is completely independent of rembg and uses only
#     ONNX Runtime for inference. It handles the complete pipeline:
#     1. Load and preprocess the input image
#     2. Run inference using our isolated ONNX session
#     3. Process the output mask
#     4. Apply the mask to create transparency
    
#     Args:
#         image_data: Input image as bytes
    
#     Returns:
#         Processed image with background removed as bytes
#     """
#     try:
#         logger.info("Starting background removal with isolated U2Net")
        
#         # Get our guaranteed ONNX Runtime session
#         session = get_isolated_onnx_session()
        
#         # Verify we have the correct session type
#         if not isinstance(session, ort.InferenceSession):
#             raise RuntimeError(f"Wrong session type: {type(session)}. Expected ort.InferenceSession")
        
#         # Load the input image
#         image = Image.open(io.BytesIO(image_data))
#         logger.debug(f"Loaded image: {image.size}, {image.mode}")
        
#         # Preprocess the image for the model
#         processed_input, original_size = preprocess_image_for_u2net(image)
        
#         # Get the model's input name
#         input_name = session.get_inputs()[0].name
#         logger.debug(f"Model input name: {input_name}")
        
#         # Run inference
#         logger.debug("Running model inference")
#         outputs = session.run(None, {input_name: processed_input})
        
#         # Extract the mask from model outputs
#         # U2Net typically returns the main prediction as the first output
#         mask = outputs[0]
#         logger.debug(f"Model output shape: {mask.shape}")
        
#         # Process the raw mask into a usable format
#         processed_mask = postprocess_u2net_output(mask, original_size)
        
#         # Apply the mask to create transparency
#         # Convert original image to RGBA to support alpha channel
#         original_rgba = image.convert('RGBA')
#         original_array = np.array(original_rgba)
        
#         # Replace the alpha channel with our processed mask
#         # This creates the background removal effect
#         original_array[:, :, 3] = processed_mask
        
#         # Convert back to PIL Image
#         result_image = Image.fromarray(original_array, 'RGBA')
        
#         # Save to bytes for return
#         output_buffer = io.BytesIO()
#         result_image.save(output_buffer, format='PNG', optimize=True)
        
#         logger.info("Background removal completed successfully")
#         return output_buffer.getvalue()
        
#     except Exception as e:
#         logger.error(f"Background removal failed: {str(e)}")
#         raise ImageProcessingError(f"Background removal failed: {str(e)}") from e

# def process_background_removal(image_data: bytes, enhance: bool = True) -> bytes:
#     """
#     Main function for background removal processing.
    
#     This function maintains the same interface as your original code
#     but uses our completely isolated ONNX Runtime implementation.
#     All conflicts with rembg should be resolved.
    
#     Args:
#         image_data: Input image as bytes
#         enhance: Whether to apply post-processing enhancements
    
#     Returns:
#         Processed image with background removed as bytes
#     """
#     try:
#         logger.info("Starting background removal process")
        
#         # Try to optimize the image if the function exists
#         optimized_image_data = image_data
#         try:
#             image = optimize_image_for_processing(image_data)
#             img_byte_arr = io.BytesIO()
#             image.save(img_byte_arr, format='PNG')
#             optimized_image_data = img_byte_arr.getvalue()
#             logger.debug("Applied image optimization")
#         except NameError:
#             logger.debug("optimize_image_for_processing not available, using original image")
        
#         # Remove background using our isolated implementation
#         result_data = remove_background_with_isolated_u2net(optimized_image_data)
        
#         # Apply enhancement if requested and available
#         if enhance:
#             try:
#                 if isinstance(result_data, bytes):
#                     result_image = Image.open(io.BytesIO(result_data))
#                 elif isinstance(result_data, Image.Image):
#                     result_image = result_data
#                 else:
#                     raise ImageProcessingError("Unexpected result_data type for enhancement")
                
#                 # Try to enhance if the function exists
#                 try:
#                     result_image = enhance_image_quality(result_image)
#                     logger.debug("Applied image enhancement")
#                 except NameError:
#                     logger.debug("enhance_image_quality not available, skipping enhancement")
                
#                 # Convert back to bytes
#                 enhanced_byte_arr = io.BytesIO()
#                 result_image.save(enhanced_byte_arr, format='PNG', optimize=True)
#                 result_data = enhanced_byte_arr.getvalue()
                
#             except Exception as e:
#                 logger.warning(f"Enhancement failed: {str(e)}, continuing without enhancement")
        
#         # Clean up memory
#         gc.collect()
        
#         # Ensure we return bytes
#         if isinstance(result_data, bytes):
#             return result_data
#         elif isinstance(result_data, Image.Image):
#             buf = io.BytesIO()
#             result_data.save(buf, format='PNG')
#             return buf.getvalue()
#         elif isinstance(result_data, np.ndarray):
#             img = Image.fromarray(result_data)
#             buf = io.BytesIO()
#             img.save(buf, format='PNG')
#             return buf.getvalue()
#         else:
#             raise ImageProcessingError("Unexpected result_data type in background removal")
    
#     except Exception as e:
#         logger.error(f"Background removal process failed: {str(e)}")
#         raise ImageProcessingError(f"Background removal failed: {str(e)}") from e

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced Background Removal API",
        "version": "2.0.0",
        "status": "operational",
        "model": config.MODEL_NAME
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": str(bg_removal_session is not None)
    }

@app.post("/remove-background", response_class=StreamingResponse)
async def remove_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file with background to remove"),
    enhance: bool = True,
    token: str = Depends(verify_token)
):
    """
    Remove background from uploaded image with advanced optimizations
    
    Features:
    - Intelligent caching for repeated requests
    - Image quality enhancement
    - Memory optimization
    - Error handling and logging
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate file
        validate_image_file(file)
        
        # Check file size
        file_size = 0
        image_data = bytearray()
        
        # Read file in chunks to handle large files efficiently
        while chunk := await file.read(8192):  # 8KB chunks
            file_size += len(chunk)
            if file_size > config.MAX_FILE_SIZE * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            image_data.extend(chunk)
        
        image_data = bytes(image_data)
        
        # Generate cache key
        cache_key = generate_cache_key(image_data, config.MODEL_NAME)
        
        # Check cache first
        cached_result = await get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Cache hit for key: {cache_key}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return StreamingResponse(
                io.BytesIO(cached_result),
                media_type="image/png",
                headers={
                    "X-Processing-Time": str(processing_time),
                    "X-Cache-Hit": "true",
                    "Content-Disposition": f'attachment; filename="no_bg_{file.filename}"'
                }
            )
        
        # Process image in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result_data = await loop.run_in_executor(
            thread_pool,
            process_background_removal,
            image_data,
            enhance
        )
        
        # Cache the result asynchronously
        background_tasks.add_task(set_cached_result, cache_key, result_data)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Background removal completed in {processing_time:.2f}s")
        
        return StreamingResponse(
            io.BytesIO(result_data),
            media_type="image/png",
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Cache-Hit": "false",
                "Content-Disposition": f'attachment; filename="no_bg_{file.filename}"'
            }
        )
        
    except HTTPException:
        raise
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in background removal: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )

# @app.post("/remove-background-async", response_model=ProcessingStatus)
# async def remove_background_async(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     enhance: bool = True,
#     token: str = Depends(verify_token)
# ):
#     """
#     Asynchronous background removal for better scalability
#     Returns immediately with a task ID for status checking
#     """
#     try:
#         validate_image_file(file)
        
#         # Generate task ID
#         task_id = str(uuid.uuid4())
        
#         # Store task status
#         task_status = {
#             "status": "processing",
#             "created_at": datetime.utcnow().isoformat(),
#             "filename": file.filename
#         }
        
#         if redis_client:
#             await asyncio.get_event_loop().run_in_executor(
#                 None, redis_client.setex, f"task:{task_id}", 3600, json.dumps({"status": task_status})
#             )
        
#         # Add background task for processing
#         background_tasks.add_task(
#             process_async_background_removal,
#             task_id,
#             await file.read(),
#             file.filename or "",
#             enhance
#         )
        
#         return ProcessingStatus(
#             status="accepted",
#             message="Background removal task queued for processing",
#             task_id=task_id,
#             processing_time=None
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error in async background removal: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to queue background removal task")

# async def process_async_background_removal(
#     task_id: str,
#     image_data: bytes,
#     filename: str,
#     enhance: bool
# ):
#     """Background task for processing image asynchronously"""
#     try:
#         # Process the image
#         loop = asyncio.get_event_loop()
#         result_data = await loop.run_in_executor(
#             thread_pool,
#             process_background_removal,
#             image_data,
#             enhance
#         )
        
#         # Store result temporarily (in production, use cloud storage)
#         result_path = os.path.join(config.TEMP_DIR, f"{task_id}.png")
#         async with aiofiles.open(result_path, 'wb') as f:
#             await f.write(result_data)
        
#         # Update task status
#         if redis_client:
#             task_status = {
#                 "status": "completed",
#                 "result_path": result_path,
#                 "completed_at": datetime.utcnow().isoformat()
#             }
#             await asyncio.get_event_loop().run_in_executor(
#                 None, redis_client.setex, f"task:{task_id}", 3600, json.dumps({"status": task_status})
#             )
        
#         logger.info(f"Async task {task_id} completed successfully")
        
#     except Exception as e:
#         logger.error(f"Async task {task_id} failed: {str(e)}")
#         if redis_client:
#             task_status = {
#                 "status": "failed",
#                 "error": str(e),
#                 "failed_at": datetime.utcnow().isoformat()
#             }
#             await asyncio.get_event_loop().run_in_executor(
#                 None, redis_client.setex, f"task:{task_id}", 3600, json.dumps({"status": task_status})
#             )

# @app.get("/task-status/{task_id}", response_model=ProcessingStatus)
# async def get_task_status(task_id: str, token: str = Depends(verify_token)):
#     """Get the status of an async background removal task"""
#     try:
#         if not redis_client:
#             raise HTTPException(status_code=503, detail="Task tracking not available")
        
#         task_status = await asyncio.get_event_loop().run_in_executor(
#             None, redis_client.get, f"task:{task_id}"
#         )
        
#         if not task_status:
#             raise HTTPException(status_code=404, detail="Task not found")
#         if isinstance(task_status, (bytes, bytearray)):
#             status_dict: Any = json.loads(task_status.decode("utf-8"))
#         else:
#             status_dict: Any = task_status
#         # No need to parse again; status_dict is already a dict
#         logger.info("status_dict: ", status_dict)
#         return ProcessingStatus(
#             status=status_dict["status"],
#             message=f"Task is {status_dict['status']}",
#             task_id=task_id,
#             processing_time=None
#         )
#     except json.JSONDecodeError:
#         logger.error("Bad JSON in Redis for task %s: %r", task_id)
#         raise HTTPException(status_code=500, detail="Corrupt task data")    
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error retrieving task status: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve task status")


@app.post("/remove-background-async", response_model=ProcessingStatus)
async def remove_background_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enhance: bool = True,
    token: str = Depends(verify_token)
):
    """
    Queue a background removal task. Returns immediately with a task_id.
    """
    # 1) Validate file, size, type, etc.
    validate_image_file(file)

    # 2) Create task_id and initial status
    task_id = str(uuid.uuid4())
    initial_status = {
        "status": "processing",
        "created_at": datetime.utcnow().isoformat(),
        "filename": file.filename,
    }

    # 3) Store initial status in Redis
    if redis_client:
        await asyncio.get_event_loop().run_in_executor(
            None,
            redis_client.setex,
            f"task:{task_id}",
            config.CACHE_TTL,
            json.dumps(initial_status)
        )

    # 4) Queue the background worker
    image_bytes = await file.read()
    background_tasks.add_task(
        process_async_background_removal,
        task_id,
        image_bytes,
        file.filename or "",
        enhance
    )

    # 5) Return the accepted response
    return ProcessingStatus(
        status="accepted",
        message="Task queued for processing",
        task_id=task_id,
        processing_time=None
    )

# ----------------------------
# Background worker function
# ----------------------------
async def process_async_background_removal(
    task_id: str,
    image_data: bytes,
    filename: str,
    enhance: bool
):
    try:
        # Run the sync removal in a thread pool
        loop = asyncio.get_event_loop()
        result_data = await loop.run_in_executor(
            None,
            process_background_removal,
            image_data,
            enhance
        )

        # Save output to TEMP_DIR/<task_id>.png
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        result_path = os.path.join(config.TEMP_DIR, f"{task_id}.png")
        async with aiofiles.open(result_path, "wb") as out_f:
         await out_f.write(result_data)

        # Store completed status
        completed_status = {
            "status": "completed",
            "result_path": result_path,
            "completed_at": datetime.utcnow().isoformat()
        }
        if redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None,
                redis_client.setex,
                f"task:{task_id}",
                config.CACHE_TTL,
                json.dumps(completed_status)
            )
        logger.info("Async task %s completed: %s", task_id, completed_status)

    except Exception as e:
        logger.error("Async task %s failed: %s", task_id, str(e))
        failure_status = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        }
        if redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None,
                redis_client.setex,
                f"task:{task_id}",
                config.CACHE_TTL,
                json.dumps(failure_status)
            )

@app.get("/task-status/{task_id}", response_model=ProcessingStatus)
async def get_task_status(task_id: str, token: str = Depends(verify_token)):
    """
    Retrieve the current status of a background task.
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Task tracking not available")

    # 1) Fetch raw bytes or None
    raw = await asyncio.get_event_loop().run_in_executor(
        None,
        redis_client.get,
        f"task:{task_id}"
    )
    if raw is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # 2) Decode to str
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw

    # 3) Parse JSON → dict
    try:
        status_dict: Any = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Bad JSON in Redis for task %s: %r", task_id, text)
        raise HTTPException(status_code=500, detail="Corrupt task data")

    # 4) Log the parsed dict
    logger.info("status_dict for %s: %s", task_id, status_dict)

    # 5) Build ProcessingStatus
    return ProcessingStatus(
        status           = status_dict.get("status"),
        message          = f"Task is {status_dict.get('status')}",
        task_id          = task_id,
        processing_time  = status_dict.get("processing_time")
    )


@app.get("/models", response_model=Dict[str, Any])
async def get_available_models(token: str = Depends(verify_token)):
    """Get information about available background removal models"""
    return {
        "current_model": config.MODEL_NAME,
        "available_models": [
            {
                "name": "u2net",
                "description": "General purpose model with good accuracy",
                "speed": "medium"
            },
            {
                "name": "u2netp",
                "description": "Lighter version of u2net, faster processing",
                "speed": "fast"
            },
            {
                "name": "silueta",
                "description": "Optimized for people and portraits",
                "speed": "medium"
            }
        ]
    }

# Error handlers
@app.exception_handler(ImageProcessingError)
async def image_processing_error_handler(request, exc):
    return HTTPException(status_code=422, detail=str(exc))

@app.exception_handler(AuthenticationError)
async def auth_error_handler(request, exc):
    return HTTPException(status_code=401, detail=str(exc))

if __name__ == "__main__":
    # Configuration for running the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1,  # Increase for production
        log_level="info"
    )