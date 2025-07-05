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
from PIL import Image, ImageEnhance, UnidentifiedImageError
import numpy as np
import cv2
from rembg import remove, new_session
import redis
from cachetools import TTLCache
import aiofiles
import asyncio
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import onnxruntime as ort
from typing import Tuple
import requests

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

    MODELS_DIR = os.path.join(BASE_DIR, "models")
    raw_model = os.getenv("BG_REMOVAL_MODEL", "u2net")
    MODEL_NAME = raw_model.split("#", 1)[0].strip()

     # Define the actual model file path
    MODEL_FILE_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.onnx")

    # Model download URLs - these are the actual locations of pre-trained models
    MODEL_URLS = {
        "u2net": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "u2netpn": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netpn.onnx",
        "silueta": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
        "u2net_human_seg": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
    }

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

# def validate_image_file(file: UploadFile) -> None:
#     """Validate uploaded image file"""
#     # Check file extension
#     file_ext = os.path.splitext(file.filename.lower())[1] if file.filename else ""
#     if file_ext not in config.ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file format. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
#         )
    
#     # Check content type
#     if not file.content_type or not file.content_type.startswith('image/'):
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid content type. Must be an image file."
#         )

# async def validate_image_file(file: UploadFile) -> bytes:
#     """
#     Validate an uploaded image:
#       1) Must not be empty (400)
#       2) Must not exceed MAX_FILE_SIZE_MB (413)
#       3) Must have an allowed extension (400)
#       4) Must have an image content type (400)
#       5) Must be a loadable, non‐corrupted image (422)
#     Returns the raw bytes for downstream processing.
#     """
#     # Read all bytes
#     data = await file.read()
#     size = len(data)

#     # 1) Empty?
#     if size == 0:
#         raise HTTPException(status_code=400, detail="Empty file upload")

#     # 2) Too large?
#     max_bytes = config.MAX_FILE_SIZE * 1024 * 1024
#     if size > max_bytes:
#         raise HTTPException(
#             status_code=413,
#             detail=f"File too large. Max size is {config.MAX_FILE_SIZE_MB} MB"
#         )

#     # 3) Extension check
#     ext = os.path.splitext((file.filename or "").lower())[1]
#     if ext not in config.ALLOWED_EXTENSIONS:
#         allowed = ", ".join(config.ALLOWED_EXTENSIONS)
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file format. Allowed: {allowed}"
#         )

#     # 4) Content‐type check
#     if not file.content_type or not file.content_type.startswith("image/"):
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid content type. Must be an image file."
#         )

#     # 5) Corruption check
#     try:
#         Image.open(io.BytesIO(data)).verify()
#     except UnidentifiedImageError:
#         raise HTTPException(status_code=422, detail="Invalid or corrupted image file")
#     except Exception:
#         raise HTTPException(status_code=422, detail="Could not process image file")

#     # Rewind so later code can `await file.read()` again if needed
#     file.file.seek(0)
#     return data

async def validate_image_file(file: UploadFile) -> bytes:
    """
    Validate an uploaded image:
      1) Must not be empty (400)
      2) Must not exceed MAX_FILE_SIZE_MB (413)
      3) Must have an allowed extension (400)
      4) Must have an image content type (400)
      5) Must be a loadable, non‐corrupted image (422)
    Returns the raw bytes for downstream processing.
    """
    # Read all bytes
    data = await file.read()
    size = len(data)

    # 1) Empty?
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty file upload")

    # 2) Too large?
    max_bytes = config.MAX_FILE_SIZE * 1024 * 1024
    if size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {config.MAX_FILE_SIZE} MB"
        )

    # 3) Extension check
    ext = os.path.splitext((file.filename or "").lower())[1]
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(config.ALLOWED_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed}"
        )

    # 4) Content‐type check
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Must be an image file."
        )

    # 5) Corruption check
    try:
        Image.open(io.BytesIO(data)).verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Invalid or corrupted image file")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not process image file")

    # Rewind so later code can `await file.read()` again if needed
    file.file.seek(0)
    return data

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

# def process_background_removal(image_data: bytes, enhance: bool = True) -> bytes:
#     """Process background removal with optimizations"""
#     try:
#         # Optimize image for processing
#         image = optimize_image_for_processing(image_data)
        
#         # Convert PIL image back to bytes for rembg
#         img_byte_arr = io.BytesIO()
#         image.save(img_byte_arr, format='PNG')
#         optimized_image_data = img_byte_arr.getvalue()
        
#         # Remove background using the global session
#         result_data = remove(optimized_image_data, session=bg_removal_session)
        
#         # Post-process if enhancement is requested
#         if enhance:
#             # Ensure result_data is bytes before using BytesIO
#             if isinstance(result_data, bytes):
#                 result_image = Image.open(io.BytesIO(result_data))
#             elif isinstance(result_data, Image.Image):
#                 result_image = result_data
#             else:
#                 raise ImageProcessingError("Unexpected result_data type for enhancement")
#             result_image = enhance_image_quality(result_image)
            
#             # Convert back to bytes
#             enhanced_byte_arr = io.BytesIO()
#             result_image.save(enhanced_byte_arr, format='PNG', optimize=True)
#             result_data = enhanced_byte_arr.getvalue()
        
#         # Force garbage collection to free memory
#         gc.collect()

#         # Ensure result_data is bytes before returning
#         if isinstance(result_data, bytes):
#             return result_data
#         elif isinstance(result_data, Image.Image):
#             buf = io.BytesIO()
#             result_data.save(buf, format='PNG')
#             return buf.getvalue()
#         elif isinstance(result_data, np.ndarray):
#             # Convert ndarray to PIL Image, then to bytes
#             img = Image.fromarray(result_data)
#             buf = io.BytesIO()
#             img.save(buf, format='PNG')
#             return buf.getvalue()
#         else:
#             raise ImageProcessingError("Unexpected result_data type in background removal")

#     except Exception as e:
#         logger.error(f"Background removal failed: {str(e)}")
#         raise ImageProcessingError(f"Background removal failed: {str(e)}")


def ensure_models_directory():
    """
    Create the models directory if it doesn't exist.
    This is like creating a bookshelf before you can put books on it.
    """
    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)
        logger.info(f"Created models directory at: {config.MODELS_DIR}")


def download_model(model_name: str, force_download: bool = False) -> str:
    """
    Download the U2Net model file if it doesn't exist.
    
    This function handles the entire download process:
    1. Checks if we have a URL for the requested model
    2. Creates the models directory if needed
    3. Downloads the model if it's missing or if forced
    4. Verifies the download was successful
    
    Args:
        model_name: Name of the model to download (e.g., 'u2net')
        force_download: Whether to re-download even if file exists
    
    Returns:
        Path to the downloaded model file
    """
    try:
        # First, make sure we have a URL for this model
        if model_name not in config.MODEL_URLS:
            available_models = ", ".join(config.MODEL_URLS.keys())
            raise ImageProcessingError(
                f"Unknown model '{model_name}'. Available models: {available_models}"
            )
        
        # Ensure the models directory exists
        ensure_models_directory()
        
        model_path = os.path.join(config.MODELS_DIR, f"{model_name}.onnx")
        
        # Check if we need to download
        if os.path.exists(model_path) and not force_download:
            logger.info(f"Model {model_name} already exists at {model_path}")
            return model_path
        
        # Download the model
        model_url = config.MODEL_URLS[model_name]
        logger.info(f"Downloading {model_name} model from {model_url}")
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the model file
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify the download
        if not os.path.exists(model_path):
            raise ImageProcessingError(f"Model download failed - file not found at {model_path}")
        
        file_size = os.path.getsize(model_path)
        if file_size < 1000:  # Model should be much larger than 1KB
            raise ImageProcessingError(f"Model download appears incomplete - file size is only {file_size} bytes")
        
        logger.info(f"Successfully downloaded {model_name} model ({file_size} bytes)")
        return model_path
        
    except requests.RequestException as e:
        error_msg = f"Failed to download model {model_name}: {str(e)}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e
    except Exception as e:
        error_msg = f"Error downloading model {model_name}: {str(e)}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e


def create_fresh_onnx_session() -> ort.InferenceSession:
    """
    Create a brand new ONNX Runtime session, completely separate from any rembg code.
    
    This function now properly handles model downloading and verification:
    1. Downloads the model if it doesn't exist
    2. Verifies the model file is valid
    3. Creates the ONNX Runtime session
    4. Validates the session is working correctly
    """
    try:
        logger.info(f"Creating new ONNX Runtime session for {config.MODEL_NAME} model")
        
        # Step 1: Ensure we have the model file
        model_path = download_model(config.MODEL_NAME)
        
        # Step 2: Verify the model file exists and is readable
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        if not os.access(model_path, os.R_OK):
            raise PermissionError(f"Cannot read model file at {model_path}")
        
        # Step 3: Configure ONNX Runtime session with performance optimizations
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set up execution providers (GPU if available, otherwise CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Step 4: Create the session - this is a pure ONNX Runtime session
        fresh_session = ort.InferenceSession(
            model_path,  # Use the full path to the downloaded model
            sess_options=session_options,
            providers=providers
        )
        
        # Step 5: Verify this is actually an ONNX Runtime session by checking its type
        logger.info(f"Session type: {type(fresh_session)}")
        logger.info(f"Session providers: {fresh_session.get_providers()}")
        
        # Step 6: Double-check that our session has the methods we need
        if not hasattr(fresh_session, 'get_inputs'):
            raise RuntimeError("Session doesn't have get_inputs method - this shouldn't happen with ONNX Runtime")
        
        if not hasattr(fresh_session, 'run'):
            raise RuntimeError("Session doesn't have run method - this shouldn't happen with ONNX Runtime")
        
        # Step 7: Get input information to verify the model loaded correctly
        input_info = fresh_session.get_inputs()[0]
        logger.info(f"Model input name: {input_info.name}")
        logger.info(f"Model input shape: {input_info.shape}")
        
        logger.info("ONNX Runtime session created successfully!")
        return fresh_session
        
    except FileNotFoundError as e:
        error_msg = f"Model file not found: {str(e)}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to create ONNX Runtime session: {str(e)}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e


def get_isolated_onnx_session() -> ort.InferenceSession:
    """
    Get or create our isolated ONNX Runtime session.
    
    This function ensures we always work with a pure ONNX Runtime session,
    never any session from rembg or other libraries.
    """
    global _onnx_u2net_session
    
    # If we don't have a session yet, or if somehow we got the wrong type, create a new one
    if _onnx_u2net_session is None:
        logger.info("Creating new ONNX session for the first time")
        _onnx_u2net_session = create_fresh_onnx_session()
    else:
        # Double-check that we still have the right type of session
        # This prevents issues if some other code overwrote our session
        if not isinstance(_onnx_u2net_session, ort.InferenceSession):
            logger.warning(f"Session was overwritten with wrong type: {type(_onnx_u2net_session)}")
            logger.info("Recreating ONNX session")
            _onnx_u2net_session = create_fresh_onnx_session()
    
    return _onnx_u2net_session


def preprocess_image_for_u2net(image: Image.Image) -> tuple[np.ndarray, tuple]:
    """
    Preprocess image for U2Net model input.
    
    This function prepares the image exactly as the U2Net model expects:
    - Converts to RGB format (removes alpha channel if present)
    - Resizes to 320x320 (standard U2Net input size)
    - Normalizes pixel values to [0, 1] range
    - Rearranges from HWC to CHW format (channels first)
    - Adds batch dimension
    
    Returns:
        Tuple containing:
        - Preprocessed image array ready for model input
        - Original image size for restoring output dimensions
    """
    # Convert to RGB if the image is in a different format
    # This is crucial - the model expects exactly 3 channels
    if image.mode != 'RGB':
        logger.debug(f"Converting image from {image.mode} to RGB")
        image = image.convert('RGB')
    
    # Store original dimensions for later restoration
    original_size = image.size  # PIL uses (width, height) format
    logger.debug(f"Original image size: {original_size}")
    
    # Resize to the size expected by U2Net (320x320)
    # Using LANCZOS for high-quality resizing
    target_size = (320, 320)
    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert PIL image to numpy array
    # This gives us shape (height, width, channels) with values in [0, 255]
    image_array = np.array(image_resized, dtype=np.float32)
    
    # Normalize pixel values to [0, 1] range as expected by the model
    # Neural networks typically work better with normalized inputs
    image_array = image_array / 255.0
    
    # Transpose from HWC to CHW format (channels first)
    # Deep learning models expect channels first: (channels, height, width)
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    # Models expect batch processing even for single images
    image_array = np.expand_dims(image_array, axis=0)
    
    logger.debug(f"Preprocessed image shape: {image_array.shape}")
    logger.debug(f"Preprocessed image range: [{image_array.min():.3f}, {image_array.max():.3f}]")
    
    return image_array, original_size


def postprocess_u2net_output(mask: np.ndarray, original_size: tuple) -> np.ndarray:
    """
    Convert U2Net model output back to a usable image mask.
    
    FIXED VERSION: This addresses the core issue where masks weren't being processed
    correctly, resulting in no visible background removal.
    
    Key fixes:
    1. Better handling of multi-dimensional model outputs
    2. Proper normalization of mask values
    3. More aggressive thresholding for cleaner separation
    4. Improved edge refinement
    
    Args:
        mask: Raw model output (probability map)
        original_size: Target size for the final mask (width, height)
    
    Returns:
        Processed mask ready to use for background removal
    """
    logger.debug(f"Processing mask with shape: {mask.shape}")
    logger.debug(f"Mask value range: [{mask.min():.3f}, {mask.max():.3f}]")
    
    # CRITICAL FIX: Handle U2Net's multi-output structure correctly
    # U2Net often outputs multiple prediction maps, we need the main one
    if isinstance(mask, (list, tuple)):
        mask = mask[0]  # Take the first output (main prediction)
    
    # Remove batch dimension if present
    if mask.ndim == 4:  # (1, 1, H, W) or (1, C, H, W)
        mask = mask[0]
    if mask.ndim == 3:  # (1, H, W) or (C, H, W)
        if mask.shape[0] == 1:
            mask = mask[0]  # Remove channel dimension if it's 1
        else:
            mask = mask[0]  # Take first channel if multiple channels
    
    # Ensure mask is 2D at this point
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after preprocessing, got shape: {mask.shape}")
    
    # CRITICAL FIX: Proper normalization based on actual data range
    # Sometimes model outputs are in different ranges than expected
    mask_min, mask_max = mask.min(), mask.max()
    logger.debug(f"Mask range before normalization: [{mask_min:.3f}, {mask_max:.3f}]")
    
    if mask_max > mask_min:
        # Normalize to [0, 1] range
        mask = (mask - mask_min) / (mask_max - mask_min)
    else:
        # Handle edge case where mask is uniform
        mask = np.ones_like(mask)
    
    # CRITICAL FIX: Apply sigmoid if values seem to be logits (common U2Net output)
    # This is often the missing step that causes poor mask quality
    if mask_min < 0 or mask_max > 1:
        mask = 1.0 / (1.0 + np.exp(-mask))
    
    # CRITICAL FIX: More aggressive thresholding for cleaner separation
    # The original 0.5 threshold might be too conservative
    # Using Otsu's method for automatic threshold selection
    mask_uint8_temp = (mask * 255).astype(np.uint8)
    threshold_value, _ = cv2.threshold(mask_uint8_temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_normalized = threshold_value / 255.0
    
    logger.debug(f"Automatic threshold selected: {threshold_normalized:.3f}")
    
    # Apply the threshold
    mask_binary = np.where(mask > threshold_normalized, 1.0, 0.0)
    
    # CRITICAL FIX: Morphological operations to clean up the mask
    # This removes small holes and smooths edges
    kernel = np.ones((3, 3), np.uint8)
    mask_binary_uint8 = (mask_binary * 255).astype(np.uint8)
    
    # Fill small holes in the foreground
    mask_binary_uint8 = cv2.morphologyEx(mask_binary_uint8, cv2.MORPH_CLOSE, kernel)
    # Remove small noise in the background
    mask_binary_uint8 = cv2.morphologyEx(mask_binary_uint8, cv2.MORPH_OPEN, kernel)
    
    # CRITICAL FIX: Apply edge smoothing only to the mask edges
    # This prevents the entire mask from becoming blurry
    mask_blurred = cv2.GaussianBlur(mask_binary_uint8, (5, 5), 0)
    
    # Create edge mask to apply smoothing only at boundaries
    edges = cv2.Canny(mask_binary_uint8, 50, 150)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Combine: use blurred version only at edges, binary elsewhere
    mask_final = np.where(edges_dilated > 0, mask_blurred, mask_binary_uint8)
    
    # Resize back to original image dimensions
    mask_resized = cv2.resize(mask_final, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    logger.debug(f"Final mask shape: {mask_resized.shape}")
    logger.debug(f"Final mask range: [{mask_resized.min()}, {mask_resized.max()}]")
    logger.debug(f"Final mask unique values: {len(np.unique(mask_resized))}")
    
    return mask_resized


def apply_mask_to_image_with_black_background(original_image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Apply the processed mask to create a black background instead of transparency.
    
    FIXED VERSION: This addresses your specific request for black background
    instead of transparent background.
    
    Args:
        original_image: The original input image
        mask: The processed mask from U2Net
    
    Returns:
        Image with black background where the original background was removed
    """
    # Convert original image to RGB (we don't need alpha channel for black background)
    if original_image.mode != 'RGB':
        logger.debug(f"Converting image from {original_image.mode} to RGB")
        original_rgb = original_image.convert('RGB')
    else:
        original_rgb = original_image.copy()
    
    # Convert to numpy array for manipulation
    original_array = np.array(original_rgb)
    
    # Ensure the mask has the same dimensions as the image
    if mask.shape != original_array.shape[:2]:
        logger.warning(f"Mask shape {mask.shape} doesn't match image shape {original_array.shape[:2]}")
        mask = cv2.resize(mask, (original_array.shape[1], original_array.shape[0]))
    
    # Create a black background
    black_background = np.zeros_like(original_array)
    
    # CRITICAL FIX: Apply mask properly for black background
    # Normalize mask to [0, 1] range for proper blending
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # Expand mask to 3 channels (RGB)
    mask_3d = np.stack([mask_normalized] * 3, axis=2)
    
    # Blend: foreground where mask is 1, black background where mask is 0
    result_array = (original_array.astype(np.float32) * mask_3d + 
                   black_background.astype(np.float32) * (1 - mask_3d))
    
    # Convert back to uint8
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(result_array)
    
    return result_image


def remove_background_with_isolated_u2net(image_data: bytes) -> bytes:
    """
    Remove background using our isolated U2Net ONNX implementation.
    
    FIXED VERSION: This incorporates all the critical fixes and uses black background
    as requested.
    
    Args:
        image_data: Input image as bytes
    
    Returns:
        Processed image with black background as bytes
    """
    try:
        logger.info("Starting background removal with isolated U2Net")
        
        # Get our guaranteed ONNX Runtime session
        session = get_isolated_onnx_session()
        
        # Verify we have the correct session type
        if not isinstance(session, ort.InferenceSession):
            raise RuntimeError(f"Wrong session type: {type(session)}. Expected ort.InferenceSession")
        
        # Load the input image
        image = Image.open(io.BytesIO(image_data))
        logger.debug(f"Loaded image: {image.size}, {image.mode}")
        
        # Preprocess the image for the model
        processed_input, original_size = preprocess_image_for_u2net(image)
        
        # Get the model's input name
        input_name = session.get_inputs()[0].name
        logger.debug(f"Model input name: {input_name}")
        
        # Run inference
        logger.debug("Running model inference")
        outputs = session.run(None, {input_name: processed_input})
        
        # CRITICAL FIX: Handle U2Net's multiple outputs correctly
        # U2Net typically returns multiple prediction maps, we want the main one
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            mask = outputs[0]  # Main prediction is usually the first output
        else:
            mask = outputs
        
        logger.debug(f"Model output shape: {mask.shape}")
        logger.debug(f"Model output range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        # CRITICAL FIX: Use the improved mask processing
        processed_mask = postprocess_u2net_output(mask, original_size)
        
        # CRITICAL FIX: Apply the mask to create black background instead of transparency
        result_image = apply_mask_to_image_with_black_background(image, processed_mask)
        
        # Save to bytes for return
        # Use JPEG since we don't need transparency with black background
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='JPEG', optimize=True, quality=95)
        
        logger.info("Background removal completed successfully")
        return output_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}")
        raise ImageProcessingError(f"Background removal failed: {str(e)}") from e


def debug_mask_values(image_data: bytes) -> dict:
    """
    Enhanced debug function to help understand mask processing issues.
    
    This function provides detailed analysis of why masks might not be working.
    """
    try:
        # Get session and load image
        session = get_isolated_onnx_session()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess
        processed_input, original_size = preprocess_image_for_u2net(image)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: processed_input})
        
        # Get raw mask
        raw_mask = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        
        # Process mask step by step for debugging
        mask_analysis = {}
        
        # Raw mask analysis
        mask_analysis["raw_mask_shape"] = raw_mask.shape
        mask_analysis["raw_mask_min"] = float(raw_mask.min())
        mask_analysis["raw_mask_max"] = float(raw_mask.max())
        mask_analysis["raw_mask_mean"] = float(raw_mask.mean())
        mask_analysis["raw_mask_std"] = float(raw_mask.std())
        
        # After dimension reduction
        mask_2d = raw_mask
        while mask_2d.ndim > 2:
            mask_2d = mask_2d[0]
        
        mask_analysis["mask_2d_shape"] = mask_2d.shape
        mask_analysis["mask_2d_min"] = float(mask_2d.min())
        mask_analysis["mask_2d_max"] = float(mask_2d.max())
        
        # After normalization
        if mask_2d.max() > mask_2d.min():
            mask_norm = (mask_2d - mask_2d.min()) / (mask_2d.max() - mask_2d.min())
        else:
            mask_norm = np.ones_like(mask_2d)
        
        mask_analysis["mask_normalized_min"] = float(mask_norm.min())
        mask_analysis["mask_normalized_max"] = float(mask_norm.max())
        mask_analysis["mask_normalized_mean"] = float(mask_norm.mean())
        
        # Check if mask has actual variation
        mask_analysis["mask_has_variation"] = len(np.unique(mask_norm)) > 10
        mask_analysis["mask_unique_values_count"] = len(np.unique(mask_norm))
        
        # Threshold analysis
        mask_uint8_temp = (mask_norm * 255).astype(np.uint8)
        threshold_value, _ = cv2.threshold(mask_uint8_temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_analysis["otsu_threshold"] = int(threshold_value)
        mask_analysis["otsu_threshold_normalized"] = float(threshold_value / 255.0)
        
        # Final processed mask
        processed_mask = postprocess_u2net_output(raw_mask, original_size)
        mask_analysis["final_mask_shape"] = processed_mask.shape
        mask_analysis["final_mask_min"] = int(processed_mask.min())
        mask_analysis["final_mask_max"] = int(processed_mask.max())
        mask_analysis["final_mask_unique_values"] = len(np.unique(processed_mask))
        
        # Check if final mask will actually remove background
        mask_analysis["will_remove_background"] = (processed_mask.max() > processed_mask.min() and 
                                                  len(np.unique(processed_mask)) > 5)
        
        return mask_analysis
        
    except Exception as e:
        return {"error": str(e)}


def process_background_removal(image_data: bytes, enhance: bool = True) -> bytes:
    """
    FIXED VERSION: Main function for background removal with black background.
    
    This version incorporates all the critical fixes and provides black background
    as requested.
    """
    try:
        logger.info("Starting background removal process")
        
        # Debug the mask processing
        debug_info = debug_mask_values(image_data)
        logger.debug(f"Mask analysis: {debug_info}")
        
        # Check if mask will actually work
        if not debug_info.get("will_remove_background", False):
            logger.warning("Mask analysis suggests background removal may not work properly")
            logger.warning("This could be due to:")
            logger.warning("1. Model not detecting clear foreground/background separation")
            logger.warning("2. Input image may not be suitable for background removal")
            logger.warning("3. Model may need different preprocessing")
        
        # Try to optimize the image if the function exists
        optimized_image_data = image_data
        try:
            image = optimize_image_for_processing(image_data)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            optimized_image_data = img_byte_arr.getvalue()
            logger.debug("Applied image optimization")
        except NameError:
            logger.debug("optimize_image_for_processing not available, using original image")
        
        # Remove background using our improved implementation
        result_data = remove_background_with_isolated_u2net(optimized_image_data)
        
        # Apply enhancement if requested and available
        if enhance:
            try:
                if isinstance(result_data, bytes):
                    result_image = Image.open(io.BytesIO(result_data))
                elif isinstance(result_data, Image.Image):
                    result_image = result_data
                else:
                    raise ImageProcessingError("Unexpected result_data type for enhancement")
                
                # Try to enhance if the function exists
                try:
                    result_image = enhance_image_quality(result_image)
                    logger.debug("Applied image enhancement")
                except NameError:
                    logger.debug("enhance_image_quality not available, skipping enhancement")
                
                # Convert back to bytes
                enhanced_byte_arr = io.BytesIO()
                result_image.save(enhanced_byte_arr, format='JPEG', optimize=True, quality=95)
                result_data = enhanced_byte_arr.getvalue()
                
            except Exception as e:
                logger.warning(f"Enhancement failed: {str(e)}, continuing without enhancement")
        
        # Clean up memory
        gc.collect()
        
        # Ensure we return bytes
        if isinstance(result_data, bytes):
            return result_data
        elif isinstance(result_data, Image.Image):
            buf = io.BytesIO()
            result_data.save(buf, format='JPEG', quality=95)
            return buf.getvalue()
        else:
            raise ImageProcessingError("Unexpected result_data type in background removal")
    
    except Exception as e:
        logger.error(f"Background removal process failed: {str(e)}")
        raise ImageProcessingError(f"Background removal failed: {str(e)}") from e


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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    start_time = datetime.now(timezone.utc)
    
    try:
        # Validate file
        await validate_image_file(file)
        
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
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
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
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
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
#             "created_at": datetime.now(timezone.utc).isoformat(),
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
#                 "completed_at": datetime.now(timezone.utc).isoformat()
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
#                 "failed_at": datetime.now(timezone.utc).isoformat()
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
    image_bytes = await validate_image_file(file)

    # 2) Create task_id and initial status
    task_id = str(uuid.uuid4())
    initial_status = {
        "status": "processing",
        "created_at": datetime.now(timezone.utc).isoformat(),
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
    # image_bytes = await file.read()
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
            "completed_at": datetime.now(timezone.utc).isoformat()
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
            "failed_at": datetime.now(timezone.utc).isoformat()
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