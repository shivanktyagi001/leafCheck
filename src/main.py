"""
LeafCheck - FastAPI Application Backend
Exposes endpoints for model health, classes catalog, and leaf disease inference.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.schemas import (
    PredictionResponse,
    HealthResponse,
    ClassesResponse,
    ErrorResponse
)
from src.model_loader import get_model, load_classes
from src.predict import predict_leaf_disease, ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leafcheck.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for model pre-heating."""
    logger.info("Initializing LeafCheck service...")
    try:
        # Pre-warm model and classes into memory on startup
        get_model()
        classes = load_classes()
        logger.info(f"LeafCheck ready with {len(classes)} target classes.")
    except Exception as e:
        logger.warning(f"Startup model initialization note: {e}")
    yield
    logger.info("LeafCheck service shutting down.")


app = FastAPI(
    title="LeafCheck - Tomato Leaf Disease Detection API",
    description=(
        "Production-grade REST API utilizing Deep Learning (MobileNetV2 Transfer Learning) "
        "and MLOps to accurately classify 10 tomato leaf diseases and recommend actionable "
        "agricultural treatments."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware to allow cross-origin requests from web or mobile dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["General"],
    summary="Root Welcome Endpoint",
    response_description="Basic API identification and status"
)
async def root():
    return {
        "project": "LeafCheck – Crop Disease Detection System",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "classes_endpoint": "/classes",
        "predict_endpoint": "/predict"
    }


@app.get(
    "/health",
    tags=["Diagnostics"],
    summary="Service Health Check",
    response_model=HealthResponse,
    responses={200: {"description": "Service is healthy and operational"}}
)
async def health_check():
    try:
        classes = load_classes()
        model = get_model()
        model_loaded = model is not None
        arch_name = getattr(model, "name", "MobileNetV2")
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            model_version="1.0.0",
            classes_count=len(classes),
            architecture=arch_name
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_version="1.0.0",
            classes_count=0,
            architecture="unknown"
        )


@app.get(
    "/classes",
    tags=["Diagnostics"],
    summary="Get Supported Tomato Disease Classes",
    response_model=ClassesResponse
)
async def list_classes():
    classes = load_classes()
    return ClassesResponse(total_classes=len(classes), classes=classes)


@app.post(
    "/predict",
    tags=["Inference"],
    summary="Identify Leaf Disease from Uploaded Image",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse, "description": "Successful classification"},
        400: {"model": ErrorResponse, "description": "Invalid image or unsupported format"},
        413: {"model": ErrorResponse, "description": "File payload exceeds 10 MB limit"},
        500: {"model": ErrorResponse, "description": "Internal server or inference failure"}
    }
)
async def predict_image(
    file: UploadFile = File(..., description="JPEG, PNG, or WEBP image of a tomato leaf")
):
    # 1. Content-Type check (soft filter; strict byte inspection occurs inside predict service)
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                f"Supported types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            )
        )

    # 2. Read bytes
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # 3. Size check
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # 4. Perform prediction
    try:
        prediction_result = predict_leaf_disease(
            file_bytes=contents,
            filename=file.filename or "uploaded_leaf.jpg"
        )
        return prediction_result
    except ValueError as ve:
        logger.warning(f"Validation rejection for file '{file.filename}': {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as ex:
        logger.error(f"Inference pipeline execution error: {ex}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(ex)}"
        )
