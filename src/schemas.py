"""
LeafCheck - Pydantic Request & Response Schemas
Defines structured API contracts for FastAPI validation and Swagger OpenAPI docs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionCandidate(BaseModel):
    class_name: str = Field(..., description="Raw label identifier")
    display_name: str = Field(..., description="Human-readable common name")
    confidence: float = Field(..., description="Softmax confidence score (0.0 to 1.0)")
    percentage: str = Field(..., description="Formatted percentage string")


class DiseaseDetails(BaseModel):
    disease_name: str
    causal_agent: str
    symptoms: str
    prevention: List[str]
    treatment: List[str]


class PredictionResponse(BaseModel):
    success: bool = True
    filename: str
    predicted_class: str
    display_name: str
    confidence: float
    confidence_percentage: str
    is_confident: bool = Field(True, description="False if confidence is below 60% threshold")
    warning: Optional[str] = Field(None, description="Guidance warning for ambiguous or out-of-distribution inputs")
    top_3_predictions: List[PredictionCandidate]
    disease_info: DiseaseDetails
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    classes_count: int
    architecture: str


class ClassesResponse(BaseModel):
    total_classes: int
    classes: List[str]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str
