from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    detail: str
    loaded_models: List[str] = Field(default_factory=list)
    loaded_model_count: int = 0
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]


class PredictionResponse(BaseModel):
    id: int
    filename: str
    duration_seconds: float
    cnn: PredictionResult
    cnn_bilstm: PredictionResult
    ensemble: PredictionResult
    selected_model: str = "ensemble"
    selected_prediction: PredictionResult
    created_at: str


class HistoryItem(BaseModel):
    id: int
    filename: str
    duration_seconds: float
    emotion: str
    model_used: str = "ensemble"
    confidence: Optional[float] = None
    created_at: str


class DatasetSample(BaseModel):
    filename: str
    emotion: str
    actor: str


class DatasetSamplesResponse(BaseModel):
    configured: bool
    directory_exists: bool = False
    dataset_available: bool = False
    total_wav_files: Optional[int] = None
    valid_audio_count: int = 0
    dataset_path: Optional[str] = None
    error: Optional[str] = None
    valid_audio_files: int
    actor_count: int
    emotion_counts: Dict[str, int]
    samples: List[DatasetSample]
    detail: Optional[str] = None