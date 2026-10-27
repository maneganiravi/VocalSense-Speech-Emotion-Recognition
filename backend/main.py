import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.database import (
    get_history,
    get_prediction_count,
    get_prediction_details,
    initialize_database,
    save_prediction,
)
from backend.predictor import model_status, predict_audio
from backend.schemas import (
    DatasetSamplesResponse,
    HealthResponse,
    HistoryItem,
    PredictionResponse,
)
from training.preprocessing import EMOTIONS, scan_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATASET_CONFIG_PATH = PROJECT_ROOT / "dataset_config.json"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(
    title="VocalSense",
    description="Speech emotion recognition with CNN and CNN-BiLSTM models.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    initialize_database()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = model_status()
    return HealthResponse(
        status="ok",
        models_ready=status["ready"],
        detail=status["detail"],
        loaded_models=status.get("loaded_models", []),
        loaded_model_count=status.get("loaded_model_count", 0),
        models=status.get("models", {}),
    )


def _dataset_path() -> Path:
    try:
        with DATASET_CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
        configured_path = Path(config["dataset_path"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=500, detail=f"Dataset configuration is unavailable: {error}"
        ) from error
    return configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path


def _dataset_report() -> DatasetSamplesResponse:
    try:
        root = _dataset_path()
    except HTTPException as error:
        return DatasetSamplesResponse(
            configured=False,
            directory_exists=False,
            dataset_available=False,
            valid_audio_count=0,
            dataset_path=None,
            error=str(error.detail),
            valid_audio_files=0,
            actor_count=0,
            emotion_counts={},
            samples=[],
            detail=str(error.detail),
        )
    dataset_path = str(root.resolve())
    if not root.is_dir():
        message = f"Configured dataset directory does not exist: {dataset_path}"
        return DatasetSamplesResponse(
            configured=True,
            directory_exists=False,
            dataset_available=False,
            total_wav_files=None,
            valid_audio_count=0,
            dataset_path=dataset_path,
            error=message,
            valid_audio_files=0,
            actor_count=0,
            emotion_counts={},
            samples=[],
            detail=message,
        )

    try:
        total_wav_files = sum(
            1
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        records = scan_dataset(root, require_all_emotions=False)
    except (OSError, ValueError) as error:
        message = f"Could not scan configured dataset at {dataset_path}: {error}"
        return DatasetSamplesResponse(
            configured=True,
            directory_exists=True,
            dataset_available=False,
            total_wav_files=None,
            valid_audio_count=0,
            dataset_path=dataset_path,
            error=message,
            valid_audio_files=0,
            actor_count=0,
            emotion_counts={},
            samples=[],
            detail=message,
        )

    counts = {emotion: 0 for emotion in EMOTIONS.values()}
    examples: Dict[str, Dict[str, str]] = {}
    actors = set()
    for path, emotion, actor in records:
        counts[emotion] += 1
        actors.add(actor)
        examples.setdefault(
            emotion,
            {"filename": path.name, "emotion": emotion, "actor": f"{actor:02d}"},
        )
    missing_emotions = sorted(
        emotion for emotion, count in counts.items() if count == 0
    )
    diagnostic = (
        "Dataset is missing emotion classes: " + ", ".join(missing_emotions)
        if missing_emotions
        else None
    )
    return DatasetSamplesResponse(
        configured=True,
        directory_exists=True,
        dataset_available=bool(records) and not missing_emotions,
        total_wav_files=total_wav_files,
        valid_audio_count=len(records),
        dataset_path=dataset_path,
        error=diagnostic,
        valid_audio_files=len(records),
        actor_count=len(actors),
        emotion_counts=counts,
        samples=list(examples.values()),
        detail=diagnostic or "Unique speech clips filtered from RAVDESS by filename.",
    )


@app.get("/api/samples", response_model=DatasetSamplesResponse)
def samples() -> DatasetSamplesResponse:
    return _dataset_report()


@app.get("/api/dataset-status", response_model=DatasetSamplesResponse)
def dataset_status() -> DatasetSamplesResponse:
    return _dataset_report()


@app.get("/api/dashboard")
def dashboard() -> Dict[str, Any]:
    model_info = model_status()
    dataset = _dataset_report()
    history_rows = get_history(limit=100)
    prediction_count = get_prediction_count()
    last_prediction = history_rows[0]["created_at"] if history_rows else None
    model_classes = set(model_info.get("class_order", []))
    dataset_classes = set(EMOTIONS.values())
    missing_model_classes = sorted(dataset_classes - model_classes)
    return {
        "backend_status": "ok" if model_info.get("ready") else "degraded",
        "models_ready": bool(model_info.get("ready")),
        "detail": model_info.get("detail", ""),
        "loaded_models": model_info.get("loaded_models", []),
        "loaded_model_count": model_info.get("loaded_model_count", 0),
        "models": model_info.get("models", {}),
        "model_count": model_info.get("loaded_model_count", 0),
        "configured": dataset.configured,
        "directory_exists": dataset.directory_exists,
        "dataset_available": dataset.dataset_available,
        "total_wav_files": dataset.total_wav_files,
        "valid_audio_count": dataset.valid_audio_count,
        "actor_count": dataset.actor_count,
        "dataset_path": dataset.dataset_path,
        "dataset_error": dataset.error,
        "dataset_model_compatible": bool(model_info.get("ready")) and not missing_model_classes,
        "model_missing_dataset_emotions": missing_model_classes,
        "model_dataset_detail": (
            "Trained models do not cover these dataset emotions: "
            + ", ".join(missing_model_classes)
            if missing_model_classes
            else None
        ),
        "dataset_configured": bool(dataset.configured),
        "dataset_files": dataset.valid_audio_count,
        "total_predictions": prediction_count,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "last_prediction": last_prediction,
    }


@app.get("/api/sample-audio")
def sample_audio(filename: str) -> FileResponse:
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="A sample filename is required.")
    root = _dataset_path()
    try:
        target = next(
            path
            for path, _emotion, _actor in scan_dataset(
                root, require_all_emotions=False
            )
            if path.name == filename
        ).resolve()
    except StopIteration as error:
        raise HTTPException(status_code=404, detail="Sample audio file not found.") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Dataset samples could not be scanned: {error}") from error
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise HTTPException(status_code=403, detail="Sample access is restricted to the configured dataset folder.") from error
    if not target.is_file() or target.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="Sample audio file not found.")
    return FileResponse(target, media_type="audio/wav", filename=target.name)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    selected_model: str = Form("ensemble"),
) -> Dict[str, Any]:
    if selected_model not in {"cnn", "cnn_bilstm", "ensemble"}:
        raise HTTPException(
            status_code=422,
            detail="Select one of: cnn, cnn_bilstm, or ensemble.",
        )
    upload = audio or file
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide a WAV upload in the 'audio' field.")
    filename = Path(upload.filename or "").name
    if Path(filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=415, detail="Upload a WAV (.wav) audio file.")
    if upload.content_type not in (
        None,
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=415, detail="The uploaded file must be a WAV audio file.")
    audio_bytes = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The audio file must be 20 MB or smaller.")
    if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise HTTPException(status_code=400, detail="The file is not a valid WAV audio file.")
    try:
        predictions, duration = predict_audio(audio_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    saved = save_prediction(
        filename=filename,
        duration_seconds=duration,
        emotion=predictions[selected_model]["emotion"],
        cnn_emotion=predictions["cnn"]["emotion"],
        cnn_bilstm_emotion=predictions["cnn_bilstm"]["emotion"],
        probabilities=predictions,
        confidence=predictions[selected_model]["confidence"],
        model_used=selected_model,
    )
    return {
        **saved,
        "cnn": predictions["cnn"],
        "cnn_bilstm": predictions["cnn_bilstm"],
        "ensemble": predictions["ensemble"],
        "selected_model": selected_model,
        "selected_prediction": predictions[selected_model],
    }


@app.get("/api/history", response_model=List[HistoryItem])
def history(limit: int = 50) -> List[Dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="Limit must be between 1 and 100.")
    return get_history(limit)


@app.get("/api/history/{prediction_id}")
def prediction_details(prediction_id: int) -> Dict[str, Any]:
    if prediction_id < 1:
        raise HTTPException(status_code=422, detail="Prediction id must be positive.")
    try:
        details = get_prediction_details(prediction_id)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    if details is None:
        raise HTTPException(status_code=404, detail="Prediction history item not found.")
    return details


class ModelInfoResponse(BaseModel):
    ready: bool
    detail: str
    loaded_models: List[str] = Field(default_factory=list)
    loaded_model_count: int = 0
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    class_order: List[str] = Field(default_factory=list)
    metrics: Any = None


@app.get("/api/model-info", response_model=ModelInfoResponse)
def model_info() -> Dict[str, Any]:
    return model_status()


@app.get("/api/metrics")
def metrics() -> Dict[str, Any]:
    status = model_status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status["detail"])
    return status["metrics"]