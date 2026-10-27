import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model

from training.features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "training" / "artifacts"
MODEL_FILES = {
    "cnn": "cnn_model.keras",
    "cnn_bilstm": "cnn_bilstm_model.keras",
    "temporal": "temporal_model.keras",
    "regularized": "regularized_model.keras",
}
MODEL_INPUTS = {
    "cnn": (108, 1),
    "cnn_bilstm": (108, 1),
    "temporal": (64, 54),
    "regularized": (64, 54),
}
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 30.0
_loaded: Optional[Tuple[Dict[str, Any], Any, Any, Any, Dict[str, Any]]] = None


def _load_artifacts() -> Tuple[Dict[str, Any], Any, Any, Any, Dict[str, Any]]:
    global _loaded
    if _loaded is not None:
        return _loaded

    required = [
        *(MODEL_DIR / filename for filename in MODEL_FILES.values()),
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "temporal_scaler.pkl",
        MODEL_DIR / "label_encoder.pkl",
        MODEL_DIR / "metrics.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Training artifacts are missing: " + ", ".join(missing))

    try:
        with (MODEL_DIR / "metrics.json").open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        temporal_scaler = joblib.load(MODEL_DIR / "temporal_scaler.pkl")
        encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
        models = {
            name: load_model(MODEL_DIR / filename, compile=False)
            for name, filename in MODEL_FILES.items()
        }
    except (OSError, ValueError, TypeError, EOFError, ImportError, KeyError) as error:
        raise RuntimeError(f"Training artifacts could not be loaded: {error}") from error

    class_order = list(encoder.classes_)
    if class_order != metrics.get("class_order"):
        raise RuntimeError("The label encoder and metrics use different class orders.")
    if getattr(scaler, "n_features_in_", None) != 108:
        raise RuntimeError("The saved summary scaler does not match 108 input features.")
    if getattr(temporal_scaler, "n_features_in_", None) != 54:
        raise RuntimeError("The saved temporal scaler does not match 54 input features.")
    systems = metrics.get("models")
    required_systems = {"cnn", "cnn_bilstm", "temporal", "regularized", "ensemble"}
    if not isinstance(systems, dict) or not required_systems.issubset(systems):
        raise RuntimeError("Metrics do not include every trained model and the ensemble.")

    for name, model in models.items():
        if tuple(model.input_shape[1:]) != MODEL_INPUTS[name]:
            raise RuntimeError(f"The saved {name} model has an unexpected input shape.")
        if model.output_shape[-1] != len(class_order):
            raise RuntimeError(f"The saved {name} model does not match the class order.")

    _loaded = models, scaler, temporal_scaler, encoder, metrics
    return _loaded


def model_status() -> Dict[str, Any]:
    try:
        models, _scaler, _temporal_scaler, encoder, metrics = _load_artifacts()
    except RuntimeError as error:
        model_statuses = {
            name: {
                "artifact_exists": (MODEL_DIR / filename).is_file(),
                "loaded": False,
                "ready_for_inference": False,
                "detail": str(error),
            }
            for name, filename in MODEL_FILES.items()
        }
        return {
            "ready": False,
            "detail": str(error),
            "loaded_models": [],
            "loaded_model_count": 0,
            "models": model_statuses,
        }
    loaded_models = list(models)
    model_statuses = {
        name: {
            "artifact_exists": (MODEL_DIR / MODEL_FILES[name]).is_file(),
            "loaded": name in models,
            "ready_for_inference": name in {"cnn", "cnn_bilstm"} and name in models,
            "detail": (
                "Loaded and used by the audio inference pipeline."
                if name in {"cnn", "cnn_bilstm"}
                else "Loaded successfully; not used by the live audio inference pipeline."
            ),
        }
        for name in MODEL_FILES
    }
    return {
        "ready": True,
        "detail": "All four trained models and preprocessing artifacts loaded successfully.",
        "loaded_models": loaded_models,
        "loaded_model_count": len(loaded_models),
        "models": model_statuses,
        "class_order": list(encoder.classes_),
        "metrics": metrics,
    }


def predict_audio(audio_bytes: bytes) -> Tuple[Dict[str, Any], float]:
    try:
        duration = float(sf.info(io.BytesIO(audio_bytes)).duration)
    except (RuntimeError, OSError, ValueError) as error:
        raise ValueError(f"Could not read WAV audio: {error}") from error
    if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"Audio must be between {MIN_DURATION_SECONDS:g} and "
            f"{MAX_DURATION_SECONDS:g} seconds long."
        )

    model_set, scaler, _temporal_scaler, encoder, _metrics = _load_artifacts()
    summary, _temporal = extract_features(io.BytesIO(audio_bytes))
    batch = scaler.transform(summary.reshape(1, -1)).astype(np.float32)[..., np.newaxis]
    cnn_probabilities = np.asarray(model_set["cnn"].predict(batch, verbose=0), dtype=np.float64)
    bilstm_probabilities = np.asarray(
        model_set["cnn_bilstm"].predict(batch, verbose=0), dtype=np.float64
    )
    expected_shape = (1, len(encoder.classes_))
    if cnn_probabilities.shape != expected_shape or bilstm_probabilities.shape != expected_shape:
        raise RuntimeError("A model output does not match the saved class order.")
    if not np.isfinite(cnn_probabilities).all() or not np.isfinite(bilstm_probabilities).all():
        raise RuntimeError("A model returned invalid probability values.")
    for name, probabilities in (
        ("CNN", cnn_probabilities),
        ("CNN-BiLSTM", bilstm_probabilities),
    ):
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise RuntimeError(f"{name} returned values outside the probability range.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-4, atol=1e-4):
            raise RuntimeError(f"{name} returned probabilities that do not sum to 1.")

    ensemble_probabilities = (cnn_probabilities + bilstm_probabilities) / 2.0

    def result(probabilities: np.ndarray) -> Dict[str, Any]:
        values = probabilities[0]
        best_index = int(np.argmax(values))
        return {
            "emotion": str(encoder.classes_[best_index]),
            "confidence": float(values[best_index]),
            "probabilities": {
                str(label): float(value)
                for label, value in zip(encoder.classes_, values)
            },
        }

    return {
        "cnn": result(cnn_probabilities),
        "cnn_bilstm": result(bilstm_probabilities),
        "ensemble": result(ensemble_probabilities),
    }, duration