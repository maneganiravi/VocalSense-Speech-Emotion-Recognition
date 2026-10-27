import io
from pathlib import Path
from typing import Any, Dict, Union

import librosa
import numpy as np
import soundfile as sf


DEFAULT_CONFIG: Dict[str, Any] = {
    "sample_rate": 22050,
    "duration_seconds": 4.0,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 256,
    "fmin": 20,
    "top_db": 80,
    "scaling": "clip_db_then_divide_by_top_db",
}


def extract_features(
    source: Union[str, Path, bytes], config: Dict[str, Any] = DEFAULT_CONFIG
) -> np.ndarray:
    """Return a fixed-shape, normalized log-mel spectrogram."""
    if isinstance(source, bytes):
        audio, sample_rate = sf.read(io.BytesIO(source), dtype="float32", always_2d=False)
    else:
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)

    if audio.size == 0:
        raise ValueError("The audio file contains no samples.")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if not np.isfinite(audio).all():
        raise ValueError("The audio file contains invalid sample values.")
    if sample_rate != config["sample_rate"]:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=config["sample_rate"]
        )

    target_samples = int(config["sample_rate"] * config["duration_seconds"])
    audio = audio[:target_samples]
    if audio.size < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.size))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["sample_rate"] // 2,
        power=2.0,
        center=False,
    )
    db = librosa.power_to_db(mel, ref=np.max, top_db=config["top_db"])
    normalized = (np.clip(db, -config["top_db"], 0.0) + config["top_db"]) / config[
        "top_db"
    ]
    return normalized.astype(np.float32)[..., np.newaxis]
