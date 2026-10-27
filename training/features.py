from pathlib import Path
from typing import Tuple

import librosa
import numpy as np

SAMPLE_RATE = 22050
MFCC_COUNT = 40
TEMPORAL_FRAMES = 64


def extract_features(source: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the notebook's 108-value summary and 64-by-54 temporal features."""
    audio, _sample_rate = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file contains no samples: {source}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio file contains invalid sample values: {source}")

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_COUNT)
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)
    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)

    summary = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            zero_crossing.mean(axis=1),
            zero_crossing.std(axis=1),
            rms.mean(axis=1),
            rms.std(axis=1),
            chroma.mean(axis=1),
            chroma.std(axis=1),
        ]
    ).astype(np.float32)

    frame_features = np.concatenate(
        [mfcc, zero_crossing, rms, chroma], axis=0
    )
    source_frames = np.arange(frame_features.shape[1], dtype=np.float32)
    target_frames = np.linspace(
        0, frame_features.shape[1] - 1, TEMPORAL_FRAMES, dtype=np.float32
    )
    temporal = np.stack(
        [np.interp(target_frames, source_frames, row) for row in frame_features],
        axis=1,
    ).astype(np.float32)

    if summary.shape != (108,) or temporal.shape != (TEMPORAL_FRAMES, 54):
        raise RuntimeError(
            f"Unexpected feature shapes: {summary.shape} and {temporal.shape}"
        )
    return summary, temporal