import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import soundfile as sf
from sklearn.preprocessing import LabelEncoder, StandardScaler

from training.features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG_PATH = PROJECT_ROOT / "dataset_config.json"
MODEL_DIR = PROJECT_ROOT / "training" / "artifacts"
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
RAVDESS_PATTERN = re.compile(
    r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$",
    re.IGNORECASE,
)
SPLIT_ACTORS = {
    "train": frozenset(range(1, 19)),
    "validation": frozenset(range(19, 22)),
    "test": frozenset(range(22, 25)),
}
logger = logging.getLogger(__name__)
Record = Tuple[Path, str, int]


def get_dataset_root() -> Path:
    with DATASET_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    root = Path(config["dataset_path"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RAVDESS dataset directory not found: {root}")
    return root


def scan_dataset(root: Path, require_all_emotions: bool = True) -> List[Record]:
    if not root.is_dir():
        raise FileNotFoundError(f"RAVDESS dataset directory not found: {root}")

    records: List[Record] = []
    seen_filenames = set()
    ignored = 0
    duplicates = 0
    unreadable = 0
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        match = RAVDESS_PATTERN.fullmatch(path.name)
        if match is None:
            ignored += 1
            continue
        modality, channel, emotion_code, _intensity, _statement, _repetition, actor_text = match.groups()
        if modality != "03" or channel != "01" or emotion_code not in EMOTIONS:
            ignored += 1
            continue
        actor = int(actor_text)
        if actor not in range(1, 25):
            ignored += 1
            continue
        try:
            if sf.info(str(path)).frames <= 0:
                raise ValueError("WAV contains no audio frames")
        except (OSError, RuntimeError, ValueError) as error:
            unreadable += 1
            logger.warning("Skipped unreadable WAV file %s: %s", path, error)
            continue
        filename_key = path.name.casefold()
        if filename_key in seen_filenames:
            duplicates += 1
            continue
        seen_filenames.add(filename_key)
        records.append((path, EMOTIONS[emotion_code], actor))

    if ignored:
        logger.info("Ignored %d WAV files outside the selected RAVDESS speech classes.", ignored)
    if duplicates:
        logger.warning("Skipped %d duplicate WAV filenames.", duplicates)
    if unreadable:
        logger.warning("Skipped %d unreadable WAV files.", unreadable)
    if not records:
        raise ValueError(f"No valid RAVDESS speech files found in {root}")

    found_emotions = {record[1] for record in records}
    missing = sorted(set(EMOTIONS.values()) - found_emotions)
    if missing and require_all_emotions:
        raise ValueError("Dataset is missing these emotion classes: " + ", ".join(missing))
    return records


def split_by_actor(records: List[Record]) -> Dict[str, np.ndarray]:
    actors = np.asarray([record[2] for record in records], dtype=np.int64)
    splits = {
        name: np.flatnonzero(np.isin(actors, sorted(actor_ids)))
        for name, actor_ids in SPLIT_ACTORS.items()
    }
    assigned = np.concatenate(list(splits.values()))
    if len(assigned) != len(records) or len(np.unique(assigned)) != len(records):
        raise RuntimeError("The fixed actor split did not assign each record exactly once.")
    return splits


def extract_dataset(
    records: List[Record],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Record], LabelEncoder]:
    encoder = LabelEncoder()
    encoder.fit([record[1] for record in records])
    summary_features: List[np.ndarray] = []
    temporal_features: List[np.ndarray] = []
    labels: List[int] = []
    readable_records: List[Record] = []
    for index, record in enumerate(records, start=1):
        try:
            summary, temporal = extract_features(record[0])
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning("Skipping unreadable audio file %s: %s", record[0].name, error)
            continue
        summary_features.append(summary)
        temporal_features.append(temporal)
        labels.append(int(encoder.transform([record[1]])[0]))
        readable_records.append(record)
        if index % 250 == 0:
            logger.info("Processed %d/%d audio files.", index, len(records))

    if not summary_features:
        raise ValueError("No readable audio files were found.")
    found_labels = set(labels)
    if len(found_labels) != len(encoder.classes_):
        readable_classes = {encoder.classes_[label] for label in found_labels}
        missing = sorted(set(encoder.classes_) - readable_classes)
        raise ValueError(
            "No readable audio files remain for these emotion classes: "
            + ", ".join(missing)
        )
    return (
        np.stack(summary_features),
        np.stack(temporal_features),
        np.asarray(labels, dtype=np.int64),
        readable_records,
        encoder,
    )


def scale_features(
    summary_features: np.ndarray,
    temporal_features: np.ndarray,
    splits: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    train_indices = splits["train"]
    if train_indices.size == 0:
        raise ValueError("The training actor split contains no readable records.")

    scaler = StandardScaler().fit(summary_features[train_indices])
    temporal_scaler = StandardScaler().fit(
        temporal_features[train_indices].reshape(-1, temporal_features.shape[-1])
    )
    scaled_summary = scaler.transform(summary_features).astype(np.float32)[..., np.newaxis]
    scaled_temporal = temporal_scaler.transform(
        temporal_features.reshape(-1, temporal_features.shape[-1])
    ).reshape(temporal_features.shape).astype(np.float32)
    return scaled_summary, scaled_temporal, scaler, temporal_scaler


def save_preprocessing_artifacts(
    scaler: StandardScaler,
    temporal_scaler: StandardScaler,
    encoder: LabelEncoder,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(temporal_scaler, MODEL_DIR / "temporal_scaler.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")