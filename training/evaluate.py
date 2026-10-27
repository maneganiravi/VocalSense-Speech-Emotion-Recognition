import json
from pathlib import Path
from typing import Any, Dict

import joblib
from tensorflow.keras.models import load_model

from training.preprocessing import MODEL_DIR

EXPECTED_MODELS = {
    "cnn": ("cnn_model.keras", (108, 1)),
    "cnn_bilstm": ("cnn_bilstm_model.keras", (108, 1)),
    "temporal": ("temporal_model.keras", (64, 54)),
    "regularized": ("regularized_model.keras", (64, 54)),
}


def load_metrics() -> Dict[str, Any]:
    path = MODEL_DIR / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Training metrics are missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    class_order = metrics.get("class_order")
    if (
        not isinstance(class_order, list)
        or not class_order
        or not all(isinstance(label, str) for label in class_order)
        or len(class_order) != len(set(class_order))
    ):
        raise ValueError("Metrics must contain a non-empty, unique class order.")
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    if not encoder_path.is_file():
        raise FileNotFoundError(f"Training label encoder is missing: {encoder_path}")
    encoder = joblib.load(encoder_path)
    if class_order != list(encoder.classes_):
        raise ValueError("Metrics class order does not match the saved label encoder.")
    systems = metrics.get("models")
    if not isinstance(systems, dict):
        raise ValueError("Metrics do not contain model evaluation results.")
    for name in (*EXPECTED_MODELS, "ensemble"):
        result = systems.get(name, {}).get("results")
        if not isinstance(result, dict) or not {"validation", "test"}.issubset(result):
            raise ValueError(f"Validation/test metrics are missing for {name}.")
    return metrics


def main() -> None:
    metrics = load_metrics()
    class_count = len(metrics["class_order"])
    for name, (filename, expected_input) in EXPECTED_MODELS.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Trained model is missing: {path}")
        model = load_model(path, compile=False)
        if tuple(model.input_shape[1:]) != expected_input:
            raise ValueError(f"{name} input shape does not match its saved metrics.")
        if model.output_shape[-1] != class_count:
            raise ValueError(f"{name} output width does not match the saved class order.")
        print(f"Loaded {name}: input={expected_input}, classes={class_count}")

    for name, system in metrics["models"].items():
        for split in ("validation", "test"):
            scores = system["results"][split]
            print(
                f"{name} {split}: accuracy={scores['accuracy']:.4f}, "
                f"macro_f1={scores['macro_f1']:.4f}, "
                f"weighted_f1={scores['weighted_f1']:.4f}"
            )


if __name__ == "__main__":
    main()