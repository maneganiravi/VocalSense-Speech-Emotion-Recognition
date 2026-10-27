import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import layers, models, regularizers

from training.preprocessing import (
    MODEL_DIR,
    extract_dataset,
    get_dataset_root,
    save_preprocessing_artifacts,
    scan_dataset,
    scale_features,
    split_by_actor,
)

SEED = 42
EPOCHS = 40
BATCH_SIZE = 32
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_cnn(input_shape: Tuple[int, int], class_count: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            layers.Conv1D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(class_count, activation="softmax"),
        ],
        name="cnn",
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_bilstm(input_shape: Tuple[int, int], class_count: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(class_count, activation="softmax"),
        ],
        name="cnn_bilstm",
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_temporal_model(input_shape: Tuple[int, int], class_count: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            layers.Bidirectional(layers.LSTM(48)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(class_count, activation="softmax"),
        ],
        name="temporal_cnn_bilstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_regularized_model(input_shape: Tuple[int, int], class_count: int) -> tf.keras.Model:
    l2 = regularizers.l2(1e-4)
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 5, padding="same", activation="relu", kernel_regularizer=l2),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.35),
            layers.Bidirectional(
                layers.LSTM(24, kernel_regularizer=l2, recurrent_regularizer=l2)
            ),
            layers.Dense(32, activation="relu", kernel_regularizer=l2),
            layers.Dropout(0.45),
            layers.Dense(class_count, activation="softmax"),
        ],
        name="regularized_cnn_bilstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _score_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Iterable[str],
) -> Dict[str, Any]:
    names = list(class_names)
    predictions = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(len(names)), zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(names)))
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_precision": float(np.mean(precision)),
        "weighted_precision": float(np.average(precision, weights=support)) if support.sum() else 0.0,
        "macro_recall": float(np.mean(recall)),
        "weighted_recall": float(np.average(recall, weights=support)) if support.sum() else 0.0,
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def _train_callbacks(patience: int) -> List[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2 if patience == 6 else 3, min_lr=1e-6,
        ),
    ]


def _fit_model(
    model: tf.keras.Model,
    name: str,
    filename: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    epochs: int = EPOCHS,
    patience: int = 8,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    logger.info("Training %s on %d samples for up to %d epochs.", name, len(train_labels), epochs)
    fit = model.fit(
        train_features,
        train_labels,
        validation_data=(validation_features, validation_labels),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=_train_callbacks(patience),
        verbose=1,
    )
    model.save(MODEL_DIR / filename)
    history = {
        key: [float(value) for value in values]
        for key, values in fit.history.items()
        if key in {"accuracy", "val_accuracy", "loss", "val_loss"}
    }
    history["best_epoch"] = int(np.argmin(history["val_loss"]) + 1)
    return model, history


def _dataset_metrics(records: List[Tuple[Path, str, int]], dataset_root: Path) -> Dict[str, Any]:
    distribution: Dict[str, int] = {}
    for _path, emotion, _actor in records:
        distribution[emotion] = distribution.get(emotion, 0) + 1
    return {
        "total_wav_files": sum(1 for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
        "selected_clips": len(records),
        "emotion_distribution": distribution,
        "sample_records": [
            {"filename": path.name, "emotion": emotion, "actor": actor, "path": str(path.relative_to(dataset_root))}
            for path, emotion, actor in records[:20]
        ],
    }


def _split_metrics(
    records: List[Tuple[Path, str, int]],
    labels: np.ndarray,
    splits: Dict[str, np.ndarray],
    class_names: List[str],
) -> Dict[str, Any]:
    result = {}
    for name, indices in splits.items():
        counts = np.bincount(labels[indices], minlength=len(class_names))
        result[name] = {
            "samples": int(len(indices)),
            "actors": sorted({records[int(index)][2] for index in indices}),
            "class_distribution": {
                class_names[index]: int(count) for index, count in enumerate(counts)
            },
        }
    return result


def _write_csvs(metrics: Dict[str, Any]) -> None:
    comparison_path = MODEL_DIR / "model_comparison_final.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "model", "accuracy", "macro_f1", "weighted_f1"],
        )
        writer.writeheader()
        for model_name, model in metrics["models"].items():
            for split, scores in model["results"].items():
                writer.writerow(
                    {
                        "split": split,
                        "model": model_name,
                        "accuracy": scores["accuracy"],
                        "macro_f1": scores["macro_f1"],
                        "weighted_f1": scores["weighted_f1"],
                    }
                )

    report_path = MODEL_DIR / "classification_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["split", "model", "emotion", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for model_name, model in metrics["models"].items():
            for split, scores in model["results"].items():
                for emotion, values in scores["per_class"].items():
                    writer.writerow(
                        {"split": split, "model": model_name, "emotion": emotion, **values}
                    )


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset_root = get_dataset_root()
    records = scan_dataset(dataset_root)
    logger.info("Extracting features from %d unique speech clips.", len(records))
    summary, temporal, labels, records, encoder = extract_dataset(records)
    class_names = list(encoder.classes_)
    splits = split_by_actor(records)
    scaled_summary, scaled_temporal, scaler, temporal_scaler = scale_features(
        summary, temporal, splits
    )
    save_preprocessing_artifacts(scaler, temporal_scaler, encoder)

    train_indices = splits["train"]
    validation_indices = splits["validation"]
    split_features = {
        "cnn": scaled_summary,
        "cnn_bilstm": scaled_summary,
        "temporal": scaled_temporal,
        "regularized": scaled_temporal,
    }
    specifications = (
        ("cnn", build_cnn, (108, 1), "cnn_model.keras", EPOCHS, 8),
        ("cnn_bilstm", build_cnn_bilstm, (108, 1), "cnn_bilstm_model.keras", EPOCHS, 8),
        ("temporal", build_temporal_model, (64, 54), "temporal_model.keras", EPOCHS, 8),
        ("regularized", build_regularized_model, (64, 54), "regularized_model.keras", 30, 6),
    )

    results: Dict[str, Dict[str, Any]] = {}
    probabilities: Dict[str, Dict[str, np.ndarray]] = {}
    for name, builder, input_shape, filename, epochs, patience in specifications:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = builder(input_shape, len(class_names))
        model, history = _fit_model(
            model,
            name,
            filename,
            split_features[name][train_indices],
            labels[train_indices],
            split_features[name][validation_indices],
            labels[validation_indices],
            epochs,
            patience,
        )
        split_results = {}
        model_probabilities = {}
        for split_name, indices in splits.items():
            model_probabilities[split_name] = model.predict(
                split_features[name][indices], batch_size=BATCH_SIZE, verbose=0
            )
            split_results[split_name] = _score_probabilities(
                labels[indices], model_probabilities[split_name], class_names
            )
        results[name] = {
            "architecture": model.to_json(),
            "parameter_count": int(model.count_params()),
            "history": history,
            "results": split_results,
        }
        probabilities[name] = model_probabilities

    ensemble_results = {}
    for split_name, indices in splits.items():
        ensemble_probabilities = (
            probabilities["cnn"][split_name] + probabilities["cnn_bilstm"][split_name]
        ) / 2.0
        ensemble_results[split_name] = _score_probabilities(
            labels[indices], ensemble_probabilities, class_names
        )
    results["ensemble"] = {
        "architecture": "Mean of CNN and CNN-BiLSTM class probabilities.",
        "parameter_count": None,
        "history": None,
        "results": ensemble_results,
    }

    metrics = {
        "seed": SEED,
        "dataset": _dataset_metrics(records, dataset_root),
        "features": {
            "shape": list(summary.shape),
            "temporal_shape": list(temporal.shape[1:]),
            "summary_description": {
                "mfcc_mean_std": 80,
                "zero_crossing_rate_mean_std": 2,
                "rms_mean_std": 2,
                "chroma_stft_mean_std": 24,
                "sample_rate": 22050,
            },
            "scaler": {
                "type": "StandardScaler",
                "fit_samples": int(len(train_indices)),
                "feature_count": int(scaler.n_features_in_),
                "temporal_type": "StandardScaler fitted across flattened training frames",
                "temporal_fit_frames": int(len(train_indices) * temporal.shape[1]),
            },
        },
        "class_order": class_names,
        "splits": _split_metrics(records, labels, splits, class_names),
        "models": results,
    }
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    _write_csvs(metrics)
    logger.info("Saved models, scalers, metrics, and reports to %s.", MODEL_DIR)


if __name__ == "__main__":
    main()