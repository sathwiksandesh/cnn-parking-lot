
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "cnrpark_ext"
    / "metadata"
    / "cnr_ext_frame_manifest.csv"
)

MODEL_PATH = PROJECT_ROOT / "models" / "cnr_ext_cnn_fast.keras"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cnr_ext_fast_demo"
OUTPUT_JSON = OUTPUT_DIR / "full_frame_validation.json"

IMAGE_SIZE = 96

# These are the validation dates from the saved date-based split.
VALIDATION_DATES = {"2015-11-12", "2016-01-13"}

# Test several candidate thresholds on validation data only.
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def load_model_and_manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
    "frame_path",
    "camera_num",
    "camera_name",
    "datetime_parsed",
    "date_group",
    "slot_id",
    "occupancy",
    "x1",
    "y1",
    "x2",
    "y2",
}

    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    manifest["date_group"] = manifest["date_group"].astype(str)

    validation = manifest[
        manifest["date_group"].isin(VALIDATION_DATES)
    ].copy()

    if validation.empty:
        raise ValueError(
            "No validation rows found for the configured dates. "
            "Check date_group values in the manifest."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    output_shape = model.output_shape
    if isinstance(output_shape, list) or output_shape[-1] != 1:
        raise ValueError(
            "Expected the CNR-EXT model to have one sigmoid output."
        )

    return model, validation


def predict_validation_frames(model, validation):
    # Predict every annotated slot in the validation frames.
    # Cache each decoded frame to avoid repeatedly reading it.
    frame_cache = {}
    records = []

    for row in validation.itertuples(index=False):
        frame_path = Path(row.frame_path)

        if not frame_path.is_absolute():
            frame_path = PROJECT_ROOT / frame_path

        if not frame_path.is_file():
            raise FileNotFoundError(
                f"Full-frame image not found: {frame_path}"
            )

        cache_key = str(frame_path)

        if cache_key not in frame_cache:
            frame = cv2.imread(cache_key)

            if frame is None:
                raise ValueError(
                    f"Could not read image: {frame_path}"
                )

            frame_cache[cache_key] = frame

        frame = frame_cache[cache_key]

        height, width = frame.shape[:2]

        x1 = max(0, min(width, int(row.x1)))
        y1 = max(0, min(height, int(row.y1)))
        x2 = max(0, min(width, int(row.x2)))
        y2 = max(0, min(height, int(row.y2)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid crop coordinates for slot {row.slot_id}: "
                f"{x1}, {y1}, {x2}, {y2}"
            )

        crop = frame[y1:y2, x1:x2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        image = rgb.astype(np.float32) / 255.0

        records.append({
            "date": str(row.date_group),
            "camera": str(row.camera_num),
            "datetime": str(row.datetime_parsed),
            "slot_id": str(row.slot_id),
            "true_label": int(row.occupancy),
            "image": image,
        })

    if not records:
        raise ValueError("No validation crops were generated.")

    images = np.stack([record["image"] for record in records])

    # Predict in batches to control memory use.
    probabilities = model.predict(
        images,
        batch_size=64,
        verbose=1,
    ).reshape(-1)

    for record, probability in zip(records, probabilities):
        record["busy_probability"] = float(probability)
        del record["image"]

    return records


def evaluate_threshold(records, threshold):
    y_true = np.array(
        [record["true_label"] for record in records],
        dtype=int,
    )

    busy_probabilities = np.array(
        [record["busy_probability"] for record in records],
        dtype=float,
    )

    y_pred = (busy_probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "free_precision": float(precision[0]),
        "free_recall": float(recall[0]),
        "free_f1": float(f1[0]),
        "occupied_precision": float(precision[1]),
        "occupied_recall": float(recall[1]),
        "occupied_f1": float(f1[1]),
        "free_correct": int(tn),
        "free_predicted_occupied": int(fp),
        "occupied_predicted_free": int(fn),
        "occupied_correct": int(tp),
        "support_free": int(support[0]),
        "support_occupied": int(support[1]),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and validation manifest...")
    model, validation = load_model_and_manifest()

    print("Validation dates:", sorted(VALIDATION_DATES))
    print("Validation slot crops:", len(validation))
    print("Unique validation frames:", validation["frame_path"].nunique())

    print("\nPredicting validation crops...")
    records = predict_validation_frames(model, validation)

    results = [
        evaluate_threshold(records, threshold)
        for threshold in THRESHOLDS
    ]

    print("\n=== FULL-FRAME VALIDATION RESULTS ===")
    print(
        "Threshold | Accuracy | Occupied Precision | "
        "Occupied Recall | Occupied F1 | Busy->Free"
    )

    for result in results:
        print(
            f"{result['threshold']:9.2f} | "
            f"{result['accuracy']:.4f}   | "
            f"{result['occupied_precision']:.4f}             | "
            f"{result['occupied_recall']:.4f}          | "
            f"{result['occupied_f1']:.4f}      | "
            f"{result['occupied_predicted_free']}"
        )

    # Select the threshold with the best occupied-class F1.
    # This choice uses validation data only, not the test data.
    best = max(
        results,
        key=lambda result: (
            result["occupied_f1"],
            result["occupied_recall"],
        ),
    )

    output = {
        "model_path": str(MODEL_PATH),
        "evaluation_type": "full_frame_validation",
        "validation_dates": sorted(VALIDATION_DATES),
        "validation_slots": len(records),
        "validation_frames": len({
            (r["date"], r["camera"], r["datetime"])
            for r in records
        }),
        "threshold_results": results,
        "selected_threshold_by_occupied_f1": best["threshold"],
        "selected_threshold_metrics": best,
        "note": (
            "Threshold selected on validation data. "
            "Keep the held-out test dates untouched for final evaluation."
        ),
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\n=== SELECTED VALIDATION THRESHOLD ===")
    print("Threshold:", best["threshold"])
    print("Accuracy:", round(best["accuracy"], 4))
    print("Occupied precision:", round(best["occupied_precision"], 4))
    print("Occupied recall:", round(best["occupied_recall"], 4))
    print("Occupied F1:", round(best["occupied_f1"], 4))
    print("Confusion matrix:")
    print(np.array(best["confusion_matrix"]))
    print("\nSaved report:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
