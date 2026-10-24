
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "cnr_ext_cnn_fast.keras"
MANIFEST_PATH = (
    ROOT / "dataset" / "cnrpark_ext" / "metadata"
    / "cnr_ext_frame_manifest.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "cnr_ext_fast_demo"

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64

# Freeze the threshold selected using validation dates.
THRESHOLD = 0.30

# Held-out test dates: do not tune the threshold on these dates.
TEST_DATES = {"2015-12-03", "2015-12-10", "2016-01-09"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    print("Loading manifest...")
    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
        "frame_path", "camera_num", "datetime_parsed",
        "date_group", "slot_id", "occupancy",
        "x1", "y1", "x2", "y2",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    manifest["date_group"] = manifest["date_group"].astype(str)

    # Keep all annotated slots from all held-out test dates.
    test = manifest[
        manifest["date_group"].isin(TEST_DATES)
    ].copy()

    if test.empty:
        raise ValueError(f"No rows found for test dates: {sorted(TEST_DATES)}")

    # Validate labels before inference.
    test["occupancy"] = pd.to_numeric(test["occupancy"], errors="raise").astype(int)
    if not test["occupancy"].isin([0, 1]).all():
        raise ValueError("Expected occupancy labels 0=free and 1=busy.")

    print("Test dates:", sorted(TEST_DATES))
    print("Annotated slot rows:", len(test))
    print("Unique frames:", test["frame_path"].nunique())
    print("Loading model...")

    model = tf.keras.models.load_model(MODEL_PATH)
    if isinstance(model.output_shape, list) or model.output_shape[-1] != 1:
        raise ValueError("Expected a single sigmoid output representing P(occupied).")

    all_results = []
    frame_cache = {}
    total_rows = len(test)

    # Process one complete frame at a time, with predictions batched by frame.
    for frame_number, (frame_path_value, frame_rows) in enumerate(
        test.groupby("frame_path", sort=True), start=1
    ):
        frame_path = Path(str(frame_path_value))

        if not frame_path.is_absolute():
            frame_path = ROOT / frame_path

        cache_key = str(frame_path)

        if cache_key not in frame_cache:
            frame = cv2.imread(cache_key)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame: {frame_path}")
            frame_cache[cache_key] = frame

        frame = frame_cache[cache_key]
        height, width = frame.shape[:2]

        crops = []
        valid_rows = []

        for row in frame_rows.itertuples(index=False):
            x1 = max(0, min(width, int(row.x1)))
            y1 = max(0, min(height, int(row.y1)))
            x2 = max(0, min(width, int(row.x2)))
            y2 = max(0, min(height, int(row.y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(
                crop,
                IMAGE_SIZE,
                interpolation=cv2.INTER_AREA,
            )

            crops.append(crop.astype(np.float32) / 255.0)
            valid_rows.append(row)

        if not crops:
            continue

        probabilities = model.predict(
            np.stack(crops),
            batch_size=BATCH_SIZE,
            verbose=0,
        ).reshape(-1)

        for row, probability in zip(valid_rows, probabilities):
            true_label = int(row.occupancy)
            busy_probability = float(probability)
            predicted_label = int(busy_probability >= THRESHOLD)

            all_results.append({
                "date": str(row.date_group),
                "camera": int(row.camera_num),
                "datetime": str(row.datetime_parsed),
                "slot_id": int(row.slot_id),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "busy_probability": busy_probability,
                "threshold": THRESHOLD,
                "correct": bool(true_label == predicted_label),
            })

        if frame_number % 25 == 0:
            print(
                f"Processed {frame_number} frames; "
                f"{len(all_results)} slot predictions collected."
            )

    if not all_results:
        raise RuntimeError("No predictions generated. Check frame paths and crops.")

    results_df = pd.DataFrame(all_results)
    results_csv = OUTPUT_DIR / "heldout_test_predictions.csv"
    results_df.to_csv(results_csv, index=False)

    y_true = results_df["true_label"].to_numpy()
    y_pred = results_df["predicted_label"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    report = {
        "evaluation_type": "held-out full-frame test",
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "test_dates": sorted(TEST_DATES),
        "threshold_frozen_from_validation": THRESHOLD,
        "number_of_slots_evaluated": int(len(results_df)),
        "number_of_unique_frames": int(
            results_df[["date", "camera", "datetime"]]
            .drop_duplicates()
            .shape[0]
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix_rows_true_columns_predicted": cm.tolist(),
        "label_mapping": {"0": "free", "1": "occupied"},
        "free_class": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "occupied_class": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "errors": {
            "free_predicted_occupied": int(np.sum((y_true == 0) & (y_pred == 1))),
            "occupied_predicted_free": int(np.sum((y_true == 1) & (y_pred == 0))),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["free", "occupied"],
            output_dict=True,
            zero_division=0,
        ),
        "note": (
            "Threshold was selected on validation data and frozen before "
            "held-out test evaluation. No threshold tuning was performed here."
        ),
    }

    report_path = OUTPUT_DIR / "heldout_test_evaluation.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n========== HELD-OUT FULL-FRAME TEST ==========")
    print(f"Frozen threshold: {THRESHOLD:.2f}")
    print(f"Slots evaluated: {len(results_df)}")
    print(f"Unique frames: {report['number_of_unique_frames']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(cm)
    print("\nClass metrics:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["free", "occupied"],
        zero_division=0,
    ))
    print("Free predicted occupied:", report["errors"]["free_predicted_occupied"])
    print("Occupied predicted free:", report["errors"]["occupied_predicted_free"])
    print("\nSaved predictions:", results_csv)
    print("Saved evaluation:", report_path)


if __name__ == "__main__":
    main()
