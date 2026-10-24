
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "cnr_ext_cnn_fast.keras"
MANIFEST_PATH = (
    ROOT / "dataset" / "cnrpark_ext" / "metadata"
    / "cnr_ext_frame_manifest.csv"
)
ANNOTATION_DIR = (
    ROOT / "dataset" / "cnrpark_ext" / "full_images"
    / "FULL_IMAGE_1000x750"
)
OUTPUT_DIR = ROOT / "outputs" / "cnr_ext_fast_demo"

IMAGE_SIZE = (96, 96)

# Use only dates that were reserved for the test split.
TEST_DATES = ["2015-12-03", "2015-12-10", "2016-01-09"]


def find_frame(camera, datetime_value):
    """Find a full-frame image matching the metadata timestamp."""
    timestamp = pd.to_datetime(datetime_value).strftime("%Y-%m-%d_%H%M")

    matches = list(
        ANNOTATION_DIR.glob(f"**/camera{camera}/{timestamp}.jpg")
    )

    if not matches:
        matches = list(
            ANNOTATION_DIR.glob(f"**/camera{camera}/{timestamp}.jpeg")
        )

    return matches[0] if matches else None


def load_camera_annotations(camera):
    """Load slot rectangles in original 2592x1944 coordinates."""
    candidates = [
        ROOT / "dataset" / "cnrpark_ext" / "full_images"
        / f"camera{camera}.csv",
        ROOT / "dataset" / "cnrpark_ext" / "full_images"
        / "FULL_IMAGE_1000x750" / f"camera{camera}.csv",
    ]

    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"Could not find annotation CSV for camera {camera}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["datetime"] = pd.to_datetime(manifest["datetime_parsed"])
    manifest["date"] = manifest["datetime"].dt.strftime("%Y-%m-%d")

    # One held-out frame per date and camera, where available.
    selected = (
        manifest[manifest["date"].isin(TEST_DATES)]
        .sort_values("datetime")
        .groupby(["date", "camera_num"], as_index=False)
        .first()
    )

    model = tf.keras.models.load_model(MODEL_PATH)

    all_results = []

    for _, frame_row in selected.iterrows():
        camera = int(frame_row["camera_num"])
        timestamp = frame_row["datetime"]
        frame_path = Path(frame_row["frame_path"])

        if not frame_path.exists():
            frame_path = find_frame(camera, timestamp)

        if frame_path is None or not frame_path.exists():
            print(f"Skipping missing frame: camera {camera}, {timestamp}")
            continue

        image = Image.open(frame_path).convert("RGB")
        image_array = np.asarray(image)
        height, width = image_array.shape[:2]

        # Use the manifest's annotations for this exact frame.
        frame_rows = manifest[
            (manifest["date"] == frame_row["date"])
            & (manifest["camera_num"] == camera)
            & (manifest["datetime"] == timestamp)
        ]

        crops = []
        crop_rows = []

        for _, row in frame_rows.iterrows():
            x1 = max(0, min(width, int(round(row["x1"]))))
            y1 = max(0, min(height, int(round(row["y1"]))))
            x2 = max(0, min(width, int(round(row["x2"]))))
            y2 = max(0, min(height, int(round(row["y2"]))))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2)).resize(IMAGE_SIZE)
            crops.append(np.asarray(crop, dtype=np.float32) / 255.0)
            crop_rows.append(row)

        if not crops:
            print(f"No valid crops: camera {camera}, {timestamp}")
            continue

        predictions = model.predict(
            np.asarray(crops, dtype=np.float32),
            batch_size=64,
            verbose=0,
        ).reshape(-1)

        # Model output is the probability of BUSY/occupied.
        for row, probability in zip(crop_rows, predictions):
            true_label = int(row["occupancy"])
            predicted_label = int(probability >= 0.5)

            all_results.append({
                "date": frame_row["date"],
                "camera": camera,
                "datetime": str(timestamp),
                "slot_id": int(row["slot_id"]),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "busy_probability": float(probability),
                "correct": bool(true_label == predicted_label),
            })

        # Annotate the full frame for visual inspection.
        preview = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        for row, probability in zip(crop_rows, predictions):
            x1, y1 = int(row["x1"]), int(row["y1"])
            x2, y2 = int(row["x2"]), int(row["y2"])

            predicted_busy = probability >= 0.5
            color = (0, 0, 255) if predicted_busy else (0, 200, 0)

            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            text = f'{int(row["slot_id"])}: {probability:.2f}'
            cv2.putText(
                preview, text, (x1, max(20, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        output_name = (
            f'{frame_row["date"]}_camera{camera}_prediction.jpg'
        )
        cv2.imwrite(str(OUTPUT_DIR / output_name), preview)

        print(
            f'{frame_row["date"]} | camera {camera} | '
            f'{len(crops)} slots evaluated'
        )

    if not all_results:
        raise RuntimeError(
            "No predictions were generated. Check the manifest and frame paths."
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    y_true = results_df["true_label"].to_numpy()
    y_pred = results_df["predicted_label"].to_numpy()

    cm = tf.math.confusion_matrix(
        y_true, y_pred, num_classes=2
    ).numpy().tolist()

    report = {
        "test_dates": TEST_DATES,
        "number_of_slots": int(len(results_df)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "confusion_matrix_rows_true_columns_predicted": cm,
        "label_mapping": {"0": "free", "1": "busy"},
        "interpretation": {
            "free_predicted_busy": int(np.sum((y_true == 0) & (y_pred == 1))),
            "busy_predicted_free": int(np.sum((y_true == 1) & (y_pred == 0))),
        },
    }

    with open(OUTPUT_DIR / "evaluation.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n========== HELD-OUT FULL-FRAME RESULTS ==========")
    print(json.dumps(report, indent=2))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
