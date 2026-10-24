from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------- Configuration ----------------
MANIFEST = Path("dataset/cnrpark_ext/metadata/cnr_ext_frame_manifest.csv")
MODEL_PATH = Path("models/cnr_ext_cnn_fast.keras")
OUTPUT_DIR = Path("outputs/cnr_ext_fast_demo")

DEMO_DATE = "2015-11-12"
DEMO_CAMERA = "camera7"

IMAGE_SIZE = 96
THRESHOLD = 0.30
OUTPUT_FPS = 5

VIDEO_PATH = OUTPUT_DIR / "cnr_ext_camera7_demo.mp4"
CSV_PATH = OUTPUT_DIR / "camera7_demo_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "camera7_demo_summary.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CNR-EXT manifest...")
    df = pd.read_csv(MANIFEST)

    selected = df[
        (df["date_group"].astype(str) == DEMO_DATE)
        & (df["camera_name"].astype(str) == DEMO_CAMERA)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No manifest rows found for {DEMO_DATE}, {DEMO_CAMERA}"
        )

    selected["datetime_parsed"] = pd.to_datetime(
        selected["datetime_parsed"], errors="coerce"
    )
    selected = selected.dropna(subset=["datetime_parsed"])

    # One full-frame image can have many parking-slot rows.
    frames = (
        selected[["frame_path", "datetime_parsed"]]
        .drop_duplicates(subset=["frame_path"])
        .sort_values("datetime_parsed")
    )

    if frames.empty:
        raise ValueError("No valid timestamped frames were found.")

    print(f"Selected frames: {len(frames)}")
    print(f"Camera: {DEMO_CAMERA} | Date: {DEMO_DATE}")
    print(f"Loading model: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = int(input_shape[1] or IMAGE_SIZE)
    width = int(input_shape[2] or IMAGE_SIZE)

    rows_by_frame = {
        path: group.copy()
        for path, group in selected.groupby("frame_path", sort=False)
    }

    writer = None
    results = []
    processed_frames = 0
    total_slots = 0
    correct_slots = 0

    for frame_number, (_, frame_info) in enumerate(frames.iterrows(), start=1):
        frame_path = str(frame_info["frame_path"])
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: cannot read {frame_path}")
            continue

        frame_rows = rows_by_frame[frame_path]
        crops = []
        valid_rows = []

        for _, row in frame_rows.iterrows():
            x1, y1, x2, y2 = [
                int(round(float(row[col])))
                for col in ("x1", "y1", "x2", "y2")
            ]

            # Clamp coordinates to the actual frame boundaries.
            x1 = max(0, min(x1, frame.shape[1] - 1))
            x2 = max(0, min(x2, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            y2 = max(0, min(y2, frame.shape[0]))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (width, height))
            crops.append(crop.astype(np.float32) / 255.0)
            valid_rows.append((row, x1, y1, x2, y2))

        if not crops:
            continue

        batch = np.asarray(crops, dtype=np.float32)
        probabilities = model.predict(batch, verbose=0).reshape(-1)

        frame_free = 0
        frame_busy = 0

        for (row, x1, y1, x2, y2), probability in zip(
            valid_rows, probabilities
        ):
            # Model output is P(occupied).
            predicted = int(probability >= THRESHOLD)
            actual = int(row["occupancy"])

            pred_label = "OCCUPIED" if predicted else "FREE"
            actual_label = "OCCUPIED" if actual else "FREE"

            # Green = predicted free; red = predicted occupied.
            color = (0, 0, 255) if predicted else (0, 200, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{pred_label} {probability:.2f}"
            text_y = max(18, y1 - 5)
            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

            slot_id = row["slot_id"]
            results.append({
                "date": DEMO_DATE,
                "camera": DEMO_CAMERA,
                "timestamp": str(row["datetime_parsed"]),
                "frame_path": frame_path,
                "slot_id": slot_id,
                "actual": actual_label,
                "predicted": pred_label,
                "occupied_probability": float(probability),
                "correct": bool(predicted == actual),
            })

            total_slots += 1
            correct_slots += int(predicted == actual)
            frame_busy += predicted
            frame_free += 1 - predicted

        timestamp_text = str(frame_info["datetime_parsed"])
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (25, 25, 25), -1)
        cv2.putText(
            frame,
            f"CNR-EXT | {DEMO_CAMERA} | {timestamp_text}",
            (12, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Slots: {len(valid_rows)} | Free: {frame_free} | "
            f"Occupied: {frame_busy} | Threshold: {THRESHOLD:.2f}",
            (12, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(VIDEO_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                OUTPUT_FPS,
                (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create the output MP4.")

        writer.write(frame)
        processed_frames += 1
        print(
            f"Frame {frame_number}/{len(frames)} | "
            f"slots: {len(valid_rows)}"
        )

    if writer is not None:
        writer.release()

    results_df = pd.DataFrame(results)
    results_df.to_csv(CSV_PATH, index=False)

    summary = {
        "dataset": "CNR-EXT",
        "date": DEMO_DATE,
        "camera": DEMO_CAMERA,
        "video_type": "Video assembled from sampled dataset frames",
        "threshold": THRESHOLD,
        "output_fps": OUTPUT_FPS,
        "frames_written": processed_frames,
        "slot_predictions": total_slots,
        "correct_predictions": correct_slots,
        "demo_slot_accuracy": (
            round(correct_slots / total_slots, 4) if total_slots else None
        ),
        "video_path": str(VIDEO_PATH),
        "predictions_csv": str(CSV_PATH),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if processed_frames == 0:
        raise RuntimeError("No frames were written. Check frame paths.")

    print("\nDemo created successfully!")
    print(json.dumps(summary, indent=2))
    print(f"\nVideo: {VIDEO_PATH}")
    print(f"Predictions: {CSV_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
