
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

MANIFEST = Path("dataset/cnrpark_ext/metadata/cnr_ext_frame_manifest.csv")
MODEL_PATH = Path("models/cnr_ext_cnn_fast.keras")
OUT_DIR = Path("outputs/cnr_ext_fast_demo")

DATE = "2015-11-12"
CAMERA = "camera7"
THRESHOLD = 0.30
SIZE = 96
FPS = 5

VIDEO_PATH = OUT_DIR / "cnr_ext_camera7_review.mp4"
CSV_PATH = OUT_DIR / "cnr_ext_camera7_review.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST, low_memory=False)
    df["date_group"] = df["date_group"].astype(str)
    df["datetime_parsed"] = pd.to_datetime(df["datetime_parsed"])

    df = df[
        (df["date_group"] == DATE)
        & (df["camera_name"] == CAMERA)
    ].copy()

    if df.empty:
        raise RuntimeError(f"No rows for {DATE}, {CAMERA}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    input_shape = model.input_shape
    height, width = int(input_shape[1]), int(input_shape[2])

    frames = (
        df[["frame_path", "datetime_parsed"]]
        .drop_duplicates("frame_path")
        .sort_values("datetime_parsed")
    )

    writer = None
    records = []

    for frame_index, (_, frame_info) in enumerate(
        frames.iterrows(), start=1
    ):
        frame_path = str(frame_info["frame_path"])
        frame = cv2.imread(frame_path)

        if frame is None:
            print("Skipping unreadable frame:", frame_path)
            continue

        rows = df[df["frame_path"] == frame_path]
        batch = []
        valid = []

        for _, row in rows.iterrows():
            x1, y1, x2, y2 = [
                int(row[c]) for c in ("x1", "y1", "x2", "y2")
            ]

            x1 = max(0, min(x1, frame.shape[1] - 1))
            x2 = max(0, min(x2, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            y2 = max(0, min(y2, frame.shape[0]))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (width, height))
            batch.append(crop.astype(np.float32) / 255.0)
            valid.append((row, x1, y1, x2, y2))

        if not batch:
            continue

        probabilities = model.predict(
            np.asarray(batch), verbose=0
        ).reshape(-1)

        correct = 0

        for (row, x1, y1, x2, y2), probability in zip(
            valid, probabilities
        ):
            actual = int(row["occupancy"])
            predicted = int(probability >= THRESHOLD)
            is_correct = predicted == actual
            correct += int(is_correct)

            pred_label = "BUSY" if predicted else "FREE"
            gt_label = "BUSY" if actual else "FREE"

            # Green means prediction matches ground truth.
            # Red means prediction differs from ground truth.
            color = (0, 190, 0) if is_correct else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = (
                f"{row['slot_id']} P:{pred_label} "
                f"GT:{gt_label} {probability:.2f}"
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.38
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )

            tx = max(0, min(x1, frame.shape[1] - tw - 4))
            ty = max(th + 4, y1)

            cv2.rectangle(
                frame,
                (tx, ty - th - 4),
                (tx + tw + 4, ty + baseline),
                color,
                -1,
            )
            cv2.putText(
                frame, label, (tx + 2, ty - 2),
                font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA,
            )

            records.append({
                "date": DATE,
                "camera": CAMERA,
                "timestamp": str(row["datetime_parsed"]),
                "slot_id": row["slot_id"],
                "ground_truth": gt_label,
                "prediction": pred_label,
                "occupied_probability": float(probability),
                "correct": is_correct,
            })

        timestamp = str(frame_info["datetime_parsed"])
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (30, 30, 30), -1)
        cv2.putText(
            frame, f"CNR-EXT | {CAMERA} | {timestamp}",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Correct: {correct}/{len(valid)} | "
            f"Threshold: {THRESHOLD:.2f} | Green=correct Red=wrong",
            (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(VIDEO_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create MP4 writer.")

        writer.write(frame)
        print(f"Processed frame {frame_index}/{len(frames)}")

    if writer is not None:
        writer.release()

    result = pd.DataFrame(records)
    result.to_csv(CSV_PATH, index=False)

    print("\nVideo saved:", VIDEO_PATH.resolve())
    print("CSV saved:", CSV_PATH.resolve())
    print("Slot predictions:", len(result))
    if not result.empty:
        print("Demo accuracy:", f"{result['correct'].mean():.2%}")
        print("\nPrediction counts:")
        print(pd.crosstab(
            result["ground_truth"], result["prediction"]
        ))


if __name__ == "__main__":
    main()
