
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.detection.occupancy_detector import OccupancyDetector
from src.detection.slot_config import load_slots
from src.utils.config_loader import load_config, project_path


COLORS = {
    "vacant": (0, 200, 0),       # Green
    "occupied": (0, 0, 255),     # Red
    "uncertain": (0, 191, 255),  # Amber
}



def draw_results(frame, results):
    output = frame.copy()
    height, width = output.shape[:2]

    vacant = sum(r["status"] == "vacant" for r in results)
    occupied = sum(r["status"] == "occupied" for r in results)
    uncertain = sum(r["status"] == "uncertain" for r in results)

    header_height = min(48, max(32, height // 10))

    cv2.rectangle(
        output, (0, 0), (width, header_height), (25, 25, 25), -1
    )

    summary = (
        f"Slots: {len(results)} | Free: {vacant} | "
        f"Occupied: {occupied} | Unknown: {uncertain}"
    )

    cv2.putText(
        output,
        summary,
        (6, min(header_height - 8, 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 if width < 600 else 0.65,
        (255, 255, 255),
        1 if width < 600 else 2,
        cv2.LINE_AA,
    )

    for result in results:
        polygon = result["polygon"]
        status = result["status"]
        confidence = result["confidence"]
        slot_id = result["id"]

        color = COLORS[status]

        cv2.polylines(
            output, [polygon], True, color, 2, cv2.LINE_AA
        )

        x, y, box_width, box_height = cv2.boundingRect(polygon)

        label = f"{slot_id}: {status.upper()} {confidence:.0%}"
        font_scale = 0.42 if width < 600 else 0.55
        thickness = 1 if width < 600 else 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, thickness
        )

        # Prefer a label above the polygon.
        label_x = max(0, min(x, width - text_width - 6))
        label_y = y - 5

        if label_y - text_height - baseline < header_height:
            label_y = min(
                height - baseline - 3,
                y + text_height + baseline + 5,
            )

        label_y = max(text_height + baseline, label_y)

        cv2.rectangle(
            output,
            (label_x, label_y - text_height - baseline),
            (
                min(width - 1, label_x + text_width + 6),
                min(height - 1, label_y + baseline),
            ),
            color,
            -1,
        )

        cv2.putText(
            output,
            label,
            (label_x + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output


def open_source(source: str):
    # Integer strings select camera devices, e.g. "0".
    if source.isdigit():
        return cv2.VideoCapture(int(source))

    return cv2.VideoCapture(str(project_path(source)))


def process_video(
    source: str,
    slots_path: Path,
    model_path: Path,
    output_path: Path,
    summary_path: Path,
    image_size: int = 128,
    confidence_threshold: float = 0.60,
    display: bool = False,
    process_every_n_frames: int = 1,
):
    slots = load_slots(slots_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    capture = open_source(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open input source: {source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Unable to determine video dimensions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / process_every_n_frames,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    detector = OccupancyDetector(
        model_path=str(model_path),
        image_size=image_size,
        confidence_threshold=confidence_threshold,
    )

    frame_index = 0
    processed_frames = 0
    start_time = time.perf_counter()
    latest_results = []

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            frame_index += 1

            if (frame_index - 1) % process_every_n_frames != 0:
                continue

            latest_results = detector.predict_slots(frame, slots)
            annotated = draw_results(frame, latest_results)

            writer.write(annotated)
            processed_frames += 1

            if display:
                cv2.imshow("CNN Parking Occupancy", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        capture.release()
        writer.release()

        if display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time

    vacant = sum(r["status"] == "vacant" for r in latest_results)
    occupied = sum(r["status"] == "occupied" for r in latest_results)
    uncertain = sum(r["status"] == "uncertain" for r in latest_results)

    summary = {
        "input_source": source,
        "model_path": str(model_path),
        "slots_config": str(slots_path),
        "input_resolution": [width, height],
        "input_fps": fps,
        "frames_read": frame_index,
        "frames_processed": processed_frames,
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(
            processed_frames / elapsed, 3
        ) if elapsed > 0 else 0.0,
        "total_slots": len(slots),
        "last_processed_frame": {
            "vacant": vacant,
            "occupied": occupied,
            "uncertain": uncertain,
        },
        "slot_results": [
            {
                "id": r["id"],
                "status": r["status"],
                "confidence": round(r["confidence"], 5),
                "probabilities": r["probabilities"],
            }
            for r in latest_results
        ],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== VIDEO PROCESSING COMPLETE ===")
    print("Frames read:", frame_index)
    print("Frames processed:", processed_frames)
    print("Slots configured:", len(slots))
    print("Last-frame vacant:", vacant)
    print("Last-frame occupied:", occupied)
    print("Last-frame uncertain:", uncertain)
    print("Processing FPS:", summary["processing_fps"])
    print("Annotated video:", output_path)
    print("Summary JSON:", summary_path)


def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        description="CNN parking-lot occupancy video processor"
    )
    parser.add_argument(
        "--source",
        default="dataset/raw/videos/parking_lot.mp4",
        help="Video path relative to project root, or camera index",
    )
    parser.add_argument(
        "--slots",
        default="configs/parking_slots.json",
    )
    parser.add_argument(
        "--model",
        default=config["paths"]["model"],
    )
    parser.add_argument(
        "--output",
        default="outputs/processed_videos/parking_output.mp4",
    )
    parser.add_argument(
        "--summary",
        default="outputs/reports/parking_summary.json",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=float(config["model"]["threshold"]),
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Process every Nth frame",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live preview; press Q to stop",
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")

    if args.every < 1:
        parser.error("--every must be at least 1")

    process_video(
        source=args.source,
        slots_path=project_path(args.slots),
        model_path=project_path(args.model),
        output_path=project_path(args.output),
        summary_path=project_path(args.summary),
        image_size=int(config["data"]["image_width"]),
        confidence_threshold=args.confidence,
        display=args.display,
        process_every_n_frames=args.every,
    )


if __name__ == "__main__":
    main()
