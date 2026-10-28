"""
Qualitative demo on an arbitrary photo, using hand-drawn slots.

  python -m src.detection.slot_annotator --source dataset\\uploads\\photo.jpg --output configs\\parking_slots_myphoto.json
  python scripts\\predict_image.py --image dataset\\uploads\\photo.jpg --slots configs\\parking_slots_myphoto.json
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.occupancy_detector import OccupancyDetector  # noqa: E402
from src.detection.slot_config import load_slots               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--model", default=str(ROOT / "models" / "cnr_ext_cnn_fast.keras"))
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "cnr_ext_fast_demo" / "custom_photo_result.png"))
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"Cannot read image: {a.image}")

    slots = load_slots(a.slots)
    det = OccupancyDetector(a.model, image_size=96, confidence_threshold=a.threshold)
    results = det.predict_slots(img, slots)

    if len(results) < len(slots):
        print(f"WARNING: {len(slots) - len(results)} slots were outside the image and skipped.")

    s = max(1.0, img.shape[1] / 1000)          # scale drawing to the image size
    th, fs = max(2, int(2 * s)), 0.55 * s
    out = img.copy()
    free = occ = 0

    for r in results:
        busy = r["status"] == "occupied"
        occ += busy
        free += not busy
        color = (0, 0, 255) if busy else (0, 200, 0)    # red = occupied, green = free
        cv2.polylines(out, [r["polygon"]], True, color, th, cv2.LINE_AA)
        x, y, _, _ = cv2.boundingRect(r["polygon"])
        label = f"{r['id']} {r['probabilities']['occupied']:.0%}"
        cv2.putText(out, label, (x, max(int(18 * s), y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)

    bar = int(40 * s)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar), (25, 25, 25), -1)
    cv2.putText(out, f"Free: {free} | Occupied: {occ} | label = P(occupied)",
                (int(8 * s), int(27 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s,
                (255, 255, 255), th, cv2.LINE_AA)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, out)
    print(f"Free {free} | Occupied {occ} | saved {a.out}")
    print("NOTE: qualitative demo only. This photo was not evaluated, and the model was trained on CNR-EXT.")


if __name__ == "__main__":
    main()