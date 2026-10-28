"""
Demo: run the trained CNN on CNR-EXT frames of one camera, using the dataset's
own slot boxes (no guessed coordinates). Writes an annotated MP4, a few PNG
stills for the report, and a slots JSON compatible with slot_config.load_slots.

  python scripts\demo_cnr_camera.py --camera 1 --date 2015-12-03 --limit 120
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.occupancy_detector import OccupancyDetector  # noqa: E402

MANIFEST = ROOT / "dataset" / "cnrpark_ext" / "metadata" / "cnr_ext_frame_manifest.csv"
MODEL = ROOT / "models" / "cnr_ext_cnn_fast.keras"
OUT_DIR = ROOT / "outputs" / "cnr_ext_fast_demo"
THRESHOLD = 0.30
IMAGE_SIZE = 96
TEST_DATES = {"2015-12-03", "2015-12-10", "2016-01-09"}


def draw(frame, results, gt_frame, title):
    out = frame.copy()
    free = occ = agree = n_gt = 0

    for r in results:
        busy = r["status"] == "occupied"
        occ += busy
        free += not busy
        color = (0, 0, 255) if busy else (0, 200, 0)  # red = occupied, green = free
        sid = int(r["id"][1:])

        if sid in gt_frame:
            n_gt += 1
            if int(busy) == gt_frame[sid]:
                agree += 1
            else:  # yellow halo = disagrees with the dataset label
                cv2.polylines(out, [r["polygon"]], True, (0, 255, 255), 5, cv2.LINE_AA)

        cv2.polylines(out, [r["polygon"]], True, color, 2, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (out.shape[1], 58), (25, 25, 25), -1)
    cv2.putText(out, f"{title} | Free: {free} | Occupied: {occ}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if n_gt:
        cv2.putText(out, f"Matches dataset labels: {agree}/{n_gt}  (yellow = mismatch)",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out, agree, n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--date", default="2015-12-03")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--stills", type=int, default=3)
    a = ap.parse_args()

    m = pd.read_csv(MANIFEST)
    m["date_group"] = m["date_group"].astype(str)
    cam = m[m["camera_num"] == a.camera]
    day = cam[cam["date_group"] == a.date]

    if day.empty:
        print("No frames for that camera/date. Available held-out test combinations:")
        avail = (m[m["date_group"].isin(TEST_DATES)]
                 .groupby(["date_group", "camera_num"])["frame_path"].nunique())
        print(avail.to_string())
        sys.exit(1)

    # Slot boxes straight from the dataset annotations (fixed per camera).
    changing = cam.groupby("slot_id")[["x1", "y1", "x2", "y2"]].nunique().max(axis=1)
    if (changing > 1).any():
        print(f"WARNING: {(changing > 1).sum()} slots have more than one box; using the first.")
    boxes = cam.drop_duplicates("slot_id")[["slot_id", "x1", "y1", "x2", "y2"]]
    slots = [{
        "id": f"S{int(r.slot_id)}",
        "points": [[int(r.x1), int(r.y1)], [int(r.x2), int(r.y1)],
                   [int(r.x2), int(r.y2)], [int(r.x1), int(r.y2)]],
    } for r in boxes.itertuples(index=False)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slots_json = ROOT / "configs" / f"parking_slots_cnr_camera{a.camera}.json"
    slots_json.write_text(json.dumps(slots, indent=2), encoding="utf-8")

    frames = day.sort_values("datetime_parsed")["frame_path"].drop_duplicates().tolist()
    frames = frames[::max(1, len(frames) // a.limit)][:a.limit]
    gt = {fp: dict(zip(g["slot_id"].astype(int), g["occupancy"].astype(int)))
          for fp, g in day.groupby("frame_path")}

    held_out = a.date in TEST_DATES
    print(f"Camera {a.camera} | {a.date} | slots {len(slots)} | frames {len(frames)} | "
          f"{'held-out test date' if held_out else 'NOT a held-out date'}")

    det = OccupancyDetector(str(MODEL), image_size=IMAGE_SIZE, confidence_threshold=THRESHOLD)

    video_path = OUT_DIR / f"camera{a.camera}_{a.date}.mp4"
    writer, size = None, None
    still_idx = set(np.linspace(0, len(frames) - 1, a.stills, dtype=int)) if a.stills else set()
    tot_agree = tot_gt = 0

    for i, fp in enumerate(frames):
        frame = cv2.imread(fp)
        if frame is None:
            print("Skipping unreadable frame:", fp)
            continue
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, size)
        elif (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)

        results = det.predict_slots(frame, slots)
        title = f"Camera {a.camera} {Path(fp).stem}"
        annotated, agree, n_gt = draw(frame, results, gt.get(fp, {}), title)
        tot_agree += agree
        tot_gt += n_gt
        writer.write(annotated)
        if i in still_idx:
            cv2.imwrite(str(OUT_DIR / f"still_camera{a.camera}_{a.date}_{i:03d}.png"), annotated)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(frames)} frames")

    if writer:
        writer.release()

    if tot_gt:
        print(f"\nAgreement with dataset labels: {tot_agree}/{tot_gt} = {tot_agree / tot_gt:.2%}")
        print("(" + ("held-out test date" if held_out else "train/val date, NOT a fair test") + ")")
    print("Video :", video_path)
    print("Stills:", OUT_DIR)
    print("Slots :", slots_json)


if __name__ == "__main__":
    main()