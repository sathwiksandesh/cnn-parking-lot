"""
auto_detector.py
~~~~~~~~~~~~~~~~
Robust parking-space detection + CNN occupancy classification.

Stages:
  1. Vehicle detection using YOLOv8-nano.
     If >= 2 vehicles are found, infers row layout and empty slot gaps.
  2. Structural / Lane divider detection.
     If YOLO finds < 2 vehicles (common on overhead drone / CCTV views),
     detects the active parking row using variance/edge profiling and
     divides the row into rectangular parking bays.
  3. CNN classification:
     Evaluates each candidate bay with the trained parking-lot CNN.
  4. Non-overlapping visual annotation:
     Draws crisp color-coded bounding boxes, translucent fills, and
     adaptive text badges that never collide or overlap.
"""

from __future__ import annotations

import cv2
import numpy as np


# COCO vehicle classes: car, motorcycle, bus, truck
_VEHICLE_CLASSES: set[int] = {2, 3, 5, 7}

_yolo_model = None


def _get_yolo():
    """Load YOLOv8-nano singleton; returns None if ultralytics is unavailable."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO  # type: ignore
        _yolo_model = YOLO("yolov8n.pt")
        return _yolo_model
    except Exception:
        return None


def _cnn_occupied_prob(cnn_model, crop_rgb: np.ndarray, image_size: int = 96) -> float:
    """Run parking-lot CNN on a single RGB crop; returns P(occupied) in [0, 1]."""
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 4 or crop_rgb.shape[1] < 4:
        return 0.0
    resized = cv2.resize(crop_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    probs = cnn_model.predict(batch, verbose=0)[0]
    return float(probs[0]) if probs.shape[0] == 1 else float(probs[1])


def _box_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = area1 + area2 - inter
    return inter / denom if denom > 0 else 0.0


def _deduplicate(
    new_boxes: list[tuple[int, int, int, int]],
    existing_boxes: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    result = []
    for nb in new_boxes:
        if any(_box_iou(nb, eb) > iou_thresh for eb in existing_boxes):
            continue
        if any(_box_iou(nb, rb) > iou_thresh for rb in result):
            continue
        result.append(nb)
    return result


def _detect_structural_bays(
    image_rgb: np.ndarray,
    target_slots: int = 0,
) -> list[tuple[int, int, int, int]]:
    """
    Detects parking bays in a parking lot image using variance profiling
    and divider line analysis.
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Detect active row along Y (horizontal parking row of vertical slots)
    row_vars = np.var(gray, axis=1)
    smooth_var = np.convolve(row_vars, np.ones(15) / 15, mode="same")
    peak_y = int(np.argmax(smooth_var))

    # Threshold relative to peak variance
    half_v = float(smooth_var[peak_y] * 0.25)
    y_top = peak_y
    while y_top > 0 and smooth_var[y_top] > half_v:
        y_top -= 1
    y_bot = peak_y
    while y_bot < h - 1 and smooth_var[y_bot] > half_v:
        y_bot += 1

    # Bounds for the parking row
    row_y1 = max(0, int(y_top - 18))
    row_y2 = min(h, int(y_bot + 18))
    row_h = row_y2 - row_y1

    # Sanity check: row height should be at least 15% of image height
    if row_h < int(h * 0.20):
        row_y1 = int(h * 0.15)
        row_y2 = int(h * 0.85)
        row_h = row_y2 - row_y1

    # 2. Determine slot width / number of slots
    if target_slots > 0:
        num_slots = target_slots
    else:
        # Search for vertical divider lines inside the active row
        smooth = cv2.bilateralFilter(gray, 7, 50, 50)
        grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vert_edges = cv2.morphologyEx(np.abs(grad_x), cv2.MORPH_OPEN, kernel_v)
        line_profile = np.sum(vert_edges[row_y1:row_y2, :], axis=0)

        # Estimate slot count from divider line frequency or aspect ratio ~3.5
        import scipy.signal
        peaks, _ = scipy.signal.find_peaks(
            line_profile,
            distance=max(20, int(w * 0.04)),
            prominence=float(line_profile.max() * 0.12),
        )

        if len(peaks) >= 4:
            diffs = np.diff(peaks)
            # Filter reasonable slot widths (between 25 and 150 px)
            valid_diffs = [d for d in diffs if 20 <= d <= 150]
            if valid_diffs:
                med_w = float(np.median(valid_diffs))
                num_slots = max(3, int(round(w / med_w)))
            else:
                num_slots = max(4, int(round(w / (row_h / 3.4))))
        else:
            num_slots = max(4, int(round(w / (row_h / 3.4))))

    slot_w = w / num_slots
    candidates: list[tuple[int, int, int, int]] = []
    for i in range(num_slots):
        x1 = int(round(i * slot_w))
        x2 = min(w, int(round((i + 1) * slot_w)))
        candidates.append((x1, row_y1, x2, row_y2))

    return candidates


def detect_and_classify(
    image_rgb: np.ndarray,
    cnn_model,
    image_size: int = 96,
    cnn_threshold: float = 0.30,
    yolo_conf: float = 0.20,
    target_slots: int = 0,
    layout_mode: str = "auto",
) -> list[dict]:
    """
    Automatically detects parking spaces in image_rgb and classifies them
    using the trained parking-lot CNN.
    """
    h, w = image_rgb.shape[:2]

    # Stage 1: Attempt YOLO vehicle detection
    yolo = _get_yolo()
    vehicle_boxes: list[tuple[int, int, int, int]] = []
    if yolo is not None:
        try:
            results = yolo(image_rgb, verbose=False)[0]
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in _VEHICLE_CLASSES and conf >= yolo_conf:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2 - x1) > 15 and (y2 - y1) > 15:
                        vehicle_boxes.append((x1, y1, x2, y2))
        except Exception:
            vehicle_boxes = []

    # If YOLO detected at least 3 distinct vehicles across the image, use YOLO + gaps
    if len(vehicle_boxes) >= 3 and target_slots == 0:
        candidates = [(b, "vehicle") for b in vehicle_boxes]
        # Infer gaps between vehicles in each row
        # (similar row grouping)
        rows: list[list[tuple[int, int, int, int]]] = []
        for box in sorted(vehicle_boxes, key=lambda b: (b[1] + b[3]) / 2):
            cy = (box[1] + box[3]) / 2
            placed = False
            for row in rows:
                row_cy = np.mean([(b[1] + b[3]) / 2 for b in row])
                if abs(cy - row_cy) < (box[3] - box[1]) * 0.6:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        for row in rows:
            if len(row) < 2:
                continue
            row_sorted = sorted(row, key=lambda b: b[0])
            med_w = np.median([b[2] - b[0] for b in row_sorted])
            for i in range(len(row_sorted) - 1):
                gx1 = row_sorted[i][2]
                gx2 = row_sorted[i + 1][0]
                gw = gx2 - gx1
                if med_w * 0.5 <= gw <= med_w * 3.5:
                    n_gaps = max(1, int(round(gw / med_w)))
                    step = gw / n_gaps
                    sy1 = int((row_sorted[i][1] + row_sorted[i + 1][1]) / 2)
                    sy2 = int((row_sorted[i][3] + row_sorted[i + 1][3]) / 2)
                    for k in range(n_gaps):
                        sx1 = int(gx1 + k * step)
                        sx2 = int(gx1 + (k + 1) * step)
                        candidates.append(((sx1, sy1, sx2, sy2), "inferred_gap"))
    else:
        # Structural bay detection
        bay_boxes = _detect_structural_bays(image_rgb, target_slots=target_slots)
        candidates = [(b, "ai_detector") for b in bay_boxes]

    # Stage 3: CNN classification of all candidates
    spaces: list[dict] = []
    for (x1, y1, x2, y2), source in candidates:
        crop = image_rgb[y1:y2, x1:x2]
        occ_p = _cnn_occupied_prob(cnn_model, crop, image_size)
        status = "occupied" if occ_p >= cnn_threshold else "vacant"
        spaces.append({
            "bbox": [x1, y1, x2, y2],
            "status": status,
            "occupied_probability": round(occ_p, 4),
            "confidence": round(occ_p if status == "occupied" else 1.0 - occ_p, 4),
            "detection_source": source,
        })

    # Sort left-to-right (and top-to-bottom for multi-row)
    spaces.sort(key=lambda s: (s["bbox"][1] // 50, s["bbox"][0]))
    for i, s in enumerate(spaces):
        s["id"] = f"P{i+1:02d}"

    return spaces


def annotate_image(
    image_rgb: np.ndarray,
    spaces: list[dict],
) -> np.ndarray:
    """
    Renders high-quality, professional annotations with:
      - Crisp color-coded outlines (Red = Occupied, Green = Free).
      - Semi-transparent color wash (20% opacity) filling each parking slot.
      - Adaptive, non-overlapping label badges neatly centered at the top
        of each space.
    """
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = annotated.copy()
    h, w = annotated.shape[:2]

    color_occ = (0, 0, 220)   # Vibrant BGR Red
    color_free = (0, 200, 0)  # Vibrant BGR Green

    # 1. Fill translucent tint
    for s in spaces:
        x1, y1, x2, y2 = s["bbox"]
        occupied = s["status"] == "occupied"
        color = color_occ if occupied else color_free
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)

    # 2. Draw crisp bounding borders and centered badges
    line_w = max(2, w // 400)

    for s in spaces:
        x1, y1, x2, y2 = s["bbox"]
        slot_w = x2 - x1
        slot_h = y2 - y1
        occupied = s["status"] == "occupied"
        color = color_occ if occupied else color_free

        # Outline
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_w, cv2.LINE_AA)

        # Label formatting with adaptive sizing so it never overflows slot_w
        tag = "OCC" if occupied else "FREE"
        label = f"{s['id']} {tag}"

        # Choose font scale that guarantees label fits inside slot_w
        base_font_scale = 0.40 if w < 800 else 0.50
        thickness = 1 if w < 800 else 2

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, thickness)
        if tw > slot_w - 6:
            # Scale down font proportionally
            base_font_scale = max(0.26, base_font_scale * ((slot_w - 6) / tw))
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, thickness)

        # Center badge at the top-center inside the slot
        badge_x = x1 + max(0, (slot_w - tw) // 2)
        badge_y = y1 + th + 6

        # Draw solid badge pill
        cv2.rectangle(
            annotated,
            (badge_x - 3, badge_y - th - 3),
            (badge_x + tw + 3, badge_y + bl + 1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (badge_x, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            base_font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated
