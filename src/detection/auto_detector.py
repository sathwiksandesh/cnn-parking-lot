"""
auto_detector.py  (v2 — robust multi-stage occupancy classifier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Root-cause fixes applied in this version
-----------------------------------------
BUG-1  YOLO only searched COCO vehicle classes {2,3,5,7} (car/motorcycle/bus/truck).
       Overhead parking-lot images are an out-of-distribution viewpoint for YOLOv8-nano:
       top-down cars look nothing like the side-profile "car" training images, so the
       model consistently misfires them as "cell phone" (class 67) and the filter drops
       every detection.  Fix: accept ANY high-confidence YOLO detection whose bounding
       box matches the expected aspect ratio of a top-down parked car (height > width,
       area within vehicle-size bounds), regardless of class label.

BUG-2  Structural divider detection used scipy.signal.find_peaks on the vertical-edge
       profile, which found *both* the painted parking lines AND the car-side edges,
       halving the real slot width and yielding ~16 slots instead of 7.  Fix: after
       finding candidate dividers, merge those that are closer than min_slot_px together
       so we get true slot boundaries rather than spurious internal car edges.

BUG-3  When YOLO detected ≥3 vehicles, the code used each vehicle bbox as a
       candidate *space*.  But YOLO was only finding ~6 out of 7 vehicles (the leftmost
       red car was missed at the default conf threshold).  The gap-inference logic then
       produced wrong FREE slots.  Fix: derive slot geometry *from* vehicle boxes (each
       vehicle == one slot), and then detect gaps between consecutive vehicles as
       additional FREE candidates — rather than passing raw YOLO boxes as space rects.

BUG-4  CNN classification alone (P(occupied) ≥ 0.30) was the only occupancy signal.
       When a slot crop happens to be mostly asphalt (poorly cropped), the CNN
       correctly rates it as "vacant" even though a car is there.  Fix: the primary
       occupancy signal is now the YOLO spatial-overlap check.  CNN provides a
       *secondary* boost when YOLO evidence is absent.

BUG-5  annotate_image returned BGR but callers assumed RGB (or vice-versa) causing
       colour swaps on some display paths.  Marked clearly with a docstring.

Pipeline (five stages matching the task spec):
  A. Slot geometry detection  — YOLO-vehicle-guided or structural divider analysis
  B. Vehicle detection        — YOLO with top-down aspect-ratio filtering
  C. Spatial matching         — IoU, intersection-over-slot-area (IoSA)
  D. Occupancy classification — OCC / FREE / UNKNOWN with calibrated thresholds
  E. Structured result dicts  — slot_id, occupancy, confidence, overlap_ratio, reason
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Aspect-ratio guard for top-down vehicle silhouettes:
# height > width because cars are longer than wide when viewed overhead.
_TOPDOWN_MIN_ASPECT = 1.4    # height / width minimum
_TOPDOWN_MAX_ASPECT = 5.0    # guard against very thin false positives
_TOPDOWN_MIN_AREA = 4_000    # px²  ─ ignore tiny noise boxes
_TOPDOWN_MAX_AREA = 100_000  # px²  ─ ignore full-frame misdetections

# Overlap thresholds
_OCC_IOSA_THRESH = 0.25      # IoSA (intersection / slot_area) ≥ this → OCC
_UNCERTAIN_IOSA_THRESH = 0.08  # IoSA between this and OCC_IOSA → UNKNOWN

# CNN secondary threshold (only used when no vehicle-bbox overlap info exists)
_CNN_OCC_THRESH = 0.45
_CNN_UNKNOWN_THRESH = 0.30

# Minimum valid slot size (px)
_MIN_SLOT_PX = 20

_yolo_model = None


# ---------------------------------------------------------------------------
# YOLO singleton
# ---------------------------------------------------------------------------

def _get_yolo():
    """Load YOLOv8-nano singleton; returns None if ultralytics is unavailable."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO  # type: ignore
        _yolo_model = YOLO("yolov8n.pt")
    except Exception:
        _yolo_model = None
    return _yolo_model


# ---------------------------------------------------------------------------
# BUG-1 FIX: Top-down vehicle detection (aspect-ratio + area filter)
# ---------------------------------------------------------------------------

def _detect_vehicles_topdown(
    image_rgb: np.ndarray,
    conf_thresh: float = 0.10,
) -> list[tuple[int, int, int, int, float]]:
    """
    Detect parked vehicles in an overhead image.

    Because top-down car silhouettes look nothing like the side-profile cars
    in COCO training data, YOLOv8-nano often assigns them to unrelated classes
    (e.g. "cell phone").  We therefore accept *any* detection whose bounding-box
    geometry matches the expected overhead-car profile:
      · height > width  (car is longer than wide from above)
      · aspect ratio within [_TOPDOWN_MIN_ASPECT, _TOPDOWN_MAX_ASPECT]
      · bounding-box area within [_TOPDOWN_MIN_AREA, _TOPDOWN_MAX_AREA] px²

    Returns list of (x1, y1, x2, y2, confidence) tuples.
    """
    yolo = _get_yolo()
    boxes: list[tuple[int, int, int, int, float]] = []

    if yolo is None:
        return boxes

    h, w = image_rgb.shape[:2]
    try:
        results = yolo(image_rgb, verbose=False, imgsz=1280, conf=conf_thresh)[0]
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < _MIN_SLOT_PX or bh < _MIN_SLOT_PX:
                continue
            area = bw * bh
            aspect = bh / bw
            if (
                _TOPDOWN_MIN_ASPECT <= aspect <= _TOPDOWN_MAX_ASPECT
                and _TOPDOWN_MIN_AREA <= area <= _TOPDOWN_MAX_AREA
            ):
                boxes.append((x1, y1, x2, y2, conf))
    except Exception:
        pass

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])
    return boxes


# ---------------------------------------------------------------------------
# BUG-2 FIX: Structural divider detection with divider merging
# ---------------------------------------------------------------------------

def _detect_structural_slots(
    image_rgb: np.ndarray,
    target_slots: int = 0,
) -> list[tuple[int, int, int, int]]:
    """
    Detect parking-slot bounding boxes via variance profiling and vertical
    edge (divider) detection.

    FIX: After finding candidate divider x-coordinates, we merge those that
    are closer than a minimum-slot-width threshold.  This removes the spurious
    lines that the Sobel filter finds at car edges, halving the real slot
    count (BUG-2).
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Locate the active parking row by maximum variance band
    row_vars = np.var(gray, axis=1)
    smooth_var = np.convolve(row_vars, np.ones(15) / 15, mode="same")
    peak_y = int(np.argmax(smooth_var))

    half_v = float(smooth_var[peak_y] * 0.25)
    y_top = peak_y
    while y_top > 0 and smooth_var[y_top] > half_v:
        y_top -= 1
    y_bot = peak_y
    while y_bot < h - 1 and smooth_var[y_bot] > half_v:
        y_bot += 1

    row_y1 = max(0, int(y_top - 18))
    row_y2 = min(h, int(y_bot + 18))
    row_h = row_y2 - row_y1

    if row_h < int(h * 0.20):
        row_y1 = int(h * 0.15)
        row_y2 = int(h * 0.85)
        row_h = row_y2 - row_y1

    # 2. Detect vertical divider lines
    if target_slots > 0:
        num_slots = target_slots
        slot_w = w / num_slots
        candidates = []
        for i in range(num_slots):
            x1 = int(round(i * slot_w))
            x2 = min(w, int(round((i + 1) * slot_w)))
            candidates.append((x1, row_y1, x2, row_y2))
        return candidates

    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vert_edges = cv2.morphologyEx(np.abs(grad_x), cv2.MORPH_OPEN, kernel_v)
    line_profile = np.sum(vert_edges[row_y1:row_y2, :], axis=0)

    # Estimated minimum slot width: assume aspect ratio ~3.5 (height/width)
    min_slot_px = max(30, int(row_h / 4.0))

    try:
        import scipy.signal
        peaks, _ = scipy.signal.find_peaks(
            line_profile,
            distance=max(10, int(w * 0.02)),   # allow tighter search first
            prominence=float(line_profile.max() * 0.08),
        )
    except Exception:
        peaks = np.array([], dtype=int)

    # --- BUG-2 FIX: merge dividers closer than min_slot_px ---
    if len(peaks) > 0:
        merged: list[int] = [int(peaks[0])]
        for px in peaks[1:]:
            if px - merged[-1] >= min_slot_px:
                merged.append(int(px))
            # else: skip — it is a spurious edge inside a car
        # Build slot boundaries from merged dividers
        dividers = [0] + merged + [w]
        candidates = []
        for i in range(len(dividers) - 1):
            x1, x2 = dividers[i], dividers[i + 1]
            if x2 - x1 >= min_slot_px:
                candidates.append((x1, row_y1, x2, row_y2))
    else:
        # Fallback: use aspect-ratio estimate
        num_slots = max(4, int(round(w / (row_h / 3.4))))
        slot_w = w / num_slots
        candidates = []
        for i in range(num_slots):
            x1 = int(round(i * slot_w))
            x2 = min(w, int(round((i + 1) * slot_w)))
            candidates.append((x1, row_y1, x2, row_y2))

    return candidates


# ---------------------------------------------------------------------------
# Overlap helpers (Step C — spatial matching)
# ---------------------------------------------------------------------------

def _intersection_area(
    b1: tuple[int, int, int, int],
    b2: tuple[int, int, int, int],
) -> float:
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    w_ = max(0, xB - xA)
    h_ = max(0, yB - yA)
    return float(w_ * h_)


def _slot_area(slot_box: tuple[int, int, int, int]) -> float:
    return float(
        max(0, slot_box[2] - slot_box[0]) * max(0, slot_box[3] - slot_box[1])
    )


def _iosa(
    slot_box: tuple[int, int, int, int],
    vehicle_box: tuple[int, int, int, int],
) -> float:
    """Intersection-over-Slot-Area: what fraction of the slot is covered by the vehicle?"""
    sa = _slot_area(slot_box)
    if sa == 0:
        return 0.0
    return _intersection_area(slot_box, vehicle_box) / sa


def _best_vehicle_overlap(
    slot_box: tuple[int, int, int, int],
    vehicles: list[tuple[int, int, int, int, float]],
) -> tuple[float, float]:
    """
    Return (best_iosa, best_vehicle_conf) for the vehicle that overlaps slot_box
    the most.
    """
    best_iosa = 0.0
    best_conf = 0.0
    for vx1, vy1, vx2, vy2, vconf in vehicles:
        iosa = _iosa(slot_box, (vx1, vy1, vx2, vy2))
        if iosa > best_iosa:
            best_iosa = iosa
            best_conf = vconf
    return best_iosa, best_conf


# ---------------------------------------------------------------------------
# CNN helper
# ---------------------------------------------------------------------------

def _cnn_prob(cnn_model, crop_rgb: np.ndarray, image_size: int = 96) -> float:
    """Run CNN on a single RGB crop; returns P(occupied) in [0, 1]."""
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 4 or crop_rgb.shape[1] < 4:
        return 0.0
    resized = cv2.resize(crop_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    probs = cnn_model.predict(batch, verbose=0)[0]
    return float(probs[0]) if probs.shape[0] == 1 else float(probs[1])


# ---------------------------------------------------------------------------
# BUG-3 / BUG-4 FIX: Main detect-and-classify pipeline
# ---------------------------------------------------------------------------

def _slots_from_vehicle_boxes(
    vehicles: list[tuple[int, int, int, int, float]],
    image_h: int,
) -> list[tuple[int, int, int, int]]:
    """
    BUG-3 FIX: Derive slot bounding boxes FROM vehicle detections (one slot
    per vehicle), then fill gaps between consecutive vehicles as FREE candidates.

    Old code used each vehicle box AS the candidate space, which caused some
    FREE slots to be missed and others to be wrongly classified.
    """
    if not vehicles:
        return []

    # Group vehicles by row
    sorted_v = sorted(vehicles, key=lambda b: b[0])  # left to right
    row_y1 = min(b[1] for b in sorted_v)
    row_y2 = max(b[3] for b in sorted_v)
    # Expand row bounds slightly
    margin = int((row_y2 - row_y1) * 0.05)
    row_y1 = max(0, row_y1 - margin)

    # Build slots: one per vehicle + inferred gaps between consecutive vehicles
    slots: list[tuple[int, int, int, int]] = []
    med_w = int(np.median([b[2] - b[0] for b in sorted_v]))
    med_h = int(np.median([b[3] - b[1] for b in sorted_v]))

    for vb in sorted_v:
        # Use each vehicle bbox as a slot (expanded slightly to cover full bay)
        sx1 = max(0, vb[0] - int(med_w * 0.05))
        sx2 = vb[2] + int(med_w * 0.05)
        slots.append((sx1, row_y1, sx2, row_y1 + med_h + margin * 2))

    # Add gap slots between consecutive vehicles
    for i in range(len(sorted_v) - 1):
        gap_x1 = sorted_v[i][2]
        gap_x2 = sorted_v[i + 1][0]
        gap_w = gap_x2 - gap_x1
        if gap_w >= int(med_w * 0.4):
            n = max(1, int(round(gap_w / med_w)))
            step = gap_w / n
            for k in range(n):
                gx1 = int(gap_x1 + k * step)
                gx2 = int(gap_x1 + (k + 1) * step)
                slots.append((gx1, row_y1, gx2, row_y1 + med_h + margin * 2))

    # Sort left-to-right
    slots.sort(key=lambda s: s[0])
    return slots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_and_classify(
    image_rgb: np.ndarray,
    cnn_model,
    image_size: int = 96,
    cnn_threshold: float = 0.30,
    yolo_conf: float = 0.10,
    target_slots: int = 0,
    layout_mode: str = "auto",
    occ_iosa: float = _OCC_IOSA_THRESH,
    unknown_iosa: float = _UNCERTAIN_IOSA_THRESH,
) -> list[dict]:
    """
    Detects parking spaces and classifies occupancy.

    Returns a list of dicts with keys:
        id, slot_id, occupancy, confidence, vehicle_detected,
        overlap_ratio, reason, bbox, status, occupied_probability,
        detection_source
    """
    h, w = image_rgb.shape[:2]

    # --- Stage A + B: Detect vehicles (top-down aware) -----------------
    vehicles = _detect_vehicles_topdown(image_rgb, conf_thresh=yolo_conf)

    # --- Stage A: Slot geometry -----------------------------------------
    if len(vehicles) >= 3 and target_slots == 0:
        # BUG-3 FIX: derive slots FROM vehicle layout
        slot_boxes = _slots_from_vehicle_boxes(vehicles, h)
        slot_source = "yolo_vehicle_guided"
    else:
        # BUG-2 FIX: structural detection with divider-merging
        slot_boxes = _detect_structural_slots(image_rgb, target_slots=target_slots)
        slot_source = "structural"

    if not slot_boxes:
        return []

    # --- Stage C + D: Match vehicles → slots, classify ------------------
    spaces: list[dict] = []

    for sx1, sy1, sx2, sy2 in slot_boxes:
        # Guard against degenerate boxes
        if sx2 - sx1 < _MIN_SLOT_PX or sy2 - sy1 < _MIN_SLOT_PX:
            continue
        slot_box = (sx1, sy1, sx2, sy2)

        # Spatial match
        best_iosa, best_vconf = _best_vehicle_overlap(slot_box, vehicles)
        vehicle_detected = best_iosa > 0.01

        # CNN secondary signal on the slot crop
        crop = image_rgb[sy1:sy2, sx1:sx2]
        cnn_p = _cnn_prob(cnn_model, crop, image_size)

        # --- BUG-4 FIX: primary signal = YOLO spatial overlap -----------
        if best_iosa >= occ_iosa:
            occupancy = "OCC"
            combined_conf = round(min(1.0, best_iosa * 1.5) * 0.7 + cnn_p * 0.3, 4)
            reason = (
                f"Vehicle bbox overlaps slot by {best_iosa:.0%} of slot area "
                f"(IoSA={best_iosa:.3f}, yolo_conf={best_vconf:.2f})"
            )
        elif best_iosa >= unknown_iosa:
            # Marginal overlap — use CNN to decide
            if cnn_p >= _CNN_OCC_THRESH:
                occupancy = "OCC"
                reason = (
                    f"Marginal vehicle overlap (IoSA={best_iosa:.3f}) + "
                    f"CNN confirms occupied (p={cnn_p:.3f})"
                )
            else:
                occupancy = "UNKNOWN"
                reason = (
                    f"Marginal vehicle overlap (IoSA={best_iosa:.3f}), "
                    f"CNN uncertain (p={cnn_p:.3f})"
                )
            combined_conf = round(max(best_iosa, cnn_p), 4)
        else:
            # No meaningful vehicle overlap → rely on CNN
            if cnn_p >= _CNN_OCC_THRESH:
                occupancy = "OCC"
                reason = f"No spatial vehicle overlap but CNN says occupied (p={cnn_p:.3f})"
            elif cnn_p >= _CNN_UNKNOWN_THRESH:
                occupancy = "UNKNOWN"
                reason = f"No spatial vehicle overlap, CNN uncertain (p={cnn_p:.3f})"
            else:
                occupancy = "FREE"
                reason = (
                    f"No vehicle overlap (IoSA={best_iosa:.3f}), "
                    f"CNN confirms vacant (p={cnn_p:.3f})"
                )
            combined_conf = round(1.0 - cnn_p if occupancy == "FREE" else cnn_p, 4)

        # Map to legacy "status" field expected by annotate_image / API
        status = "occupied" if occupancy == "OCC" else "vacant"

        spaces.append({
            "bbox": [sx1, sy1, sx2, sy2],
            "occupancy": occupancy,
            "status": status,
            "confidence": combined_conf,
            "occupied_probability": round(cnn_p, 4),
            "vehicle_detected": vehicle_detected,
            "overlap_ratio": round(best_iosa, 4),
            "reason": reason,
            "detection_source": slot_source,
        })

    # --- Stage E: Sort and assign IDs -----------------------------------
    spaces.sort(key=lambda s: (s["bbox"][1] // 60, s["bbox"][0]))
    for i, s in enumerate(spaces):
        s["id"] = f"P{i+1:02d}"
        s["slot_id"] = s["id"]

    return spaces


# ---------------------------------------------------------------------------
# BUG-5 noted: annotate_image returns a BGR ndarray (OpenCV convention)
# ---------------------------------------------------------------------------

def annotate_image(
    image_rgb: np.ndarray,
    spaces: list[dict],
) -> np.ndarray:
    """
    Renders occupancy annotations on the image.

    Color scheme:
        Green  (#00C800) — FREE
        Red    (#DC0000) — OCC
        Yellow (#E0A000) — UNKNOWN

    Returns:  BGR ndarray (OpenCV convention).
    """
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = annotated.copy()
    h, w = annotated.shape[:2]

    # BGR colors
    color_map = {
        "OCC":     (0, 0, 220),    # red
        "FREE":    (0, 200, 0),    # green
        "UNKNOWN": (0, 160, 224),  # orange-ish
    }
    # Legacy "status" field fallback
    legacy_map = {"occupied": "OCC", "vacant": "FREE"}

    def _color(s: dict):
        occ = s.get("occupancy") or legacy_map.get(s.get("status", "vacant"), "FREE")
        return color_map.get(occ, color_map["UNKNOWN"])

    def _tag(s: dict) -> str:
        occ = s.get("occupancy") or legacy_map.get(s.get("status", "vacant"), "FREE")
        return occ  # "OCC" / "FREE" / "UNKNOWN"

    # 1. Semi-transparent fill
    for s in spaces:
        x1, y1, x2, y2 = s["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _color(s), -1)
    cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)

    # 2. Crisp borders + adaptive label badges
    line_w = max(2, w // 400)
    for s in spaces:
        x1, y1, x2, y2 = s["bbox"]
        slot_w = x2 - x1
        color = _color(s)
        tag = _tag(s)
        label = f"{s['id']} {tag}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_w, cv2.LINE_AA)

        # Fit label inside slot width
        fs = 0.40 if w < 800 else 0.50
        thick = 1 if w < 800 else 2
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        if tw > slot_w - 6:
            fs = max(0.22, fs * ((slot_w - 6) / max(tw, 1)))
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)

        bx = x1 + max(0, (slot_w - tw) // 2)
        by = y1 + th + 6
        cv2.rectangle(annotated, (bx - 3, by - th - 3), (bx + tw + 3, by + bl + 1), color, -1)
        cv2.putText(annotated, label, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick, cv2.LINE_AA)

    return annotated
