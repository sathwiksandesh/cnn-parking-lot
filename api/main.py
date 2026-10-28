
from functools import lru_cache
from pathlib import Path
import base64
import io
import json
import sys

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cnr_ext_cnn_fast.keras"
SLOTS_PATH = BASE_DIR / "configs" / "parking_slots_new_camera.json"
OCCUPIED_THRESHOLD = 0.30

app = FastAPI(
    title="CNN Parking Lot Occupancy API",
    description="Parking-slot occupancy classification and visualization",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@lru_cache(maxsize=1)
def get_slots():
    if not SLOTS_PATH.exists():
        raise RuntimeError(f"Slot configuration not found: {SLOTS_PATH}")
    with open(SLOTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def root():
    return {
        "project": "CNN Parking Lot Occupancy Detector",
        "status": "running",
        "endpoints": ["/health", "/predict", "/detect", "/auto-detect", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_file_exists": MODEL_PATH.exists(),
        "slot_config_exists": SLOTS_PATH.exists(),
        "configured_slots": len(get_slots()) if SLOTS_PATH.exists() else 0,
        "threshold": OCCUPIED_THRESHOLD,
    }


def read_rgb_image(image_bytes):
    try:
        return np.array(
            Image.open(io.BytesIO(image_bytes)).convert("RGB")
        )
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file.")


def predict_crop(model, crop):
    height, width = int(model.input_shape[1]), int(model.input_shape[2])
    resized = cv2.resize(crop, (width, height))
    batch = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    probs = model.predict(batch, verbose=0)[0]  # shape: (num_classes,)
    # Single sigmoid output => P(occupied); 2-class softmax => probs[1] is P(occupied).
    if probs.shape[0] == 1:
        return float(probs[0])
    return float(probs[1])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_rgb_image(image_bytes)
    probability = predict_crop(get_model(), image)
    occupied = probability >= OCCUPIED_THRESHOLD

    return {
        "prediction": "Occupied" if occupied else "Free",
        "occupied_probability": round(probability, 4),
        "occupied_percentage": round(probability * 100, 2),
        "threshold": OCCUPIED_THRESHOLD,
        "input_size": "96x96",
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_rgb = read_rgb_image(image_bytes)
    image_height, image_width = image_rgb.shape[:2]

    slots = get_slots()
    model = get_model()

    # Coordinates were drafted for the uploaded 2048 x 1529 image.
    scale_x = image_width / 2048.0
    scale_y = image_height / 1529.0

    # OpenCV draws in BGR; convert from the RGB upload.
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = []

    for slot in slots:
        points = np.array(
            [
                [round(x * scale_x), round(y * scale_y)]
                for x, y in slot["points"]
            ],
            dtype=np.int32,
        )

        x, y, width, height = cv2.boundingRect(points)
        x1, y1 = max(0, x), max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)

        if x2 <= x1 or y2 <= y1:
            continue

        # Use the slot's polygon region, excluding pixels outside the polygon.
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        crop = image_rgb[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        masked_crop = cv2.bitwise_and(crop, crop, mask=crop_mask)

        probability = predict_crop(model, masked_crop)
        occupied = probability >= OCCUPIED_THRESHOLD

        color = (0, 0, 255) if occupied else (0, 200, 0)
        cv2.polylines(annotated, [points], True, color, 3)

        label = f"{slot['id']} {'OCC' if occupied else 'FREE'}"
        label_x = max(0, min(int(points[0][0]), image_width - 100))
        label_y = max(20, min(int(points[0][1]), image_height - 5))
        cv2.putText(
            annotated,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        results.append(
            {
                "slot_id": slot["id"],
                "status": "Occupied" if occupied else "Free",
                "occupied_probability": round(probability, 4),
            }
        )

    free_count = sum(item["status"] == "Free" for item in results)
    occupied_count = sum(item["status"] == "Occupied" for item in results)

    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode result image.")

    return {
        "total_slots_detected": len(results),
        "free_slots": free_count,
        "occupied_slots": occupied_count,
        "threshold": OCCUPIED_THRESHOLD,
        "slots": results,
        "annotated_image": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


@app.get("/auto-detect")
def auto_detect_info():
    return {
        "endpoint": "/auto-detect",
        "method": "POST",
        "description": "Send a multipart/form-data POST request with 'file': <image> to automatically detect and classify parking spaces.",
    }


@app.post("/auto-detect")
async def auto_detect(
    file: UploadFile = File(...),
    threshold: float = OCCUPIED_THRESHOLD,
    target_slots: int = 0,
):
    """
    Upload any parking-lot image; the API will:
      1. Detect the parking bays using AI + structural analysis.
      2. Classify every candidate with the parking-lot CNN.
      3. Return a clean annotated image (base64 JPEG) + occupancy statistics.
    """
    # Import here so the API still starts if ultralytics is not installed yet
    try:
        sys.path.insert(0, str(BASE_DIR))
        from src.detection.auto_detector import detect_and_classify, annotate_image
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"auto_detector module unavailable: {exc}",
        )

    image_bytes = await file.read()
    image_rgb   = read_rgb_image(image_bytes)

    model      = get_model()
    image_size = int(model.input_shape[1])   # CNN input size (e.g. 96)

    spaces = detect_and_classify(
        image_rgb,
        model,
        image_size=image_size,
        cnn_threshold=threshold,
        target_slots=target_slots,
    )

    annotated_bgr = annotate_image(image_rgb, spaces)

    success, encoded = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode result image.")

    free_count = sum(1 for s in spaces if s["status"] == "vacant")
    occ_count  = sum(1 for s in spaces if s["status"] == "occupied")

    return {
        "total_spaces"    : len(spaces),
        "free_spaces"     : free_count,
        "occupied_spaces" : occ_count,
        "threshold"       : OCCUPIED_THRESHOLD,
        "spaces": [
            {
                "id"                  : s["id"],
                "status"              : s["status"],
                "bbox"                : s["bbox"],
                "occupied_probability": s["occupied_probability"],
                "detection_source"    : s["detection_source"],
            }
            for s in spaces
        ],
        "annotated_image": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }
