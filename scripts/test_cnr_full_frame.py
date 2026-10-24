
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw

ROOT = Path("dataset/cnrpark_ext/full_images")
MODEL_PATH = Path("models/cnrpark_cnn.keras")
OUTPUT_DIR = Path("outputs/cnrpark_demo")

FRAME_WIDTH = 1000
FRAME_HEIGHT = 750

# Original CNR-EXT annotation resolution
ORIGINAL_WIDTH = 2592
ORIGINAL_HEIGHT = 1944

IMAGE_SIZE = (150, 150)
THRESHOLD = 0.5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Locate an extracted full-frame image.
frames = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and p.parent.name.startswith("camera")
)

if not frames:
    raise FileNotFoundError(
        f"No extracted full-frame images found under {ROOT}. "
        "Check that the TAR archive was extracted."
    )

frame_path = frames[0]
camera_name = frame_path.parent.name
camera_number = camera_name.replace("camera", "")
annotation_path = ROOT / f"camera{camera_number}.csv"

if not annotation_path.exists():
    raise FileNotFoundError(
        f"Annotation CSV not found: {annotation_path}"
    )

print("Frame:", frame_path)
print("Camera:", camera_name)
print("Annotations:", annotation_path)

model = tf.keras.models.load_model(MODEL_PATH)
annotations = pd.read_csv(annotation_path)

image = Image.open(frame_path).convert("RGB")
image_width, image_height = image.size

print("Frame dimensions:", image.size)

# Calculate scaling from original annotation coordinates.
scale_x = image_width / ORIGINAL_WIDTH
scale_y = image_height / ORIGINAL_HEIGHT

draw = ImageDraw.Draw(image)
results = []

for _, row in annotations.iterrows():
    slot_id = int(row["SlotId"])

    x1 = int(round(float(row["X"]) * scale_x))
    y1 = int(round(float(row["Y"]) * scale_y))
    x2 = int(round((float(row["X"]) + float(row["W"])) * scale_x))
    y2 = int(round((float(row["Y"]) + float(row["H"])) * scale_y))

    # Clip the box to image boundaries.
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        continue

    crop = image.crop((x1, y1, x2, y2))
    crop = crop.resize(IMAGE_SIZE)

    # The model was trained with RGB values normalized to [0, 1].
    crop_array = np.asarray(crop, dtype=np.float32) / 255.0
    probability_busy = float(
        model.predict(crop_array[None, ...], verbose=0)[0][0]
    )

    is_busy = probability_busy >= THRESHOLD
    label = "BUSY" if is_busy else "FREE"
    color = "red" if is_busy else "lime"

    draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    draw.text(
        (x1, max(0, y1 - 15)),
        f"{slot_id}: {label} {probability_busy:.2f}",
        fill=color,
        stroke_width=1,
        stroke_fill="black",
    )

    results.append({
        "slot_id": slot_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "probability_busy": probability_busy,
        "prediction": label,
    })

output_image = OUTPUT_DIR / f"{camera_name}_prediction.jpg"
output_json = OUTPUT_DIR / f"{camera_name}_prediction.json"

image.save(output_image)
output_json.write_text(json.dumps(results, indent=2))

print("\n=== FULL-FRAME PREDICTIONS ===")
print("Total slots:", len(results))
print("Predicted free:", sum(r["prediction"] == "FREE" for r in results))
print("Predicted busy:", sum(r["prediction"] == "BUSY" for r in results))
print("Annotated image:", output_image)
print("Predictions JSON:", output_json)
print(
    "\nIMPORTANT: These are preliminary predictions. "
    "They are not a measured accuracy score because ground-truth "
    "occupancy for this particular frame has not yet been compared."
)
