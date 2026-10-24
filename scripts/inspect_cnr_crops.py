
from pathlib import Path
import json
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("dataset/cnrpark_ext/full_images")
RESULTS_PATH = Path("outputs/cnrpark_demo/camera1_prediction.json")
OUTPUT_PATH = Path("outputs/cnrpark_demo/camera1_crop_montage.jpg")

# Locate the same first camera1 frame used by the prediction script.
frames = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and p.parent.name == "camera1"
)

if not frames:
    raise FileNotFoundError("No extracted camera1 frame found.")

if not RESULTS_PATH.exists():
    raise FileNotFoundError(f"Missing predictions: {RESULTS_PATH}")

frame_path = frames[0]
image = Image.open(frame_path).convert("RGB")

records = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

CELL_W = 190
CELL_H = 205
CROP_SIZE = 150
COLUMNS = 5
ROWS = math.ceil(len(records) / COLUMNS)

sheet = Image.new(
    "RGB",
    (COLUMNS * CELL_W, ROWS * CELL_H),
    "white",
)
draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("arial.ttf", 14)
except OSError:
    font = ImageFont.load_default()

for index, record in enumerate(records):
    x1, y1 = record["x1"], record["y1"]
    x2, y2 = record["x2"], record["y2"]

    crop = image.crop((x1, y1, x2, y2))
    crop.thumbnail((CROP_SIZE, CROP_SIZE))

    cell_x = (index % COLUMNS) * CELL_W
    cell_y = (index // COLUMNS) * CELL_H

    paste_x = cell_x + (CELL_W - crop.width) // 2
    sheet.paste(crop, (paste_x, cell_y + 5))

    probability = record["probability_busy"]
    label = record["prediction"]
    slot_id = record["slot_id"]

    line1 = f"Slot {slot_id}: {label}"
    line2 = f"Busy probability: {probability:.3f}"

    draw.text(
        (cell_x + 5, cell_y + 160),
        line1,
        fill="red" if label == "BUSY" else "green",
        font=font,
    )
    draw.text(
        (cell_x + 5, cell_y + 180),
        line2,
        fill="black",
        font=font,
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT_PATH, quality=95)

print("Source frame:", frame_path)
print("Slots in montage:", len(records))
print("Saved crop montage:", OUTPUT_PATH)
