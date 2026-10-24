
from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

ERRORS_PATH = (
    ROOT / "outputs" / "cnr_ext_fast_demo" / "misclassified_slots.csv"
)
MANIFEST_PATH = (
    ROOT / "dataset" / "cnrpark_ext" / "metadata"
    / "cnr_ext_frame_manifest.csv"
)
OUTPUT_PATH = (
    ROOT / "outputs" / "cnr_ext_fast_demo" / "error_contact_sheet.jpg"
)

manifest = pd.read_csv(MANIFEST_PATH)
errors = pd.read_csv(ERRORS_PATH)

manifest["datetime"] = pd.to_datetime(manifest["datetime_parsed"])
errors["datetime"] = pd.to_datetime(errors["datetime"])

# Keep one exact manifest record per misclassified slot.
merged = errors.merge(
    manifest[
        [
            "camera_num", "datetime", "slot_id",
            "frame_path", "x1", "y1", "x2", "y2",
            "occupancy",
        ]
    ].rename(columns={"camera_num": "camera"}),
    on=["camera", "datetime", "slot_id"],
    how="left",
    suffixes=("_prediction", "_manifest"),
)

# Fall back to the error CSV's true label if manifest naming differs.
if "occupancy" in merged.columns:
    merged["true_label"] = merged["occupancy"].fillna(
        merged["true_label"]
    ).astype(int)

tile_w, tile_h = 240, 200
image_w, image_h = 180, 135
columns = 3
rows = math.ceil(len(merged) / columns)

sheet = Image.new(
    "RGB",
    (columns * tile_w, rows * tile_h),
    "white",
)
draw = ImageDraw.Draw(sheet)

for index, row in merged.iterrows():
    path_value = row.get("frame_path")

    if pd.isna(path_value):
        continue

    frame_path = Path(str(path_value))
    if not frame_path.exists():
        continue

    with Image.open(frame_path) as frame:
        frame = frame.convert("RGB")
        box = (
            int(row["x1"]), int(row["y1"]),
            int(row["x2"]), int(row["y2"]),
        )
        crop = frame.crop(box).resize((image_w, image_h))

    position = int(index)
    x = (position % columns) * tile_w
    y = (position // columns) * tile_h

    sheet.paste(crop, (x, y))

    actual = "OCCUPIED" if int(row["true_label"]) == 1 else "FREE"
    predicted = (
        "OCCUPIED" if int(row["predicted_label"]) == 1 else "FREE"
    )

    draw.text(
        (x, y + image_h + 5),
        f'{row["date"]} Cam {int(row["camera"])} Slot {int(row["slot_id"])}',
        fill="black",
    )
    draw.text(
        (x, y + image_h + 23),
        f"Actual: {actual} | Pred: {predicted}",
        fill="black",
    )
    draw.text(
        (x, y + image_h + 41),
        f'Busy probability: {float(row["busy_probability"]):.3f}',
        fill="black",
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT_PATH, quality=92)

print(f"Saved contact sheet: {OUTPUT_PATH}")
print(f"Misclassified crops: {len(merged)}")
