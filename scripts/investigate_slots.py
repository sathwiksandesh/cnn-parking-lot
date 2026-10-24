
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

SCRIPT_PATH = globals().get("__file__")
ROOT = Path(SCRIPT_PATH).resolve().parents[1] if SCRIPT_PATH else Path.cwd()

manifest_path = (
    ROOT / "dataset" / "cnrpark_ext" / "metadata"
    / "cnr_ext_frame_manifest.csv"
)
output_dir = ROOT / "outputs" / "cnr_ext_fast_demo" / "investigate_slots"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(manifest_path)
df["datetime"] = pd.to_datetime(df["datetime_parsed"])

targets = [
    ("2015-12-10", 7, 285),
    ("2016-01-09", 7, 284),
    ("2016-01-09", 7, 285),
    ("2016-01-09", 8, 284),
]

for date, camera, slot_id in targets:
    rows = df[
        (df["datetime"].dt.strftime("%Y-%m-%d") == date)
        & (df["camera_num"] == camera)
        & (df["slot_id"] == slot_id)
    ]

    if rows.empty:
        print(f"Missing: {date}, camera {camera}, slot {slot_id}")
        continue

    row = rows.iloc[0]
    frame_path = Path(row["frame_path"])

    if not frame_path.exists():
        print(f"Missing frame file: {frame_path}")
        continue

    with Image.open(frame_path) as frame:
        frame = frame.convert("RGB")
        crop = frame.crop((
            int(row["x1"]), int(row["y1"]),
            int(row["x2"]), int(row["y2"]),
        ))

        crop = crop.resize((300, 300))
        draw = ImageDraw.Draw(crop)
        draw.text(
            (5, 5),
            f"GT occupancy={int(row['occupancy'])}",
            fill="yellow",
            stroke_width=2,
            stroke_fill="black",
        )

        name = f"{date}_cam{camera}_slot{slot_id}.jpg"
        crop.save(output_dir / name)

    print(
        f"{date} camera={camera} slot={slot_id} "
        f"ground_truth={int(row['occupancy'])}"
    )

print(f"\nSaved crops to: {output_dir}")
