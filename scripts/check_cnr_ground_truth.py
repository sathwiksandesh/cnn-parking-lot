
from pathlib import Path
import pandas as pd

ROOT = Path("dataset/cnrpark_ext")
METADATA_PATH = ROOT / "metadata" / "CNRPark+EXT.csv"
FRAME_ROOT = ROOT / "full_images"

df = pd.read_csv(METADATA_PATH, low_memory=False)

print("=== CAMERAS IN METADATA ===")
print(df["camera"].value_counts(dropna=False).head(15))

print("\n=== IMAGE PATH PREFIXES ===")
print(
    df["image_url"]
    .dropna()
    .astype(str)
    .str.split("/")
    .str[:3]
    .value_counts()
    .head(20)
)

print("\n=== SAMPLE NUMERIC-CAMERA ROWS ===")
numeric_camera = pd.to_numeric(df["camera"], errors="coerce")
print(
    df.loc[
        numeric_camera.notna(),
        ["camera", "datetime", "image_url", "occupancy", "slot_id"],
    ].head(10).to_string(index=False)
)

# Find the exact frame selected by the previous demo script.
frames = sorted(
    p for p in FRAME_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and p.parent.name == "camera1"
)

if not frames:
    print("\nNo camera1 full-frame images found.")
    raise SystemExit(0)

frame = frames[0]
timestamp = frame.stem.replace("-", "").replace("_", "")
print("\n=== DEMO FRAME ===")
print("Path:", frame)
print("Timestamp:", timestamp)

# Match the frame timestamp to metadata, if available.
datetime_normalized = (
    df["datetime"].astype(str)
    .str.replace("-", "", regex=False)
    .str.replace("_", "", regex=False)
)

camera_numeric = pd.to_numeric(df["camera"], errors="coerce")
matches = df[
    (camera_numeric == 1) &
    (datetime_normalized == timestamp)
]

print("\n=== EXACT TIMESTAMP MATCH ===")
print("Matching rows:", len(matches))

if not matches.empty:
    print(
        matches[
            ["camera", "datetime", "image_url", "slot_id", "occupancy"]
        ].sort_values("slot_id").to_string(index=False)
    )
else:
    print("No exact match. The frame may not have a corresponding metadata row.")
    print("We'll inspect the nearest available timestamps next.")
