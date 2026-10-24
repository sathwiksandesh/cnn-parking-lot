
from pathlib import Path
import pandas as pd

ROOT = Path("dataset/cnrpark_ext")
METADATA_PATH = ROOT / "metadata" / "CNRPark+EXT.csv"
FRAME_ROOT = ROOT / "full_images"
ANNOTATION_DIR = FRAME_ROOT

df = pd.read_csv(METADATA_PATH, low_memory=False)

# Keep only CNR-EXT rows whose images are described in the metadata.
df = df[df["image_url"].astype(str).str.startswith("CNR-EXT/PATCHES/")].copy()

df["camera_num"] = (
    df["camera"].astype(str).str.extract(r"(\d+)")[0]
)
df["datetime_parsed"] = pd.to_datetime(
    df["datetime"].astype(str),
    format="%Y-%m-%d_%H.%M",
    errors="coerce",
)

df["slot_id"] = pd.to_numeric(df["slot_id"], errors="coerce")
df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")
df = df.dropna(subset=["camera_num", "datetime_parsed", "slot_id", "occupancy"])
df["slot_id"] = df["slot_id"].astype(int)

# Select the same camera1 frame used in the previous demo.
frames = sorted(
    p for p in FRAME_ROOT.rglob("*.jpg")
    if p.parent.name == "camera1"
)
if not frames:
    raise FileNotFoundError("No camera1 frames found.")

frame = frames[0]
frame_time = pd.to_datetime(frame.stem, format="%Y-%m-%d_%H%M")
camera_num = frame.parent.name.replace("camera", "")

camera_rows = df[df["camera_num"].str.zfill(2) == camera_num.zfill(2)].copy()
camera_rows["time_difference_minutes"] = (
    camera_rows["datetime_parsed"] - frame_time
).dt.total_seconds().abs() / 60

print("Frame:", frame)
print("Frame timestamp:", frame_time)
print("Camera:", camera_num)

print("\n=== NEAREST METADATA TIMESTAMPS ===")
nearest_times = (
    camera_rows[["datetime_parsed", "time_difference_minutes"]]
    .drop_duplicates()
    .sort_values("time_difference_minutes")
    .head(5)
)
print(nearest_times.to_string(index=False))

print("\n=== SLOT LABEL COVERAGE AT NEAREST TIMESTAMP ===")
nearest_time = camera_rows["datetime_parsed"].iloc[
    camera_rows["time_difference_minutes"].argmin()
]
matched = camera_rows[camera_rows["datetime_parsed"] == nearest_time]

print("Matched metadata timestamp:", nearest_time)
print("Time difference in minutes:", abs((nearest_time - frame_time).total_seconds()) / 60)
print("Labelled slots:", len(matched))
print("Free labels (0):", int((matched["occupancy"] == 0).sum()))
print("Busy labels (1):", int((matched["occupancy"] == 1).sum()))
print("\nSample slot labels:")
print(
    matched[["slot_id", "occupancy", "image_url"]]
    .sort_values("slot_id")
    .head(15)
    .to_string(index=False)
)

print("\n=== ANNOTATION SLOT COUNT ===")
annotation_file = ANNOTATION_DIR / f"camera{camera_num}.csv"
if annotation_file.exists():
    annotations = pd.read_csv(annotation_file)
    print("Annotated slots:", len(annotations))
    print("Annotation IDs:", annotations["SlotId"].tolist())
else:
    print("Annotation file not found:", annotation_file)
