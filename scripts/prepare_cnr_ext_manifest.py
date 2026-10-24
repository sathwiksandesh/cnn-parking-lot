
from pathlib import Path
import pandas as pd

ROOT = Path("dataset/cnrpark_ext")
FRAME_ROOT = ROOT / "full_images"
METADATA_PATH = ROOT / "metadata" / "CNRPark+EXT.csv"
OUTPUT_PATH = ROOT / "metadata" / "cnr_ext_frame_manifest.csv"

FRAME_WIDTH = 1000
FRAME_HEIGHT = 750
ORIGINAL_WIDTH = 2592
ORIGINAL_HEIGHT = 1944

# Load the official labelled CNR-EXT patch metadata.
df = pd.read_csv(METADATA_PATH, low_memory=False)
df = df[
    df["image_url"].astype(str).str.startswith("CNR-EXT/PATCHES/")
].copy()

df["camera_num"] = pd.to_numeric(
    df["camera"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce",
)
df["datetime_parsed"] = pd.to_datetime(
    df["datetime"],
    format="%Y-%m-%d_%H.%M",
    errors="coerce",
)
df["slot_id"] = pd.to_numeric(df["slot_id"], errors="coerce")
df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")

df = df.dropna(
    subset=["camera_num", "datetime_parsed", "slot_id", "occupancy"]
).copy()

df["camera_num"] = df["camera_num"].astype(int)
df["slot_id"] = df["slot_id"].astype(int)
df["occupancy"] = df["occupancy"].astype(int)

# Index frames by camera and exact timestamp.
frame_index = {}

for path in FRAME_ROOT.rglob("*.jpg"):
    if not path.parent.name.startswith("camera"):
        continue

    try:
        camera_num = int(path.parent.name.replace("camera", ""))
        timestamp = pd.to_datetime(
            path.stem, format="%Y-%m-%d_%H%M", errors="raise"
        )
    except (ValueError, TypeError):
        continue

    key = (camera_num, timestamp)

    if key in frame_index:
        raise ValueError(f"Duplicate full-frame key: {key}")

    frame_index[key] = path.resolve()

# Load and combine the slot annotations for all nine cameras.
annotation_tables = []

for camera_num in range(1, 10):
    annotation_path = FRAME_ROOT / f"camera{camera_num}.csv"

    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    ann = pd.read_csv(annotation_path)
    ann = ann.rename(columns={
        "SlotId": "slot_id",
        "X": "x_original",
        "Y": "y_original",
        "W": "w_original",
        "H": "h_original",
    })

    ann["camera_num"] = camera_num
    annotation_tables.append(ann)

annotations = pd.concat(annotation_tables, ignore_index=True)

for column in ["slot_id", "x_original", "y_original", "w_original", "h_original"]:
    annotations[column] = pd.to_numeric(annotations[column], errors="coerce")

annotations = annotations.dropna(
    subset=["slot_id", "x_original", "y_original", "w_original", "h_original"]
).copy()
annotations["slot_id"] = annotations["slot_id"].astype(int)

# Every camera/slot pair should have one annotation.
if annotations.duplicated(["camera_num", "slot_id"]).any():
    raise ValueError("Duplicate camera/slot annotation pairs detected.")

# Attach the exact full-frame path.
df["frame_path"] = df.apply(
    lambda row: frame_index.get(
        (int(row["camera_num"]), row["datetime_parsed"])
    ),
    axis=1,
)

df = df.dropna(subset=["frame_path"]).copy()

# Join labels to slot bounding boxes.
manifest = df.merge(
    annotations,
    on=["camera_num", "slot_id"],
    how="inner",
    validate="many_to_one",
)

# Convert original-resolution coordinates to the 1000 x 750 frames.
manifest["x1"] = (
    manifest["x_original"] * FRAME_WIDTH / ORIGINAL_WIDTH
).round().astype(int)
manifest["y1"] = (
    manifest["y_original"] * FRAME_HEIGHT / ORIGINAL_HEIGHT
).round().astype(int)
manifest["x2"] = (
    (manifest["x_original"] + manifest["w_original"])
    * FRAME_WIDTH / ORIGINAL_WIDTH
).round().astype(int)
manifest["y2"] = (
    (manifest["y_original"] + manifest["h_original"])
    * FRAME_HEIGHT / ORIGINAL_HEIGHT
).round().astype(int)

# Clip to image bounds and remove invalid boxes.
manifest["x1"] = manifest["x1"].clip(0, FRAME_WIDTH - 1)
manifest["y1"] = manifest["y1"].clip(0, FRAME_HEIGHT - 1)
manifest["x2"] = manifest["x2"].clip(0, FRAME_WIDTH)
manifest["y2"] = manifest["y2"].clip(0, FRAME_HEIGHT)

manifest = manifest[
    (manifest["x2"] > manifest["x1"])
    & (manifest["y2"] > manifest["y1"])
].copy()

manifest["date_group"] = manifest["datetime_parsed"].dt.strftime("%Y-%m-%d")
manifest["camera_name"] = manifest["camera_num"].map(
    lambda n: f"camera{n}"
)

columns = [
    "frame_path", "camera_num", "camera_name", "datetime_parsed",
    "date_group", "slot_id", "occupancy", "x1", "y1", "x2", "y2",
]

manifest = manifest[columns].drop_duplicates(
    ["camera_num", "datetime_parsed", "slot_id"]
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(OUTPUT_PATH, index=False)

print("\n=== CNR-EXT FRAME MANIFEST ===")
print("Manifest:", OUTPUT_PATH)
print("Labelled slot crops:", f"{len(manifest):,}")
print("Unique full frames:", manifest["frame_path"].nunique())
print("Unique dates:", manifest["date_group"].nunique())
print("Unique cameras:", manifest["camera_num"].nunique())

print("\nOccupancy labels (0=free, 1=busy):")
print(manifest["occupancy"].value_counts().sort_index())

print("\nRows per camera:")
print(manifest.groupby("camera_name").size().to_string())

print("\nRows per date:")
print(manifest.groupby("date_group").size().to_string())

print("\nExample rows:")
print(manifest.head(5).to_string(index=False))
