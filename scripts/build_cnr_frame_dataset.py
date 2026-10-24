
from pathlib import Path
import pandas as pd

ROOT = Path("dataset/cnrpark_ext")
FRAME_ROOT = ROOT / "full_images"
METADATA_PATH = ROOT / "metadata" / "CNRPark+EXT.csv"

df = pd.read_csv(METADATA_PATH, low_memory=False)

df = df[
    df["image_url"].astype(str).str.startswith("CNR-EXT/PATCHES/")
].copy()

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

df = df.dropna(
    subset=["camera_num", "datetime_parsed", "slot_id", "occupancy"]
)
df["camera_num"] = df["camera_num"].astype(int)
df["slot_id"] = df["slot_id"].astype(int)
df["occupancy"] = df["occupancy"].astype(int)

# Index all extracted full-frame images by camera and timestamp.
frames = {}
for path in FRAME_ROOT.rglob("*.jpg"):
    if not path.parent.name.startswith("camera"):
        continue

    camera_text = path.parent.name.replace("camera", "")
    try:
        camera_num = int(camera_text)
        timestamp = pd.to_datetime(
            path.stem, format="%Y-%m-%d_%H%M", errors="raise"
        )
    except (ValueError, TypeError):
        continue

    key = (camera_num, timestamp)
    frames.setdefault(key, []).append(path)

# Count exact frame/label matches and check annotation coverage.
matched_frame_keys = set(frames)
matched_metadata = df[
    df.apply(
        lambda row: (
            int(row["camera_num"]), row["datetime_parsed"]
        ) in matched_frame_keys,
        axis=1,
    )
].copy()

print("=== FULL-FRAME DATASET AUDIT ===")
print("Extracted frames:", sum(len(v) for v in frames.values()))
print("Unique camera/timestamp keys:", len(frames))
print("Metadata rows:", len(df))
print("Metadata rows with exact full-frame matches:", len(matched_metadata))
print("Unique matched frames:", matched_metadata[
    ["camera_num", "datetime_parsed"]
].drop_duplicates().shape[0])

print("\nMatched labels:")
print(matched_metadata["occupancy"].value_counts().sort_index())

print("\nMatches by camera:")
print(
    matched_metadata.groupby("camera_num")["datetime_parsed"]
    .nunique()
    .sort_index()
)

# Confirm every referenced slot has a corresponding camera annotation.
annotation_issues = []
for camera_num in sorted(matched_metadata["camera_num"].unique()):
    path = FRAME_ROOT / f"camera{camera_num}.csv"
    if not path.exists():
        annotation_issues.append(f"Missing {path.name}")
        continue

    annotations = pd.read_csv(path)
    annotation_ids = set(pd.to_numeric(
        annotations["SlotId"], errors="coerce"
    ).dropna().astype(int))

    camera_rows = matched_metadata[
        matched_metadata["camera_num"] == camera_num
    ]
    missing = sorted(set(camera_rows["slot_id"]) - annotation_ids)

    if missing:
        annotation_issues.append(
            f"Camera {camera_num}: metadata slot IDs missing "
            f"from annotations: {missing[:20]}"
        )

print("\nAnnotation issues:")
if annotation_issues:
    for issue in annotation_issues:
        print("-", issue)
else:
    print("No missing annotation files or slot IDs detected.")
