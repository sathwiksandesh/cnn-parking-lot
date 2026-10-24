
from pathlib import Path
import pandas as pd

ROOT = Path("dataset/cnrpark_ext")
PATCH_ROOT = ROOT / "patches" / "extracted"
FRAME_ROOT = ROOT / "full_images"
METADATA = ROOT / "metadata" / "CNRPark+EXT.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 1. Count patch images by camera and class
print("\n=== PATCH DATASET ===")
patch_images = [
    p for p in PATCH_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]
print("Total patch images:", f"{len(patch_images):,}")

for camera in sorted(p for p in PATCH_ROOT.iterdir() if p.is_dir()):
    for label in ("free", "busy"):
        folder = camera / label
        if folder.exists():
            count = sum(
                1 for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            print(f"Camera {camera.name}, {label}: {count:,}")

# 2. Inspect the metadata and verify image paths
print("\n=== PATCH METADATA ===")
df = pd.read_csv(METADATA)
print("Rows:", f"{len(df):,}")
print("Columns:", df.columns.tolist())
print("Occupancy counts:")
print(df["occupancy"].value_counts(dropna=False).sort_index())

def locate_patch(image_url):
    if pd.isna(image_url):
        return None
    relative = Path(str(image_url).replace("\\", "/"))
    candidates = [
        PATCH_ROOT / relative,
        PATCH_ROOT / Path(*relative.parts[1:]),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

sample = df.head(10).copy()
sample["file_found"] = sample["image_url"].map(
    lambda x: locate_patch(x) is not None
)
print("\nFirst 10 metadata paths:")
print(sample[["image_url", "occupancy", "file_found"]].to_string(index=False))

# 3. Inspect full-frame images
print("\n=== FULL-FRAME IMAGES ===")
frame_images = [
    p for p in FRAME_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]
print("Total full-frame images:", f"{len(frame_images):,}")
for path in frame_images[:10]:
    print(path.relative_to(FRAME_ROOT))

# 4. Inspect official camera annotation CSVs
print("\n=== CAMERA ANNOTATIONS ===")
for i in range(1, 10):
    path = FRAME_ROOT / f"camera{i}.csv"
    if not path.exists():
        print(f"Missing: {path.name}")
        continue

    print(f"\n--- {path.name} ---")
    try:
        annotations = pd.read_csv(path)
        print("Columns:", annotations.columns.tolist())
        print(annotations.head(3).to_string(index=False))
    except Exception as exc:
        print("Could not read:", exc)
