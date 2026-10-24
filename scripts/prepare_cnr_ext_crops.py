
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
SIZE = (96, 96)

MANIFEST = Path("dataset/cnrpark_ext/metadata/cnr_ext_frame_manifest.csv")
OUTPUT = Path("dataset/cnrpark_ext/crops_96")
SPLIT_FILE = Path("dataset/cnrpark_ext/metadata/cnr_ext_splits.json")
CROP_MANIFEST = Path("dataset/cnrpark_ext/metadata/cnr_ext_crops_manifest.csv")

df = pd.read_csv(MANIFEST, low_memory=False)
df["date_group"] = df["date_group"].astype(str)
df["occupancy"] = df["occupancy"].astype(int)

# Recreate a deterministic date-based split.
dates = df[["date_group"]].drop_duplicates().reset_index(drop=True)

split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, remaining_idx = next(
    split1.split(dates, groups=dates["date_group"])
)
train_dates = set(dates.iloc[train_idx]["date_group"])
remaining = dates.iloc[remaining_idx]["date_group"].to_numpy()

split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 1)
val_idx, test_idx = next(
    split2.split(np.zeros(len(remaining)), groups=remaining)
)
val_dates = set(remaining[val_idx])
test_dates = set(remaining[test_idx])

def get_split(date):
    if date in train_dates:
        return "train"
    if date in val_dates:
        return "val"
    return "test"

df["split"] = df["date_group"].map(get_split)

# Keep a manageable training subset, smaller validation subset,
# and the entire held-out test set.
rng = np.random.default_rng(SEED)
selected = []

for split, limit in [
    ("train", 20000),
    ("val", 4000),
    ("test", None),
]:
    part = df[df["split"] == split]
    if limit is not None and len(part) > limit:
        # Stratify by occupancy so both classes are represented.
        chosen = []
        for label, group in part.groupby("occupancy"):
            n = max(1, round(limit * len(group) / len(part)))
            idx = rng.choice(group.index.to_numpy(),
                             size=min(n, len(group)),
                             replace=False)
            chosen.extend(idx.tolist())
        part = df.loc[sorted(chosen)]
        if len(part) > limit:
            part = part.sample(n=limit, random_state=SEED)
    selected.append(part)

df = pd.concat(selected).sort_values(
    ["split", "frame_path", "slot_id"]
).reset_index(drop=True)

for split in ("train", "val", "test"):
    for label in (0, 1):
        (OUTPUT / split / str(label)).mkdir(parents=True, exist_ok=True)

records = []
# Group by frame so each full-resolution image is opened only once.
for frame_number, (frame_path, rows) in enumerate(
    df.groupby("frame_path", sort=False)
):
    try:
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
            width, height = image.size

            for row in rows.itertuples():
                x1 = max(0, min(int(row.x1), width - 1))
                y1 = max(0, min(int(row.y1), height - 1))
                x2 = max(0, min(int(row.x2), width))
                y2 = max(0, min(int(row.y2), height))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image.crop((x1, y1, x2, y2))
                crop = crop.resize(SIZE, Image.Resampling.BILINEAR)

                # Stable filename from the original manifest row index.
                filename = f"{row.Index:06d}.jpg"
                out = OUTPUT / row.split / str(row.occupancy) / filename
                crop.save(out, quality=90)

                records.append({
                    "crop_path": str(out.resolve()),
                    "split": row.split,
                    "occupancy": int(row.occupancy),
                    "date_group": str(row.date_group),
                    "frame_path": str(frame_path),
                    "slot_id": str(row.slot_id),
                })
    except Exception as exc:
        raise RuntimeError(f"Failed processing frame {frame_path}") from exc

    if (frame_number + 1) % 200 == 0:
        print(f"Processed {frame_number + 1} frames; saved {len(records)} crops")

pd.DataFrame(records).to_csv(CROP_MANIFEST, index=False)

SPLIT_FILE.write_text(json.dumps({
    "seed": SEED,
    "image_size": list(SIZE),
    "train_dates": sorted(train_dates),
    "validation_dates": sorted(val_dates),
    "test_dates": sorted(test_dates),
    "selected_crops": len(records),
}, indent=2))

print("\nCrop preparation complete.")
print("Crops saved:", len(records))
print("Crop manifest:", CROP_MANIFEST)
print("Split information:", SPLIT_FILE)
print(pd.DataFrame(records).groupby(
    ["split", "occupancy"]
).size())
