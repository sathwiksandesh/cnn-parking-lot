
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    "free": "vacant",
    "vacant": "vacant",
    "empty": "vacant",
    "busy": "occupied",
    "occupied": "occupied",
    "full": "occupied",
}

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def find_images(source: Path):
    records = []

    for path in source.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = CLASS_MAP.get(path.parent.name.lower())
        if label is None:
            continue

        # Expected: camera/class/image.jpg
        camera = path.parent.parent.name

        # Example filename: 20150703_1425_32.jpg
        # The final filename component identifies the parking slot.
        slot_id = path.stem.rsplit("_", 1)[-1]

        # Include camera to avoid collisions between camera-local IDs.
        group = f"{camera}_{slot_id}"

        records.append({
            "path": path,
            "label": label,
            "camera": camera,
            "slot_id": slot_id,
            "group": group,
        })

    return records


def assign_group_splits(records, seed: int):
    groups = defaultdict(list)

    for record in records:
        groups[record["group"]].append(record)

    group_names = list(groups)
    random.Random(seed).shuffle(group_names)

    # Allocate complete slot groups, never individual images.
    total_images = len(records)
    targets = {
        "train": total_images * SPLITS["train"],
        "val": total_images * SPLITS["val"],
        "test": total_images * SPLITS["test"],
    }

    assigned = {name: [] for name in SPLITS}
    counts = Counter()

    for group_name in group_names:
        group_records = groups[group_name]

        # Fill train first, then validation, then test.
        split = min(
            SPLITS,
            key=lambda name: counts[name] / max(targets[name], 1),
        )

        assigned[split].extend(group_records)
        counts[split] += len(group_records)

    return assigned


def prepare_dataset(source: Path, output: Path, seed: int = 42):
    source = source.resolve()
    output = output.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    # Never place generated output inside the raw source directory.
    if output == source or source in output.parents:
        raise ValueError(
            "Output must not be inside the source dataset directory."
        )

    records = find_images(source)

    if not records:
        raise RuntimeError(
            "No labeled images found. Expected parent folders named "
            "free/busy or vacant/occupied."
        )

    bad = []
    for record in records:
        try:
            from PIL import Image

            with Image.open(record["path"]) as image:
                image.verify()
        except Exception as exc:
            bad.append((record["path"], str(exc)))

    if bad:
        examples = "\n".join(str(path) for path, _ in bad[:5])
        raise RuntimeError(
            f"Found {len(bad)} unreadable images. Fix or remove them first:\n"
            f"{examples}"
        )

    if len({record["label"] for record in records}) != 2:
        raise RuntimeError(
            "Both vacant and occupied classes must be present."
        )

    assigned = assign_group_splits(records, seed)

    # A fresh split output is required to avoid stale files.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}\n"
            "Move it aside or delete the generated split directory, "
            "then rerun this script."
        )

    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "manifest.csv"
    summary = {}

    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "split", "label", "camera", "slot_id",
                "group", "source", "destination",
            ],
        )
        writer.writeheader()

        for split, split_records in assigned.items():
            counts = Counter()

            for record in split_records:
                label = record["label"]
                destination_dir = output / split / label
                destination_dir.mkdir(parents=True, exist_ok=True)

                # Preserve enough path information to avoid basename clashes.
                relative = record["path"].relative_to(source)
                safe_name = "__".join(relative.parts)

                destination = destination_dir / safe_name

                shutil.copy2(record["path"], destination)
                counts[label] += 1

                writer.writerow({
                    "split": split,
                    "label": label,
                    "camera": record["camera"],
                    "slot_id": record["slot_id"],
                    "group": record["group"],
                    "source": str(record["path"]),
                    "destination": str(destination),
                })

            summary[split] = counts

    print("\n=== DATASET SPLIT COMPLETE ===")
    print("Output:", output)
    print("Manifest:", manifest_path)

    for split in SPLITS:
        counts = summary[split]
        print(
            f"{split}: total={sum(counts.values())}, "
            f"vacant={counts['vacant']}, "
            f"occupied={counts['occupied']}"
        )

    # Confirm that no slot group appears in multiple splits.
    group_sets = {
        split: {r["group"] for r in assigned[split]}
        for split in SPLITS
    }

    for first, second in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        overlap = group_sets[first] & group_sets[second]
        if overlap:
            raise RuntimeError(
                f"Group leakage between {first} and {second}: "
                f"{len(overlap)} overlapping groups"
            )

    print("Group leakage check: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/splits"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(args.source, args.output, args.seed)
