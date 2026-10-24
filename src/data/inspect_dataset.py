
from collections import Counter
from pathlib import Path
import argparse

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    "free": "vacant",
    "vacant": "vacant",
    "empty": "vacant",
    "busy": "occupied",
    "occupied": "occupied",
    "full": "occupied",
}


def inspect_dataset(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    counts = Counter()
    cameras = Counter()
    dimensions = Counter()
    corrupted = []

    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        raise RuntimeError(
            f"No supported images found under {root}. "
            "Check whether the archive was extracted."
        )

    for path in files:
        label = CLASS_MAP.get(path.parent.name.lower())

        if label is None:
            continue

        counts[label] += 1
        cameras[path.parent.parent.name] += 1

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                dimensions[image.size] += 1

        except (OSError, UnidentifiedImageError) as exc:
            corrupted.append((path, str(exc)))

    print("\n=== DATASET INSPECTION ===")
    print("Root:", root.resolve())
    print("Recognized images:", sum(counts.values()))

    for label in ("vacant", "occupied"):
        print(f"{label}: {counts[label]}")

    print("\nCamera-folder counts:")
    for camera, count in sorted(cameras.items()):
        print(f"  {camera}: {count}")

    print("\nImage dimensions:")
    for size, count in dimensions.most_common():
        print(f"  {size[0]}x{size[1]}: {count}")

    print("\nUnreadable images:", len(corrupted))
    for path, error in corrupted[:10]:
        print(f"  {path}: {error}")

    unrecognized = len(files) - sum(counts.values())
    print("Images outside recognized class folders:", unrecognized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted dataset root",
    )
    args = parser.parse_args()

    inspect_dataset(args.source)
