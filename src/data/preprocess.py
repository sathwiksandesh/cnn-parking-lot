
import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(
    image_path: str | Path,
    image_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Load an image, convert BGR to RGB, resize and normalize.

    Returns:
        float32 array with shape (height, width, 3)
        and pixel values in the range [0.0, 1.0].
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    width, height = image_size
    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32) / 255.0

    if image.shape != (height, width, 3):
        raise ValueError(
            f"Unexpected output shape: {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid numerical values.")

    return image


def validate_split(split_root: Path, samples: int = 5) -> None:
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Split directory not found: {split_root}"
        )

    image_paths = sorted(
        path for path in split_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No images found under {split_root}")

    print(f"Checking {min(samples, len(image_paths))} images...")

    for path in image_paths[:samples]:
        array = preprocess_image(path)

        print(
            f"{path.name}: "
            f"shape={array.shape}, "
            f"dtype={array.dtype}, "
            f"range=({array.min():.3f}, {array.max():.3f})"
        )

    print("Preprocessing validation: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("dataset/splits/train"),
    )
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")

    validate_split(args.split_root, args.samples)
