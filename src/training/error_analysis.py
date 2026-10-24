
import csv
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils.config_loader import load_config, project_path

CLASS_NAMES = ["vacant", "occupied"]


def main():
    config = load_config()
    image_size = int(config["data"]["image_width"])
    batch_size = int(config["data"]["batch_size"])

    test_dir = project_path("dataset/splits/test")
    model_path = project_path(config["paths"]["model"])
    output_dir = project_path("outputs/reports/error_analysis")
    image_output_dir = output_dir / "misclassified_images"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    raw_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=CLASS_NAMES,
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    # Capture paths before transforming the dataset.
    image_paths = list(raw_dataset.file_paths)

    # Match the evaluation module's normalization.
    dataset = raw_dataset.map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(dataset, verbose=1)

    if len(image_paths) != len(probabilities):
        raise RuntimeError(
            f"Image/prediction count mismatch: "
            f"{len(image_paths)} paths, {len(probabilities)} predictions"
        )

    actual_labels = np.array([
        CLASS_NAMES.index(Path(path).parent.name)
        for path in image_paths
    ])

    predicted_labels = np.argmax(probabilities, axis=1)
    incorrect_indices = np.flatnonzero(
        actual_labels != predicted_labels
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale copies so the folder represents this run only.
    if image_output_dir.exists():
        shutil.rmtree(image_output_dir)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "misclassified_images.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "actual",
                "predicted",
                "actual_class_probability",
                "predicted_class_probability",
            ],
        )
        writer.writeheader()

        for index in incorrect_indices:
            path = Path(image_paths[index])
            actual = int(actual_labels[index])
            predicted = int(predicted_labels[index])

            relative_path = path.relative_to(test_dir)
            output_path = image_output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(path, output_path)

            writer.writerow({
                "image": str(path),
                "actual": CLASS_NAMES[actual],
                "predicted": CLASS_NAMES[predicted],
                "actual_class_probability": round(
                    float(probabilities[index, actual]), 5
                ),
                "predicted_class_probability": round(
                    float(probabilities[index, predicted]), 5
                ),
            })

    total = len(actual_labels)
    errors = len(incorrect_indices)
    accuracy = float(np.mean(actual_labels == predicted_labels))

    print("\n=== ERROR ANALYSIS ===")
    print("Images processed:", total)
    print("Misclassified:", errors)
    print(f"Accuracy: {accuracy:.4%}")
    print("CSV report:", csv_path)
    print("Misclassified images:", image_output_dir)


if __name__ == "__main__":
    main()
