
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.training.train import CLASS_NAMES, make_dataset
from src.utils.config_loader import load_config, project_path


def main():
    config = load_config()

    image_size = int(config["data"]["image_width"])
    batch_size = int(config["data"]["batch_size"])
    seed = int(config["project"]["random_seed"])

    test_dir = project_path("dataset/splits/test")
    model_path = project_path(config["paths"]["model"])
    output_dir = project_path(config["paths"]["reports"])
    plot_dir = project_path(config["paths"]["plots"])

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run the training module first."
        )

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    test_ds = make_dataset(
        test_dir,
        image_size,
        batch_size,
        shuffle=False,
        seed=seed,
    )

    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []
    probabilities = []
    inference_seconds = 0.0
    image_count = 0

    for images, labels in test_ds:
        # Synchronize the first prediction by materializing its result.
        start = time.perf_counter()
        probs = model.predict(images, verbose=0)
        _ = probs.shape
        inference_seconds += time.perf_counter() - start

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probs, axis=1))
        probabilities.extend(probs.tolist())
        image_count += int(images.shape[0])

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== TEST SET EVALUATION ===")
    print(f"Test images: {image_count}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall:    {recall:.4f}")
    print(f"Macro F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    if image_count:
        print(
            "Average model.predict batch time per image:",
            f"{inference_seconds / image_count:.6f} seconds",
        )

    results = {
        "class_order": CLASS_NAMES,
        "test_images": int(image_count),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "timed_prediction_batches_seconds": float(
            inference_seconds
        ),
        "timed_prediction_images": int(image_count),
    }

    with (output_dir / "evaluation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Parking Occupancy Confusion Matrix",
    )

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        plot_dir / "confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    print("\nSaved evaluation report:", output_dir / "evaluation.json")
    print("Saved confusion matrix:", plot_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
