
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.training.build_model import build_cnn
from src.utils.config_loader import load_config, project_path


CLASS_NAMES = ["vacant", "occupied"]


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    print("TensorFlow:", tf.__version__)
    print("Visible GPUs:", gpus)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print("GPU memory configuration skipped:", exc)

    if not gpus:
        print(
            "WARNING: TensorFlow sees no GPU. "
            "Training will use the available CPU."
        )


def count_images(directory: Path) -> dict:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return {
        class_name: sum(
            1 for path in (directory / class_name).rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
        )
        for class_name in CLASS_NAMES
    }


def make_dataset(directory: Path, image_size: int, batch_size: int,
                 shuffle: bool, seed: int):
    for class_name in CLASS_NAMES:
        class_dir = directory / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        if count_images(directory)[class_name] == 0:
            raise ValueError(
                f"No images found for class '{class_name}' "
                f"in {directory}"
            )

    dataset = keras.utils.image_dataset_from_directory(
        directory,
        class_names=CLASS_NAMES,
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    # Pixel normalization: [0, 255] -> [0, 1].
    normalization = keras.layers.Rescaling(1.0 / 255.0)

    dataset = dataset.map(
        lambda images, labels: (
            normalization(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def make_class_weights(counts: dict) -> dict:
    total = sum(counts.values())

    if total == 0 or any(counts[c] == 0 for c in CLASS_NAMES):
        raise ValueError("Both classes must have training images.")

    # Class indices follow CLASS_NAMES:
    # vacant = 0, occupied = 1.
    return {
        index: total / (2.0 * counts[name])
        for index, name in enumerate(CLASS_NAMES)
    }


def plot_history(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dict = history.history

    for metric, filename, title in [
        ("loss", "loss_curve.png", "Training and Validation Loss"),
        ("accuracy", "accuracy_curve.png", "Training and Validation Accuracy"),
    ]:
        val_metric = f"val_{metric}"

        plt.figure(figsize=(8, 5))
        plt.plot(history_dict[metric], label="Training")

        if val_metric in history_dict:
            plt.plot(history_dict[val_metric], label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()


def main():
    config = load_config()
    seed = int(config["project"]["random_seed"])

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    configure_gpu()

    image_size = int(config["data"]["image_width"])
    batch_size = int(config["data"]["batch_size"])
    epochs = int(config["model"]["epochs"])
    learning_rate = float(config["model"]["learning_rate"])

    train_dir = project_path("dataset/splits/train")
    val_dir = project_path("dataset/splits/val")
    model_path = project_path(config["paths"]["model"])
    output_dir = project_path(config["paths"]["plots"])
    reports_dir = project_path(config["paths"]["reports"])

    for directory in (train_dir, val_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required dataset directory is missing: {directory}. "
                "Complete Level 2 first."
            )

    train_counts = count_images(train_dir)
    val_counts = count_images(val_dir)

    print("\nTraining class counts:", train_counts)
    print("Validation class counts:", val_counts)

    train_ds = make_dataset(
        train_dir, image_size, batch_size, True, seed
    )
    val_ds = make_dataset(
        val_dir, image_size, batch_size, False, seed
    )

    model = build_cnn(
        image_height=int(config["data"]["image_height"]),
        image_width=int(config["data"]["image_width"]),
        channels=int(config["data"]["channels"]),
        learning_rate=learning_rate,
    )

    model.summary()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(reports_dir / "training_history.csv")
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=make_class_weights(train_counts),
    )

    plot_history(history, output_dir)

    history_json = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with (reports_dir / "history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history_json, file, indent=2)

    print("\nTraining finished.")
    print("Best model path:", model_path)
    print("Plots directory:", output_dir)
    print("Reports directory:", reports_dir)


if __name__ == "__main__":
    main()
