
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "cnr_ext_cnn_fast.keras"
MANIFEST_PATH = (
    ROOT / "dataset" / "cnrpark_ext" / "metadata"
    / "cnr_ext_crops_manifest.csv"
)
SPLITS_PATH = (
    ROOT / "dataset" / "cnrpark_ext" / "metadata"
    / "cnr_ext_splits.json"
)

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 128


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)

    validation_dates = set(splits["validation_dates"])
    validation = manifest[
        (manifest["split"] == "val")
        & (manifest["date_group"].astype(str).isin(validation_dates))
    ].copy()

    if validation.empty:
        raise ValueError(
            "No validation rows found. Check the split labels and date_group."
        )

    print(f"Validation crops: {len(validation)}")
    print("Validation labels:")
    print(validation["occupancy"].value_counts().sort_index())

    paths = validation["crop_path"].astype(str).tolist()
    labels = validation["occupancy"].astype(int).to_numpy()

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(dataset, verbose=1).reshape(-1)

    results = []

    print("\nThreshold comparison (class 1 = occupied)")
    print(
        f"{'Threshold':>9} {'Accuracy':>10} {'Busy recall':>12} "
        f"{'Busy precision':>15} {'Busy F1':>9} {'Busy->Free':>12}"
    )

    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        predictions = (probabilities >= threshold).astype(int)

        tn = int(np.sum((labels == 0) & (predictions == 0)))
        fp = int(np.sum((labels == 0) & (predictions == 1)))
        fn = int(np.sum((labels == 1) & (predictions == 0)))
        tp = int(np.sum((labels == 1) & (predictions == 1)))

        accuracy = (tp + tn) / len(labels)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "busy_recall": recall,
            "busy_precision": precision,
            "busy_f1": f1,
            "free_predicted_busy": fp,
            "busy_predicted_free": fn,
            "true_free": tn,
            "true_busy": tp,
        })

        print(
            f"{threshold:9.2f} {accuracy:10.4f} {recall:12.4f} "
            f"{precision:15.4f} {f1:9.4f} {fn:12d}"
        )

    output_path = (
        ROOT / "outputs" / "cnr_ext_fast_demo"
        / "validation_threshold_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nSaved threshold results: {output_path}")


if __name__ == "__main__":
    main()
