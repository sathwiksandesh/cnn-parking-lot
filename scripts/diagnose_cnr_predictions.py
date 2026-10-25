
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

MANIFEST = Path("dataset/cnrpark_ext/metadata/cnr_ext_frame_manifest.csv")
MODEL = Path("models/cnr_ext_cnn_fast.keras")
OUTPUT = Path("outputs/cnr_ext_fast_demo/diagnostic_predictions.csv")

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
SEED = 42
THRESHOLD = 0.30

df = pd.read_csv(MANIFEST, low_memory=False)
df["occupancy"] = df["occupancy"].astype(int)
df["date_group"] = df["date_group"].astype(str)

# Use only the existing held-out test dates.
test_dates = {"2015-12-03", "2015-12-10", "2016-01-09"}
df = df[df["date_group"].isin(test_dates)].copy()

rng = np.random.default_rng(SEED)
samples = []

for label in (0, 1):
    group = df[df["occupancy"] == label]
    n = min(200, len(group))
    samples.append(group.sample(n=n, random_state=SEED))

sample = pd.concat(samples).sample(frac=1, random_state=SEED)

model = tf.keras.models.load_model(MODEL, compile=False)
records = []
batch_images = []
batch_rows = []

def predict_batch():
    if not batch_images:
        return

    probs = model.predict(
        np.asarray(batch_images, dtype=np.float32),
        verbose=0,
    ).reshape(-1)

    for row, probability in zip(batch_rows, probs):
        actual = int(row["occupancy"])
        predicted = int(probability >= THRESHOLD)

        records.append({
            "date": row["date_group"],
            "camera": row["camera_name"],
            "slot_id": row["slot_id"],
            "actual": actual,
            "occupied_probability": float(probability),
            "predicted": predicted,
            "correct": predicted == actual,
            "frame_path": row["frame_path"],
        })

    batch_images.clear()
    batch_rows.clear()

for row in sample.to_dict("records"):
    with Image.open(row["frame_path"]) as image:
        image = image.convert("RGB")
        crop = image.crop((
            int(row["x1"]), int(row["y1"]),
            int(row["x2"]), int(row["y2"]),
        ))
        crop = crop.resize(
            IMAGE_SIZE, Image.Resampling.BILINEAR
        )
        pixels = np.asarray(crop, dtype=np.float32) / 255.0

    batch_images.append(pixels)
    batch_rows.append(row)

    if len(batch_images) >= BATCH_SIZE:
        predict_batch()

predict_batch()

result = pd.DataFrame(records)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

actual = result["actual"].to_numpy()
prob = result["occupied_probability"].to_numpy()
pred = (prob >= THRESHOLD).astype(int)

print("\nSample size:", len(result))
print("Threshold:", THRESHOLD)
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(actual, pred, labels=[0, 1]))
print(classification_report(
    actual, pred,
    labels=[0, 1],
    target_names=["FREE", "OCCUPIED"],
    zero_division=0,
))
print("\nMean occupied probability by true class:")
print(result.groupby("actual")["occupied_probability"].mean())
print("\nSaved:", OUTPUT)
