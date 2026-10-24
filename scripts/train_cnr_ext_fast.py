
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 3

CROP_MANIFEST = Path(
    "dataset/cnrpark_ext/metadata/cnr_ext_crops_manifest.csv"
)
SPLIT_FILE = Path(
    "dataset/cnrpark_ext/metadata/cnr_ext_splits.json"
)

# Separate filenames: preserve every previous model.
MODEL_PATH = Path("models/cnr_ext_cnn_fast.keras")
CHECKPOINT_DIR = Path("models/cnr_ext_fast_checkpoints")
REPORT_PATH = Path("outputs/reports/cnr_ext_fast_evaluation.json")

tf.keras.utils.set_random_seed(SEED)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not CROP_MANIFEST.exists() or not SPLIT_FILE.exists():
    raise FileNotFoundError(
        "Crop manifest or split file is missing. "
        "Run scripts/prepare_cnr_ext_crops.py first."
    )

df = pd.read_csv(CROP_MANIFEST)
split_info = json.loads(SPLIT_FILE.read_text())

# --------------------------------------------------
# 1. Validate saved crops and date separation
# --------------------------------------------------
required = {
    "crop_path", "split", "occupancy", "date_group"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing manifest columns: {sorted(missing)}")

df["occupancy"] = df["occupancy"].astype(int)
df["date_group"] = df["date_group"].astype(str)

train_dates = set(split_info["train_dates"])
val_dates = set(split_info["validation_dates"])
test_dates = set(split_info["test_dates"])

assert train_dates.isdisjoint(val_dates)
assert train_dates.isdisjoint(test_dates)
assert val_dates.isdisjoint(test_dates)

for name in ("train", "val", "test"):
    part = df[df["split"] == name]
    if part.empty:
        raise ValueError(f"{name} split is empty.")

    missing_files = [
        p for p in part["crop_path"]
        if not Path(p).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{name}: missing crop files, e.g. {missing_files[:3]}"
        )

    print(
        f"{name}: {len(part):,} crops | "
        f"{part['date_group'].nunique()} dates | "
        f"labels {part['occupancy'].value_counts().sort_index().to_dict()}"
    )

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

# --------------------------------------------------
# 2. Read pre-extracted crops efficiently
# --------------------------------------------------
def decode_crop(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    return image, label

def make_dataset(frame, training=False):
    paths = frame["crop_path"].astype(str).to_numpy()
    labels = frame["occupancy"].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            min(len(frame), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        decode_crop,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# --------------------------------------------------
# 3. CNN model
# --------------------------------------------------
inputs = tf.keras.Input(shape=(96, 96, 3))
x = inputs

for filters in (24, 48, 96):
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(48, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(
    inputs, outputs, name="cnr_ext_fast_occupancy_cnn"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

# Save a checkpoint after EVERY completed epoch, and also
# keep the best validation-loss model separately.
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_DIR / "epoch_{epoch:02d}.keras"),
        save_best_only=False,
        save_freq="epoch",
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    ),
]

# --------------------------------------------------
# 4. Train
# --------------------------------------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if not MODEL_PATH.exists():
    raise RuntimeError(
        "No best model was saved. Check validation output and errors."
    )

# --------------------------------------------------
# 5. Evaluate the untouched held-out test set
# --------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
probabilities = model.predict(test_ds, verbose=1).ravel()
actual = test_df["occupancy"].to_numpy(dtype=int)

if len(probabilities) != len(actual):
    raise RuntimeError(
        f"Prediction count {len(probabilities)} != "
        f"test labels {len(actual)}"
    )

predicted = (probabilities >= 0.5).astype(int)
matrix = confusion_matrix(actual, predicted, labels=[0, 1])

class_report = classification_report(
    actual,
    predicted,
    labels=[0, 1],
    target_names=["free", "busy"],
    output_dict=True,
    zero_division=0,
)

results = {
    "model": "CNR-EXT fast CNN",
    "seed": SEED,
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "split_method": "date-grouped",
    "train_dates": sorted(train_dates),
    "validation_dates": sorted(val_dates),
    "test_dates": sorted(test_dates),
    "train_crops": len(train_df),
    "validation_crops": len(val_df),
    "test_crops": len(test_df),
    "test_accuracy": float(accuracy_score(actual, predicted)),
    "confusion_matrix_rows_true_cols_predicted": matrix.tolist(),
    "classification_report": class_report,
}

REPORT_PATH.write_text(json.dumps(results, indent=2))

print("\n=== CNR-EXT FAST HELD-OUT TEST RESULTS ===")
print(f"Test accuracy: {results['test_accuracy']:.4%}")
print("Confusion matrix (rows=true, columns=predicted):")
print(matrix)
print(classification_report(
    actual,
    predicted,
    labels=[0, 1],
    target_names=["free", "busy"],
    zero_division=0,
))
print("Best model:", MODEL_PATH)
print("Epoch checkpoints:", CHECKPOINT_DIR)
print("Report:", REPORT_PATH)
