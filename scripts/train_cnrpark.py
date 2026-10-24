
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20

ROOT = Path("dataset/cnrpark_ext/patches/extracted")
MODEL_DIR = Path("models")
REPORT_DIR = Path("outputs/reports")

MODEL_PATH = MODEL_DIR / "cnrpark_cnn.keras"
REPORT_PATH = REPORT_DIR / "cnrpark_evaluation.json"

tf.keras.utils.set_random_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Read images and infer labels from folder names
# --------------------------------------------------
records = []

for camera_dir in sorted(ROOT.iterdir()):
    if not camera_dir.is_dir():
        continue

    for label_name, label_id in [("free", 0), ("busy", 1)]:
        class_dir = camera_dir / label_name

        if not class_dir.exists():
            continue

        for path in class_dir.rglob("*"):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # Example:
            # 20150703_0805_14.jpg -> timestamp 20150703_0805
            timestamp = path.stem.rsplit("_", 1)[0]

            records.append({
                "path": str(path.resolve()),
                "label": label_id,
                "camera": camera_dir.name,
                "group": f"{camera_dir.name}_{timestamp}",
            })

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(f"No patch images found under {ROOT.resolve()}")

print("\nDataset size:", len(df))
print("Labels: 0 = free, 1 = busy")
print(df["label"].value_counts().sort_index())

# --------------------------------------------------
# 2. Split by timestamp group: 70% / 15% / 15% approx.
# --------------------------------------------------
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=SEED
)
train_idx, remaining_idx = next(
    split1.split(df, df["label"], groups=df["group"])
)

train_df = df.iloc[train_idx].copy()
remaining_df = df.iloc[remaining_idx].copy()

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=SEED
)
val_rel_idx, test_rel_idx = next(
    split2.split(
        remaining_df,
        remaining_df["label"],
        groups=remaining_df["group"],
    )
)

val_df = remaining_df.iloc[val_rel_idx].copy()
test_df = remaining_df.iloc[test_rel_idx].copy()

assert set(train_df["group"]).isdisjoint(val_df["group"])
assert set(train_df["group"]).isdisjoint(test_df["group"])
assert set(val_df["group"]).isdisjoint(test_df["group"])

for name, part in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(f"\n{name}: {len(part)} images")
    print(part["label"].value_counts().sort_index())

# --------------------------------------------------
# 3. Create TensorFlow datasets
# --------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)

def make_dataset(frame, training=False):
    paths = frame["path"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(
            min(len(frame), 4000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# --------------------------------------------------
# 4. Account for class imbalance
# --------------------------------------------------
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].to_numpy(),
)
class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}
print("\nClass weights:", class_weights)

# --------------------------------------------------
# 5. Build a CNN classifier
# --------------------------------------------------
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.025),
], name="augmentation")

inputs = tf.keras.Input(shape=(150, 150, 3))
x = augmentation(inputs)

for filters in (32, 64, 128):
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(
    192, 3, padding="same", activation="relu"
)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.20)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs, name="cnrpark_occupancy_cnn")

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

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Load the best saved checkpoint
model = tf.keras.models.load_model(MODEL_PATH)

probabilities = model.predict(test_ds, verbose=1).ravel()
predictions = (probabilities >= 0.5).astype(int)
actual = test_df["label"].to_numpy()

accuracy = accuracy_score(actual, predictions)
matrix = confusion_matrix(actual, predictions, labels=[0, 1])
report = classification_report(
    actual,
    predictions,
    labels=[0, 1],
    target_names=["free", "busy"],
    output_dict=True,
    zero_division=0,
)

results = {
    "model": "CNRPark CNN",
    "image_size": list(IMAGE_SIZE),
    "seed": SEED,
    "train_images": int(len(train_df)),
    "validation_images": int(len(val_df)),
    "test_images": int(len(test_df)),
    "test_accuracy": float(accuracy),
    "confusion_matrix_rows_true_cols_predicted": matrix.tolist(),
    "classification_report": report,
    "class_mapping": {"free": 0, "busy": 1},
}

REPORT_PATH.write_text(json.dumps(results, indent=2))

print("\n=== CNRPARK TEST RESULTS ===")
print(f"Test accuracy: {accuracy:.4f}")
print("\nConfusion matrix (rows=true, columns=predicted):")
print(matrix)
print("\nClassification report:")
print(classification_report(
    actual,
    predictions,
    labels=[0, 1],
    target_names=["free", "busy"],
    zero_division=0,
))
print(f"\nModel saved: {MODEL_PATH}")
print(f"Report saved: {REPORT_PATH}")
