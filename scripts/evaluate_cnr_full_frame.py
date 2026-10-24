
from pathlib import Path
import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path("dataset/cnrpark_ext")
METADATA_PATH = ROOT / "metadata" / "CNRPark+EXT.csv"
PREDICTIONS_PATH = Path(
    "outputs/cnrpark_demo/camera1_prediction.json"
)
OUTPUT_PATH = Path(
    "outputs/reports/cnr_full_frame_evaluation.json"
)

# Exact frame used for the demo
CAMERA = "01"
FRAME_DATETIME = "2015-11-16_07.10"

# Read official metadata and saved predictions
df = pd.read_csv(METADATA_PATH, low_memory=False)
predictions = pd.DataFrame(
    json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
)

# Normalize metadata fields for exact matching
df["camera_normalized"] = (
    df["camera"].astype(str).str.strip().str.zfill(2)
)
df["datetime_normalized"] = (
    df["datetime"].astype(str).str.strip()
)
df["slot_id"] = pd.to_numeric(df["slot_id"], errors="coerce")
df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")

ground_truth = df[
    (df["camera_normalized"] == CAMERA)
    & (df["datetime_normalized"] == FRAME_DATETIME)
    & df["image_url"].astype(str).str.startswith("CNR-EXT/PATCHES/")
].copy()

ground_truth = ground_truth.dropna(subset=["slot_id", "occupancy"])
ground_truth["slot_id"] = ground_truth["slot_id"].astype(int)
ground_truth["occupancy"] = ground_truth["occupancy"].astype(int)

# Predictions: FREE=0, BUSY=1
predictions["slot_id"] = predictions["slot_id"].astype(int)
predictions["predicted_label"] = (
    predictions["prediction"].map({"FREE": 0, "BUSY": 1})
)

if ground_truth["slot_id"].duplicated().any():
    raise ValueError("Duplicate ground-truth slot IDs found.")

if predictions["slot_id"].duplicated().any():
    raise ValueError("Duplicate prediction slot IDs found.")

comparison = ground_truth[
    ["slot_id", "occupancy", "image_url"]
].merge(
    predictions[["slot_id", "predicted_label", "probability_busy"]],
    on="slot_id",
    how="outer",
    indicator=True,
)

missing_truth = comparison[comparison["_merge"] == "right_only"]
missing_predictions = comparison[comparison["_merge"] == "left_only"]

if not missing_truth.empty or not missing_predictions.empty:
    print("Ground-truth-only slots:", missing_predictions["slot_id"].tolist())
    print("Prediction-only slots:", missing_truth["slot_id"].tolist())
    raise ValueError("Slot IDs do not match; fix alignment before scoring.")

comparison = comparison.drop(columns="_merge")
comparison["correct"] = (
    comparison["occupancy"] == comparison["predicted_label"]
)

y_true = comparison["occupancy"]
y_pred = comparison["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("\n=== EXACT-FRAME EVALUATION ===")
print("Camera:", CAMERA)
print("Timestamp:", FRAME_DATETIME)
print("Matched slots:", len(comparison))
print("Ground truth free:", int((y_true == 0).sum()))
print("Ground truth busy:", int((y_true == 1).sum()))
print("Predicted free:", int((y_pred == 0).sum()))
print("Predicted busy:", int((y_pred == 1).sum()))
print(f"Accuracy: {accuracy:.2%}")

print("\nConfusion matrix (rows=true, columns=predicted):")
print(matrix)

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["free", "busy"],
    zero_division=0,
))

print("\n=== INCORRECT SLOT PREDICTIONS ===")
wrong = comparison[~comparison["correct"]].sort_values("slot_id")

if wrong.empty:
    print("All slots were classified correctly.")
else:
    print(wrong[
        ["slot_id", "occupancy", "predicted_label", "probability_busy"]
    ].to_string(index=False))

comparison["ground_truth"] = comparison["occupancy"].map(
    {0: "FREE", 1: "BUSY"}
)
comparison["predicted_label_text"] = comparison["predicted_label"].map(
    {0: "FREE", 1: "BUSY"}
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

results = {
    "camera": CAMERA,
    "datetime": FRAME_DATETIME,
    "number_of_slots": int(len(comparison)),
    "accuracy": float(accuracy),
    "confusion_matrix_rows_true_cols_predicted": matrix.tolist(),
    "per_slot_results": [
        {
            "slot_id": int(row.slot_id),
            "ground_truth": row.ground_truth,
            "prediction": row.predicted_label_text,
            "probability_busy": float(row.probability_busy),
            "correct": bool(row.correct),
        }
        for row in comparison.itertuples()
    ],
}

OUTPUT_PATH.write_text(json.dumps(results, indent=2))

print("\nSaved detailed evaluation:", OUTPUT_PATH)
