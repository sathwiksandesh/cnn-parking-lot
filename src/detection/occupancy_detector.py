import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path


CLASS_NAMES = ["vacant", "occupied"]


def _crop(
    frame: np.ndarray,
    points: list[list[int]],
):
    polygon = np.asarray(points, dtype=np.int32)

    height, width = frame.shape[:2]

    if (
        np.any(polygon[:, 0] < 0)
        or np.any(polygon[:, 0] >= width)
        or np.any(polygon[:, 1] < 0)
        or np.any(polygon[:, 1] >= height)
    ):
        raise ValueError(
            f"Parking polygon extends outside frame bounds "
            f"({width}x{height})."
        )

    x, y, crop_width, crop_height = cv2.boundingRect(polygon)

    crop = frame[y:y + crop_height, x:x + crop_width]

    # Create a mask to retain only pixels inside the polygon.
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)

    local_polygon = polygon - np.array([x, y])
    cv2.fillPoly(mask, [local_polygon], 255)

    # Black out pixels outside the parking-space polygon.
    masked = cv2.bitwise_and(crop, crop, mask=mask)

    if masked.size == 0:
        raise ValueError("Empty parking-space crop.")

    return masked, polygon


def prepare_crop(
    crop: np.ndarray,
    image_size: int = 128,
) -> np.ndarray:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    rgb = cv2.resize(
        rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    return rgb.astype(np.float32) / 255.0


class OccupancyDetector:
    def __init__(
        self,
        model_path: str,
        image_size: int = 128,
        confidence_threshold: float = 0.60,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        # Directory for saving debug crops.
        self.debug_dir = Path("outputs/debug_crops")
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def predict_slots(
        self,
        frame: np.ndarray,
        slots: list[dict],
    ) -> list[dict]:
        crops = []
        valid_slots = []

        # Crop and prepare each parking space.
        for slot in slots:
            crop, polygon = polygon_crop(
                frame,
                slot["points"],
            )

            prepared = prepare_crop(
                crop,
                self.image_size,
            )

            # Save the actual masked crop for visual inspection.
            crop_path = self.debug_dir / f"{slot['id']}.jpg"

            saved = cv2.imwrite(str(crop_path), crop)

            if not saved:
                raise IOError(
                    f"Failed to save debug crop: {crop_path}"
                )

            crops.append(prepared)
            valid_slots.append((slot, polygon, crop_path))

        if not crops:
            return []

        # Run predictions for all parking spaces in one batch.
        batch = np.stack(crops, axis=0)

        probabilities = self.model.predict(
            batch,
            verbose=0,
        )

        results = []

        for (slot, polygon, crop_path), probs in zip(
            valid_slots,
            probabilities,
        ):
            index = int(np.argmax(probs))
            confidence = float(probs[index])

            status = (
                CLASS_NAMES[index]
                if confidence >= self.confidence_threshold
                else "uncertain"
            )

            result = {
                "id": slot["id"],
                "polygon": polygon,
                "status": status,
                "confidence": confidence,
                "probabilities": {
                    "vacant": float(probs[0]),
                    "occupied": float(probs[1]),
                },
                "debug_crop": str(crop_path),
            }

            results.append(result)

            print(
                f"{slot['id']}: {status} | "
                f"vacant={probs[0]:.4f} | "
                f"occupied={probs[1]:.4f} | "
                f"crop={crop_path}"
            )

        return results