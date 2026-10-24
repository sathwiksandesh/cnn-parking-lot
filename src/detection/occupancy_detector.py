
import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path


def _crop(frame: np.ndarray, points: list[list[int]]):
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

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    local_polygon = polygon - np.array([x, y])
    cv2.fillPoly(mask, [local_polygon], 255)

    masked = cv2.bitwise_and(crop, crop, mask=mask)

    if masked.size == 0:
        raise ValueError("Empty parking-space crop.")

    return masked, polygon


def prepare_crop(crop: np.ndarray, image_size: int = 96):
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
        image_size: int = 96,
        confidence_threshold: float = 0.50,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            raise ValueError("Multiple model outputs are not supported.")

        self.single_sigmoid_output = (
            len(output_shape) == 2 and output_shape[-1] == 1
        )

        self.debug_dir = Path("outputs/debug_crops")
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def predict_slots(
        self,
        frame: np.ndarray,
        slots: list[dict],
    ) -> list[dict]:
        crops = []
        valid_slots = []

        for slot in slots:
            crop, polygon = _crop(frame, slot["points"])
            prepared = prepare_crop(crop, self.image_size)

            crop_path = self.debug_dir / f"{slot['id']}.jpg"
            if not cv2.imwrite(str(crop_path), crop):
                raise IOError(f"Failed to save debug crop: {crop_path}")

            crops.append(prepared)
            valid_slots.append((slot, polygon, crop_path))

        if not crops:
            return []

        batch = np.stack(crops, axis=0)
        predictions = self.model.predict(batch, verbose=0)

        results = []

        for (slot, polygon, crop_path), prediction in zip(
            valid_slots, predictions
        ):
            if self.single_sigmoid_output:
                # CNR-EXT model: sigmoid output is P(occupied).
                occupied_probability = float(np.asarray(prediction).reshape(-1)[0])
                vacant_probability = 1.0 - occupied_probability
            else:
                # Existing two-class model: [vacant, occupied].
                probs = np.asarray(prediction).reshape(-1)
                if len(probs) != 2:
                    raise ValueError(
                        f"Expected 2 outputs, received {len(probs)}."
                    )
                vacant_probability = float(probs[0])
                occupied_probability = float(probs[1])

            status = (
                "occupied"
                if occupied_probability >= self.confidence_threshold
                else "vacant"
            )

            confidence = (
                occupied_probability
                if status == "occupied"
                else vacant_probability
            )

            results.append({
                "id": slot["id"],
                "polygon": polygon,
                "status": status,
                "confidence": confidence,
                "probabilities": {
                    "vacant": vacant_probability,
                    "occupied": occupied_probability,
                },
                "debug_crop": str(crop_path),
            })

            print(
                f"{slot['id']}: {status} | "
                f"vacant={vacant_probability:.4f} | "
                f"occupied={occupied_probability:.4f} | "
                f"crop={crop_path}"
            )

        return results
