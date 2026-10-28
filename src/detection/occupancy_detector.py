from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def _crop(frame: np.ndarray, points, mask_polygon: bool = False):
    """Rectangular crop of the polygon's bounding box, clamped to the frame.

    The CNN was trained on plain rectangular crops, so masking is OFF by default.
    """
    polygon = np.asarray(points, dtype=np.int32)
    height, width = frame.shape[:2]

    x0 = max(0, int(polygon[:, 0].min()))
    y0 = max(0, int(polygon[:, 1].min()))
    x1 = min(width, int(polygon[:, 0].max()))
    y1 = min(height, int(polygon[:, 1].max()))

    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("Slot lies outside the frame or is too small.")

    crop = frame[y0:y1, x0:x1]

    if mask_polygon:
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon - np.array([x0, y0])], 255)
        crop = cv2.bitwise_and(crop, crop, mask=mask)

    return crop, polygon


def prepare_crop(crop: np.ndarray, image_size: int = 96):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


class OccupancyDetector:
    def __init__(
        self,
        model_path: str,
        image_size: int = 96,
        confidence_threshold: float = 0.30,
        mask_polygon: bool = False,
        debug: bool = False,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.mask_polygon = mask_polygon
        self.debug = debug

        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            raise ValueError("Multiple model outputs are not supported.")
        self.single_sigmoid_output = (
            len(output_shape) == 2 and output_shape[-1] == 1
        )

        self.debug_dir = Path("outputs/debug_crops")
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def predict_slots(self, frame: np.ndarray, slots: list[dict]) -> list[dict]:
        crops, valid = [], []

        for slot in slots:
            try:
                crop, polygon = _crop(frame, slot["points"], self.mask_polygon)
            except ValueError:
                continue  # skip unusable slot instead of crashing the whole frame

            crop_path = None
            if self.debug:
                crop_path = self.debug_dir / f"{slot['id']}.jpg"
                cv2.imwrite(str(crop_path), crop)

            crops.append(prepare_crop(crop, self.image_size))
            valid.append((slot, polygon, crop_path))

        if not crops:
            return []

        predictions = self.model.predict(np.stack(crops), verbose=0)
        results = []

        for (slot, polygon, crop_path), prediction in zip(valid, predictions):
            probs = np.asarray(prediction).reshape(-1)
            if self.single_sigmoid_output:
                occupied_p = float(probs[0])          # sigmoid = P(occupied)
                vacant_p = 1.0 - occupied_p
            else:
                if len(probs) != 2:
                    raise ValueError(f"Expected 2 outputs, got {len(probs)}.")
                vacant_p, occupied_p = float(probs[0]), float(probs[1])

            status = "occupied" if occupied_p >= self.confidence_threshold else "vacant"
            confidence = occupied_p if status == "occupied" else vacant_p

            results.append({
                "id": slot["id"],
                "polygon": polygon,
                "status": status,
                "confidence": confidence,
                "probabilities": {"vacant": vacant_p, "occupied": occupied_p},
                "debug_crop": str(crop_path) if crop_path else None,
            })

        return results