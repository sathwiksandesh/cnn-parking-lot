
import json
from pathlib import Path


def load_slots(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Slot configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        slots = json.load(file)

    if not isinstance(slots, list) or not slots:
        raise ValueError("Slot configuration must be a non-empty list.")

    seen_ids = set()

    for slot in slots:
        if not isinstance(slot, dict):
            raise ValueError("Each slot must be a JSON object.")

        slot_id = slot.get("id")
        points = slot.get("points")

        if not isinstance(slot_id, str) or not slot_id.strip():
            raise ValueError("Every slot needs a non-empty string ID.")

        if slot_id in seen_ids:
            raise ValueError(f"Duplicate slot ID: {slot_id}")

        seen_ids.add(slot_id)

        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(
                f"Slot {slot_id} needs at least three polygon points."
            )

        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    not isinstance(value, (int, float))
                    for value in point
                )
            ):
                raise ValueError(
                    f"Invalid coordinate in slot {slot_id}: {point}"
                )

    return slots
