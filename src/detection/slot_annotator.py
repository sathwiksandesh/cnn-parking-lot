
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.config_loader import project_path


class SlotAnnotator:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.current_points = []
        self.slots = []
        self.window = "Parking Slot Annotator"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([int(x), int(y)])

    def render(self):
        canvas = self.frame.copy()

        for slot in self.slots:
            polygon = np.asarray(slot["points"], dtype=np.int32)
            cv2.polylines(
                canvas, [polygon], True, (0, 200, 0), 2
            )
            center = polygon.mean(axis=0).astype(int)
            cv2.putText(
                canvas,
                slot["id"],
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )

        if self.current_points:
            for point in self.current_points:
                cv2.circle(
                    canvas, tuple(point), 5, (0, 255, 255), -1
                )

            points = np.asarray(
                self.current_points, dtype=np.int32
            )
            if len(points) >= 2:
                cv2.polylines(
                    canvas, [points], False, (0, 255, 255), 2
                )

        instructions = [
            "Click slot corners in order",
            "N: finish slot | U: undo point",
            "R: reset current slot | Q: save and exit",
        ]

        for i, text in enumerate(instructions):
            cv2.putText(
                canvas,
                text,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

    def run(self, output_path: Path):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key == ord("n"):
                if len(self.current_points) < 3:
                    print("A slot needs at least 3 points.")
                    continue

                self.slots.append({
                    "id": f"P{len(self.slots) + 1:02d}",
                    "points": self.current_points.copy(),
                })
                print("Added:", self.slots[-1]["id"])
                self.current_points = []

            elif key == ord("u"):
                if self.current_points:
                    self.current_points.pop()

            elif key == ord("r"):
                self.current_points = []

            elif key == ord("q"):
                break

        cv2.destroyAllWindows()

        if not self.slots:
            print("No slots were saved.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.slots, file, indent=2)

        print(f"Saved {len(self.slots)} slots to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Video or image path relative to project root",
    )
    parser.add_argument(
        "--output",
        default="configs/parking_slots.json",
    )
    args = parser.parse_args()

    source = project_path(args.source)
    output = project_path(args.output)

    if not source.is_file():
        raise FileNotFoundError(f"Input not found: {source}")

    if source.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        frame = cv2.imread(str(source))
    else:
        capture = cv2.VideoCapture(str(source))
        ok, frame = capture.read()
        capture.release()

        if not ok:
            raise RuntimeError("Could not read the first video frame.")

    if frame is None:
        raise RuntimeError("Could not load the input image.")

    print("Instructions:")
    print("1. Click the corners of each parking space in order.")
    print("2. Press N to finish that slot.")
    print("3. Press U to undo a point; R to reset the current slot.")
    print("4. Press Q to save and exit.")

    SlotAnnotator(frame).run(output)


if __name__ == "__main__":
    main()
