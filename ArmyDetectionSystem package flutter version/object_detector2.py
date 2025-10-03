import cv2, json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.35
WANTED = {"person", "car", "laptop", "cell phone", "knife", "gun", "airpods", "charger"}

# ---- helpers ----
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

# Load YOLO model once globally
model = YOLO(MODEL_PATH)
names_map = model.model.names if hasattr(model, "model") else model.names

# ---- main function ----
def object_detector(frame):
    """
    Returns:
        dict: {
          "timestamp": "...",
          "objects": {"person": 3, "car": 2, ...}
        }
    """
    results = model(frame)
    objects_count = {}

    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf < CONF_THRESH:
                continue
            cls_id = int(box.cls)
            label = names_map.get(cls_id, str(cls_id))

            if label in WANTED:
                objects_count[label] = objects_count.get(label, 0) + 1

    return {
        "timestamp": now_iso(),
        "objects": objects_count
    }

# ---- quick test ----
if __name__ == "__main__":
    test_img = cv2.imread("test3.jpeg")  # replace with your test file
    print(object_detector(test_img))
