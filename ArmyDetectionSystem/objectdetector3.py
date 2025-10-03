import cv2
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
OUTPUT_JSON = "frame_log.json"
CONF_THRESH = 0.35
WANTED = {"person", "car", "laptop", "cell phone", "knife", "gun", "airpods", "charger"}

# ----- Helpers -----
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def load_json(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return []

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))

# ----- Load model -----
model = YOLO(MODEL_PATH)
names_map = model.model.names if hasattr(model, "model") else model.names

# ----- Initialize -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

log_records = load_json(OUTPUT_JSON)
current_record = None
prev_objects = []

frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Detect
        results = model(frame)
        objects_in_frame = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < CONF_THRESH:
                    continue
                cls_id = int(box.cls)
                label = names_map.get(cls_id, str(cls_id))
                if label in WANTED:
                    objects_in_frame.append(label)

        # Sort for comparison consistency
        objects_in_frame.sort()

        # If objects changed -> close old record and start new one
        if objects_in_frame != prev_objects:
            if current_record:
                current_record["end_time"] = now_iso()
                log_records.append(current_record)
                save_json(OUTPUT_JSON, log_records)

            current_record = {
                "frame": frame_idx,
                "start_time": now_iso(),
                "end_time": None,
                "objects": objects_in_frame
            }
            print(f"New state at frame {frame_idx}: {objects_in_frame}")

        prev_objects = objects_in_frame

        # Overlay time + detections
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < CONF_THRESH:
                    continue
                cls_id = int(box.cls)
                label = names_map.get(cls_id, str(cls_id))
                if label in WANTED:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
                    cv2.putText(frame, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

        cv2.imshow("Frame-based Logging", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    
    if current_record:
        current_record["end_time"] = now_iso()
        log_records.append(current_record)
        save_json(OUTPUT_JSON, log_records)
        print("Final record closed.")
