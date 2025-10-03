import cv2
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from ultralytics import YOLO

# ---------------- Settings ----------------
MODEL_PATH = "yolov8n.pt"
CLOTH_CLASSIFIER_PATH = "clothing_classifier.pth"
OUTPUT_JSON = "frame_log_with_clothing.json"
CONF_THRESH = 0.35
WANTED_PERSON_LABELS = {"person"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLOTHING_CLASSES = ["army", "doctor", "civilian", "other"]

clf_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------- Load YOLO ----------------
detector = YOLO(MODEL_PATH)
names_map = detector.model.names if hasattr(detector, "model") else detector.names

# ---------------- Load classifier ----------------
classifier = None
try:
    import torchvision.models as models
    clf = models.resnet18(pretrained=False)
    clf.fc = nn.Linear(clf.fc.in_features, len(CLOTHING_CLASSES))
    clf.load_state_dict(torch.load(CLOTH_CLASSIFIER_PATH, map_location=DEVICE))
    clf.to(DEVICE).eval()
    classifier = clf
    print("✅ Loaded clothing classifier:", CLOTH_CLASSIFIER_PATH)
except Exception as e:
    print("⚠️ No classifier found, using fallback heuristic (Reason:", e, ")")
    classifier = None

# ---------------- Helpers ----------------
def heuristic_classify(crop_bgr):
    """Fallback heuristic: detect based on color"""
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    h_med, s_med = int(np.median(h)), int(np.median(s))
    if v.mean() > 200 and s.mean() < 40:
        return "doctor", 0.6
    if 40 <= h_med <= 90 and s_med > 50:
        return "army", 0.5
    return "civilian", 0.5

def classify_crop(crop):
    """Return (label, score)"""
    if classifier is None:
        return heuristic_classify(crop)
    img_t = clf_transforms(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = classifier(img_t)
        probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return CLOTHING_CLASSES[idx], float(probs[idx])

def now_iso():
    return datetime.now().isoformat(timespec="seconds")
def load_json(p): return json.loads(Path(p).read_text()) if Path(p).exists() else []
def save_json(p, data): Path(p).write_text(json.dumps(data, indent=2))

# ---------------- Main function ----------------
def clothing_detection(cam_index=0):
    log_records = load_json(OUTPUT_JSON)
    current_record = None
    prev_objects_summary = None

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("❌ ERROR: Cannot open webcam")

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = detector(frame)
            objects_in_frame = []

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf)
                    if conf < CONF_THRESH:
                        continue
                    cls_id = int(box.cls)
                    label = names_map.get(cls_id, str(cls_id))
                    if label not in WANTED_PERSON_LABELS:
                        continue

                    # Crop for classification
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                    h,w = frame.shape[:2]
                    pad = int(0.05 * max(x2-x1, y2-y1))
                    x1c, y1c = max(0, x1-pad), max(0, y1-pad)
                    x2c, y2c = min(w-1, x2+pad), min(h-1, y2+pad)
                    crop = frame[y1c:y2c, x1c:x2c].copy()
                    if crop.size == 0:
                        continue

                    category, score = classify_crop(crop)
                    objects_in_frame.append({
                        "label": label,
                        "category": category,
                        "conf": score,
                        "bbox": [x1,y1,x2,y2]
                    })

                    # Draw box + category
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
                    cv2.putText(frame, f"{label} ({category})", (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

            # Compare with previous
            summary = sorted([f"{o['label']}:{o['category']}" for o in objects_in_frame])
            if summary != prev_objects_summary:
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
                print(f"Frame {frame_idx} changed -> {summary}")

            prev_objects_summary = summary

            # Overlay timestamp
            cv2.putText(frame, now_iso(), (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)

            cv2.imshow("Clothing classification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if current_record:
            current_record["end_time"] = now_iso()
            log_records.append(current_record)
            save_json(OUTPUT_JSON, log_records)
        print("✅ Saved:", OUTPUT_JSON)

# Example run
# if __name__ == "__main__":
#     clothing_detection()
