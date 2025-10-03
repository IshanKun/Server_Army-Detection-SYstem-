import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from datetime import datetime
from ultralytics import YOLO

# ---------------- Settings ----------------
MODEL_PATH = "yolov8n.pt"
CLOTH_CLASSIFIER_PATH = "clothing_classifier.pth"
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

detector = YOLO(MODEL_PATH)
names_map = detector.model.names if hasattr(detector, "model") else detector.names

classifier = None
try:
    import torchvision.models as models
    clf = models.resnet18(pretrained=False)
    clf.fc = nn.Linear(clf.fc.in_features, len(CLOTHING_CLASSES))
    clf.load_state_dict(torch.load(CLOTH_CLASSIFIER_PATH, map_location=DEVICE))
    clf.to(DEVICE).eval()
    classifier = clf
except Exception as e:
    print(" No classifier found, using heuristic fallback:", e)

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def heuristic_classify(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    h_med, s_med, v_med = int(np.median(h)), int(np.median(s)), int(np.median(v))

 
    if 35 <= h_med <= 90 and s_med > 60:
        return "army", 0.6
    # Detect doctor: mostly bright/white
    if v_med > 200 and s_med < 40:
        return "doctor", 0.6
    return "civilian", 0.5
def classify_crop(crop):
    if classifier is None:
        return heuristic_classify(crop)
    img_t = clf_transforms(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = classifier(img_t)
        probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return CLOTHING_CLASSES[idx], float(probs[idx])

def detect_clothing_summary(frame):

    results = detector(frame)
    category_counts = {}

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

            category, _ = classify_crop(crop)
            category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "timestamp": now_iso(),
        "summary": category_counts
    }


if __name__ == "__main__":
    test_img = cv2.imread("1.jpeg")
    print(detect_clothing_summary(test_img))
