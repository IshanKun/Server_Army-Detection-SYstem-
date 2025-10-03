

from pathlib import Path
from datetime import datetime
import json
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    raise SystemExit("Ultralytics (yolov8) required. Install: pip install ultralytics")

try:
    import mediapipe as mp
except Exception:
    raise SystemExit("mediapipe required. Install: pip install mediapipe")

# Optional emotion detector: try fer package, otherwise fallback to 'unknown'
try:
    from fer import FER
    _FER_AVAILABLE = True
except Exception:
    _FER_AVAILABLE = False

# CONFIG
MODEL_PATH = "yolov8n.pt"
OUTPUT_JSON = "sitrep_log.json"
CONF_THRESH = 0.35
IOU_MATCH_THRESH = 0.3
DISAPPEAR_FRAMES = 30
WANTED_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "laptop", "cell phone", "knife", "gun", "airpods", "charger"}
SITREP_INTERVAL_SECONDS = 10  # periodic aSITREP generation

# UTILITIES
def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))

def load_json(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return []

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter/union if union>0 else 0

# COLOR DETECTION: map avg HSV to basic colors
def dominant_color_name(img, bbox):
    x1,y1,x2,y2 = [int(max(0,min(img.shape[1]-1,v))) for v in bbox]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_mean = int(hsv[:,:,0].mean())
    s_mean = int(hsv[:,:,1].mean())
    v_mean = int(hsv[:,:,2].mean())
    if v_mean < 50:
        return "black"
    if s_mean < 30 and v_mean > 200:
        return "white"
    # hue ranges approximation
    if h_mean < 10 or h_mean > 160:
        return "red"
    if 10 <= h_mean < 25:
        return "orange"
    if 25 <= h_mean < 35:
        return "yellow"
    if 35 <= h_mean < 85:
        return "green"
    if 85 <= h_mean < 125:
        return "blue"
    if 125 <= h_mean < 160:
        return "purple"
    return "unknown"

# EMOTION DETECTION: using FER if available, otherwise placeholder
_emotion_detector = FER(mtcnn=True) if _FER_AVAILABLE else None

def detect_emotion(frame, bbox):
    if not _FER_AVAILABLE:
        return "unknown"
    x1,y1,x2,y2 = [int(v) for v in bbox]
    h,w = frame.shape[:2]
    x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    result = _emotion_detector.detect_emotions(crop)
    if not result:
        return "unknown"
    emotions = result[0].get("emotions", {})
    if not emotions:
        return "unknown"
    return max(emotions.items(), key=lambda t: t[1])[0]

# ACTIVITY: simple pose-based motion detection (running vs walking vs standing)
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def pose_keypoints(frame, bbox=None):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose_detector.process(img)
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    pts = [(p.x, p.y, p.z, p.visibility) for p in lm]
    return pts

def activity_from_keypoints(prev_pts, cur_pts, frame_time_delta):
    if not prev_pts or not cur_pts:
        return "unknown"
    # use hip (landmark 24 or 23) and ankle velocities to guess running/walking
    try:
        hip_idx = 24
        l_ankle = 27
        r_ankle = 28
        ph = np.array([prev_pts[hip_idx][0], prev_pts[hip_idx][1]])
        ch = np.array([cur_pts[hip_idx][0], cur_pts[hip_idx][1]])
        pv = np.linalg.norm(ch - ph)/frame_time_delta
        # ankles
        pa = np.array([prev_pts[l_ankle][0], prev_pts[l_ankle][1]])
        ca = np.array([cur_pts[l_ankle][0], cur_pts[l_ankle][1]])
        av = np.linalg.norm(ca - pa)/frame_time_delta
        # heuristics
        if pv > 0.02 or av > 0.03:
            return "running"
        if pv > 0.008:
            return "walking"
        return "standing"
    except Exception:
        return "unknown"

# MAIN: model load
model = YOLO(MODEL_PATH)
try:
    names_map = {int(k): v for k,v in model.model.names.items()}
except Exception:
    names_map = model.names if hasattr(model,"names") else {}

# TRACKER: simple IoU-based track management
next_track_id = 0
tracks = {}  # id -> {label,bbox,last_seen,first_seen,color,emotion,activity}
log = load_json(OUTPUT_JSON)
last_sitrep_time = time.time()
prev_pose_pts = None
prev_frame_time = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

frame_idx = 0
try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_h, frame_w = frame.shape[:2]
        results = model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < CONF_THRESH:
                    continue
                cls_id = int(box.cls)
                label = names_map.get(cls_id, str(cls_id))
                xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box.xyxy, 'cpu') else list(box.xyxy)
                x1,y1,x2,y2 = [float(v) for v in xyxy]
                detections.append((label, conf, [x1,y1,x2,y2]))

        # filter by WANTED; keep others for situational awareness
        detections = [d for d in detections if d[0] in WANTED_CLASSES]

        # match detections to tracks
        assigned = [False]*len(detections)
        for tid, tr in list(tracks.items()):
            tr['matched'] = False
        for i, (label,conf,bbox) in enumerate(detections):
            best_tid = None; best_iou = 0
            for tid, tr in tracks.items():
                if tr['label'] != label:
                    continue
                iouv = box_iou(tr['bbox'], bbox)
                if iouv > best_iou:
                    best_iou = iouv; best_tid = tid
            if best_iou >= IOU_MATCH_THRESH and best_tid is not None:
                tr = tracks[best_tid]
                tr['bbox'] = bbox
                tr['last_seen'] = frame_idx
                tr['conf'] = conf
                tr['matched'] = True
                assigned[i] = True

        # create tracks for unassigned detections
        for i, assigned_flag in enumerate(assigned):
            if assigned_flag:
                continue
            label, conf, bbox = detections[i]
            tid = next_track_id; next_track_id += 1
            color = dominant_color_name(frame, bbox)
            emotion = detect_emotion(frame, bbox) if label=="person" else "n/a"
            tracks[tid] = {
                'label': label,
                'bbox': bbox,
                'first_seen_frame': frame_idx,
                'last_seen': frame_idx,
                'entry_time': now_iso(),
                'color': color,
                'emotion': emotion,
                'conf': conf,
                'activity': "unknown"
            }

        # remove stale tracks and log exit times (for SITREP statefulness)
        to_remove = []
        for tid, tr in list(tracks.items()):
            if frame_idx - tr['last_seen'] > DISAPPEAR_FRAMES:
                tr['exit_time'] = now_iso()
                to_remove.append(tid)
        for tid in to_remove:
            del tracks[tid]

        # pose + activity estimation
        cur_pose_pts = pose_keypoints(frame, None)
        dt = 1/30.0 if prev_frame_time is None else max(1e-3, time.time()-prev_frame_time)
        if cur_pose_pts:
            act = activity_from_keypoints(prev_pose_pts, cur_pose_pts, dt)
            # assign activity to nearby person tracks (approx: center within bbox)
            for tid, tr in tracks.items():
                if tr['label']!='person':
                    continue
                x1,y1,x2,y2 = tr['bbox']
                cx = (x1+x2)/2/frame_w; cy = (y1+y2)/2/frame_h
                try:
                    # compute distance to pose mid-hip
                    hip = cur_pose_pts[24]
                    hx,hy = hip[0], hip[1]
                    if abs(hx-cx) < 0.25 and abs(hy-cy) < 0.25:
                        tr['activity'] = act
                except Exception:
                    pass
        prev_pose_pts = cur_pose_pts
        prev_frame_time = time.time()

        # Build current situational snapshot (list of objects summary)
        snapshot = []
        counts = {}
        for tid, tr in tracks.items():
            snapshot.append({
                'id': tid,
                'label': tr['label'],
                'bbox': [float(v) for v in tr['bbox']],
                'color': tr.get('color','unknown'),
                'emotion': tr.get('emotion','unknown'),
                'activity': tr.get('activity','unknown'),
                'entry_time': tr.get('entry_time')
            })
            counts[tr['label']] = counts.get(tr['label'], 0) + 1

        # generate SITREP if change in snapshot or periodic interval
        last_snapshot = log[-1]['situation']['detected_objects'] if log else None
        simple_list = sorted([s['label'] for s in snapshot])
        if last_snapshot is None or sorted(last_snapshot) != simple_list or (time.time()-last_sitrep_time) > SITREP_INTERVAL_SECONDS:
            sitrep = {
                'report_id': f"SITREP-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}",
                'timestamp': now_iso(),
                'location': 'unassigned',
                'situation': {
                    'summary': 'Automated detection snapshot',
                    'detected_objects': simple_list
                },
                'enemy_activity': {
                    'description': '',
                    'movements': [ { 'id': s['id'], 'label': s['label'], 'activity': s['activity'] } for s in snapshot ]
                },
                'human_analysis': {
                    'civilians': sum(1 for s in snapshot if (s['label']=='person' and s['emotion']!='aggression' and s['label']=='person')),
                    'hostiles': sum(1 for s in snapshot if s['label'] in ('gun','knife')),
                    'emotions': [ { 'id': s['id'], 'emotion': s['emotion'] } for s in snapshot if s['label']=='person' ]
                },
                'logistics': {},
                'command_signal': { 'communications': 'unknown' }
            }
            log.append({ 'situation': sitrep['situation'], 'report': sitrep })
            save_json(OUTPUT_JSON, log)
            last_sitrep_time = time.time()

        # Draw on frame: timestamp and object boxes
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        cv2.putText(frame, ts, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        for tid,tr in tracks.items():
            x1,y1,x2,y2 = [int(v) for v in tr['bbox']]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
            label = f"ID{tid}:{tr['label']} {tr.get('activity','')[:6]} {tr.get('emotion','')[:6]}"
            cv2.putText(frame, label, (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

        cv2.imshow('SITREP Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release(); cv2.destroyAllWindows()
    save_json(OUTPUT_JSON, log)
    pose_detector.close()
