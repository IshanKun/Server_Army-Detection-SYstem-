
from ultralytics import YOLO
import cv2
import time

# we are using the pretrained yolov8 model for object detection
model = YOLO("yolov8n.pt")  


# these objects will be detected
WANTED = { "car", "laptop", "cell phone", "book", "bottle", "handbag",'gun', 'knife'}  


def names_from_model(model):
  
    return {int(k): v for k, v in model.model.names.items()} if hasattr(model, "model") else model.names

def human_readable(results, names_map):
    
    out = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else int(box.cls)
            conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else float(box.conf)
            xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box, "xyxy") else box.xyxy
            label = names_map.get(cls, str(cls))
            out.append((label, conf, xyxy))
    return out

def run_webcam():
    names_map = names_from_model(model)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model(frame)

       
        detections = human_readable(results, names_map)
        for label, conf, (x1, y1, x2, y2) in detections:
            if label in WANTED and conf > 0.35:  # adjust threshold
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Object Detection (filtered)", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
