from ultralytics import YOLO
import cv2

# Load YOLO model once at import (saves time)
model = YOLO("yolov8n.pt")

def human_detection_yolo(frame):
    results = model(frame, classes=[0])  # detect only 'person'
    count = len(results[0].boxes)        # number of detected persons
    return {"person_count": count}


if __name__ == "__main__":
    # Load an image just to test function
    frame = cv2.imread("Untitled.jpeg")  # replace with your test image
    result = human_detection_yolo(frame)
    print(result)
