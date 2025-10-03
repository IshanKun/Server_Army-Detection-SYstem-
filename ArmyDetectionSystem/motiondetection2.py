from ultralytics import YOLO
import cv2

def human_detection_yolo():
    model = YOLO("yolov8n.pt")  # small YOLOv8 model
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0])  # class 0 = person
        annotated = results[0].plot()

        cv2.imshow("YOLO Human Detection", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     human_detection_yolo()
