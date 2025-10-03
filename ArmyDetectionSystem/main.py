from multiprocessing import Process
from motiondetection2 import human_detection_yolo
from object_detector2 import run_object_detector
from objectdetector3 import run_object_detector3
from objectdetector4 import run_object_detector4

if __name__ == "__main__":
    p1 = Process(target=human_detection_yolo)
    p2 = Process(target=run_object_detector)
    p3 = Process(target=run_object_detector3)
    p4 = Process(target=run_object_detector4)

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
