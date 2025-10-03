from multiprocessing import Process
import sys

# Import your wrapped functions
from motiondetection2 import human_detection_yolo
from objectdetector3 import  object_detector_yolo
from object_DirectorClothingbased import clothing_detection

def menu():
    print("\n=== Detection System Main ===")
    print("1. Run Human Detection (YOLO)")
    print("2. Run Frame Logger")
    print("3. Run SITREP Detector")
    print("4. Run ALL simultaneously")
    print("5. Exit")

if __name__ == "__main__":
    while True:
        menu()
        choice = input("Enter choice: ").strip()

        if choice == "1":
            human_detection_yolo()

        elif choice == "2":
             object_detector_yolo()

        elif choice == "3":
            clothing_detection()

        elif choice == "4":
            # Run all at once in parallel
            p1 = Process(target=human_detection_yolo)
            p2 = Process(target=object_detector_yolo)
            p3 = Process(target=clothing_detection)

            p1.start(); p2.start(); p3.start()
            p1.join(); p2.join(); p3.join()

        elif choice == "5":
            sys.exit("Exiting...")
        else:
            print("Invalid choice. Try again.")
