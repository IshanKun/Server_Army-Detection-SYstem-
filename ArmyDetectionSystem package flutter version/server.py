from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from  motiondetection2 import human_detection_yolo
from object_detector2 import object_detector
from  object_DirectorClothingbased import detect_clothing_summary

app = FastAPI(title="Army Detection Backend")

# Convert uploaded file to OpenCV frame
def read_imagefile(file_bytes) -> np.ndarray:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Receives a single frame from Flutter, runs all detection modules,
    and returns combined JSON.
    """
    contents = await file.read()
    frame = read_imagefile(contents)

    # Run all three modules
    humans = human_detection_yolo(frame)         # {'person_count': 5}
    objects = object_detector(frame)             # {'person': 5, 'car': 1}
    clothing = detect_clothing_summary(frame)   # {'summary': {'army': 3, 'civilian': 2}}

    result = {
        "timestamp": clothing["timestamp"],
        "humans": humans,
        "objects": objects["objects"],           # object_detector returns {'objects': {...}}
        "clothing_summary": clothing["summary"]
    }

    return result

# Run the server with:
# uvicorn server:app --reload --host 0.0.0.0 --port 8000
