import io
import json
import os
import tempfile
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Import your detector functions
from motiondetection2 import human_detection_yolo
from object_detector2 import object_detector

app = FastAPI(title="Army Detection Backend")

# Path to append per-frame logs (JSON Lines)
RESULTS_LOG = "results_log.jsonl"

# Helper: read uploaded image bytes -> OpenCV BGR frame
def read_imagefile(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

# Helper: save a per-frame record to results_log.jsonl
def append_result_log(record: dict):
    try:
        with open(RESULTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

# Utility: timestamp
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...), detailed: Optional[bool] = False):
    contents = await file.read()
    try:
        frame = read_imagefile(contents)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image uploaded")

    try:
        humans = human_detection_yolo(frame)
        objects = object_detector(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

    # Build the record
    record = {
        "timestamp": now_iso(),
        "humans": humans,
        "objects": objects.get("objects") if isinstance(objects, dict) else objects,
        "clothing_summary": {}  # removed clothing detection
    }

    append_result_log(record)
    return JSONResponse(record)


@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...), sample_rate: int = 10):
    if sample_rate < 1:
        raise HTTPException(status_code=400, detail="sample_rate must be >= 1")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open uploaded video")

        frame_idx = 0
        processed = 0
        aggregate = {
            "total_frames": 0,
            "sampled_frames": 0,
            "humans_total": 0,
            "objects_counts": {}
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            aggregate["total_frames"] += 1

            if (frame_idx % sample_rate) != 0:
                continue

            processed += 1
            try:
                humans = human_detection_yolo(frame)
                objects = object_detector(frame)
            except Exception:
                continue

            aggregate["sampled_frames"] = processed

            if isinstance(humans, dict) and "person_count" in humans:
                aggregate["humans_total"] += int(humans["person_count"])
            if isinstance(objects, dict) and "objects" in objects:
                for k, v in objects["objects"].items():
                    aggregate["objects_counts"][k] = aggregate["objects_counts"].get(k, 0) + int(v)

            record = {
                "timestamp": now_iso(),
                "frame_index": frame_idx,
                "humans": humans,
                "objects": objects.get("objects") if isinstance(objects, dict) else objects,
                "clothing_summary": {}
            }
            append_result_log(record)

        cap.release()

        summary = {
            "processed_frames": processed,
            "total_frames": aggregate["total_frames"],
            "humans_total": aggregate["humans_total"],
            "objects_counts": aggregate["objects_counts"],
            "clothing_counts": {}  # removed clothing
        }
        return JSONResponse(summary)

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.get("/ping")
async def ping():
    return {"status": "ok", "timestamp": now_iso()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
