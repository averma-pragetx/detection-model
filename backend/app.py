import argparse
import json
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
from object_detection import load_yolo, draw_custom_box

parser = argparse.ArgumentParser(description="YOLO Flask Backend")
parser.add_argument("--model", type=str, default="runs/detect/custom_yolo26/weights/best.pt", help="Path to YOLO model")
args = parser.parse_args()

app = Flask(__name__)
CORS(app)

model_name = args.model
model = load_yolo(model_name)
colors = np.random.uniform(0, 255, size=(1000, 3))

# Globals for asynchronous inference
latest_frame = None
latest_boxes = []
lock = threading.Lock()

def inference_loop():
    global latest_frame, latest_boxes
    while True:
        frame_copy = None
        with lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()
        
        if frame_copy is None:
            time.sleep(0.01)
            continue
            
        # Run inference
        results = model(frame_copy, conf=0.25, imgsz=640, stream=False)
        
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                boxes.append((coords, label, conf, cls_id))
                
        # Update boxes
        with lock:
            latest_boxes = boxes

def generate_frames():
    global latest_frame, latest_boxes
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Start the inference thread only once when streaming starts
    if not any(t.name == "InferenceThread" for t in threading.enumerate()):
        threading.Thread(target=inference_loop, name="InferenceThread", daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Mirror the video frame for a more natural feel
            frame = cv2.flip(frame, 1)
            
            with lock:
                latest_frame = frame.copy()
                current_boxes = list(latest_boxes)
                
            # Draw the latest available bounding boxes
            for (coords, label, conf, cls_id) in current_boxes:
                color = tuple(map(int, colors[cls_id]))
                draw_custom_box(frame, coords, label, conf, color, show_dims=True)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"status": "running", "model": model_name})

@app.route('/stats')
def stats():
    with lock:
        current_boxes = list(latest_boxes)
    
    unique_classes = set([label for (_, label, _, _) in current_boxes])
    return jsonify({
        "detections_count": len(current_boxes),
        "unique_classes": list(unique_classes)
    })

def detections_stream():
    """
    Generator that yields detection statistics as Server-Sent Events.
    """
    last_sent_data = None
    while True:
        with lock:
            current_boxes = list(latest_boxes)
        
        # Create a frequency map of detections
        counts = {}
        for _, label, _, _ in current_boxes:
            counts[label] = counts.get(label, 0) + 1
        
        # Create a formatted summary string similar to console output
        summary_parts = []
        for label in sorted(counts.keys()):
            count = counts[label]
            summary_parts.append(f"{count} {label}{'s' if count > 1 else ''}")
        
        summary = ", ".join(summary_parts)
        
        current_data = {
            "detections_count": len(current_boxes),
            "summary": summary if summary else "No detections",
            "unique_classes": sorted(list(counts.keys())),
            "all_detections": [{"label": l, "confidence": round(c, 2)} for _, l, c, _ in current_boxes]
        }
        
        # Only send if data has changed to reduce bandwidth
        if current_data != last_sent_data:
            yield f"data: {json.dumps(current_data)}\n\n"
            last_sent_data = current_data
            
        time.sleep(0.1)  # 10Hz update rate

@app.route('/detections')
def detections():
    return Response(detections_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
