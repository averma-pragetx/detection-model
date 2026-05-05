import argparse
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
from object_detection import load_yolo, draw_custom_box

parser = argparse.ArgumentParser(description="YOLO Flask Backend")
parser.add_argument("--model", type=str, default="yolo26n.pt", help="Path to YOLO model")
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
