import cv2
import numpy as np
from ultralytics import YOLO
import sys
import argparse

def load_yolo(model_name="yolo26n.pt"):
    """
    Load YOLOv26 model using Ultralytics.
    """
    try:
        print(f"Loading {model_name} model...")
        model = YOLO(model_name)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced YOLOv26 Object Detection")
    parser.add_argument("--model", type=str, default="yolo26n.pt", 
                        help="Model to use (e.g., yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt)")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold (0.0 to 1.0). Lower values detect more objects.")
    parser.add_argument("--imgsz", type=int, default=640, 
                        help="Input image size (e.g., 640, 1280). Larger sizes are better for small objects.")
    parser.add_argument("--device", type=str, default="", 
                        help="Device to run on (e.g., cpu, 0, 1). Leave empty for auto-detect.")
    parser.add_argument("--show-dims", action="store_true", default=True,
                        help="Show width and height of detected objects.")
    return parser.parse_args()

def draw_custom_box(frame, box, label, confidence, color, show_dims=True):
    """
    Draw a custom bounding box with optional dimensions.
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare text
    display_text = f"{label} {confidence:.2f}"
    if show_dims:
        display_text += f" ({w}x{h})"
        
    # Draw label background
    (tw, th), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    
    # Draw text
    cv2.putText(frame, display_text, (x1 + 5, y1 - 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    args = parse_args()
    
    print("Initializing YOLOv26 Object Detection...")
    model = load_yolo(args.model)
    
    print("Accessing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Detection started with model: {args.model}")
    print(f"Parameters: Confidence={args.conf}, ImageSize={args.imgsz}, ShowDims={args.show_dims}")
    print("Press 'q' to quit.")

    # Generate random colors for classes
    colors = np.random.uniform(0, 255, size=(1000, 3))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
                
            # Perform detection
            results = model(frame, 
                            conf=args.conf, 
                            imgsz=args.imgsz, 
                            device=args.device if args.device else None,
                            stream=True)
            
            # Process results
            for r in results:
                # Iterate through detected boxes
                for box in r.boxes:
                    # Get class info
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    conf = float(box.conf[0])
                    coords = box.xyxy[0]
                    
                    # Draw custom annotations
                    color = tuple(map(int, colors[cls_id]))
                    draw_custom_box(frame, coords, label, conf, color, show_dims=args.show_dims)
            
            # Display the result
            window_title = f"YOLOv26 [{args.model}] (Press 'q' to quit)"
            cv2.imshow(window_title, frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"An error occurred during detection: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    main()
