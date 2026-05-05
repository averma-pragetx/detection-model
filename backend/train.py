from ultralytics import YOLO
import argparse
import sys
import os

def train_model(data_yaml, model_size="yolo26n.pt", epochs=100, imgsz=640, batch=16):
    """
    Train a YOLOv26 model on a custom dataset.
    """
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found. Please create it first.")
        return

    # Load a pre-trained model to start training from
    print(f"Loading base model: {model_size}...")
    model = YOLO(model_size)

    # Start training
    print(f"Starting training on {data_yaml} for {epochs} epochs...")
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name="custom_yolo26",
            # device=0  # Uncomment to use first GPU
        )
        print("Training completed successfully!")
        print(f"Results saved to: {results.save_dir}")
        print(f"Best model weights: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
    except Exception as e:
        print(f"An error occurred during training: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv26 on a custom dataset")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml file")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Base model (n, s, m, l, x)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
