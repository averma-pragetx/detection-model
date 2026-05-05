# YOLOv26 Object Detection Dashboard

A real-time object detection application featuring a Python (Flask) backend and a React (Vite) frontend.

## Project Structure

- `backend/`: Python backend using Flask and Ultralytics YOLOv26.
  - `app.py`: Flask server for real-time video streaming.
  - `object_detection.py`: Core detection logic.
  - `train.py`: Script for training custom models.
  - `yolo26n.pt`: Pre-trained YOLOv26 model weights.
  - `requirements.txt`: Python dependencies.
- `frontend/`: React frontend built with Vite.
  - `src/`: React components and logic.

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the backend server:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

## Usage

Once both servers are running, open your browser to the URL provided by Vite (usually `http://localhost:5173`). The dashboard will display a real-time video feed from your webcam with object detection overlays.