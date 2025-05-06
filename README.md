# Hand Tracker

A Python-based hand and finger tracking application using OpenCV and MediaPipe.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

- Press 'q' to quit the application

## Project Structure

```
handTracker/
├── src/              # Source code directory
├── requirements.txt  # Project dependencies
├── main.py          # Entry point
└── README.md        # Documentation
```

## Features

Phase 1 (Current):
- Basic webcam capture and display

Future Phases:
- Hand detection
- Finger tracking
- Landmark detection
- Gesture recognition 