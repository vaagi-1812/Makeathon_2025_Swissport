# Airport Turnaround Object Detection

A YOLO11-based object detection project for monitoring airport turnaround operations. Detects key vehicles and equipment in video footage to help track ground handling activity.

## Detected Classes

| Class | Description |
|---|---|
| `bridge_connected` | Jet bridge connected to aircraft |
| `cleaning_crew_vehicle` | Cleaning crew vehicle on apron |
| `luggage_vehicle` | Luggage/baggage handling vehicle |

## Project Structure

```
Object_detection/
├── turnaround clip.mp4   # Source video footage
├── turnaround.py         # Frame extraction & dataset creation
├── train_model.py        # YOLO11 model training script
├── yolo11n.pt            # YOLO11 nano pretrained weights
├── dataset/              # Generated training dataset (created by turnaround.py)
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── data.yaml
└── runs/detect/          # Training outputs (weights, plots, metrics)
    └── train/
        └── weights/
            ├── best.pt   # Best model checkpoint
            └── last.pt   # Latest model checkpoint
```

## Workflow

### Step 1 — Extract frames and build dataset

```bash
cd Object_detection
python turnaround.py
```

This extracts one frame every 4 seconds from `turnaround clip.mp4`, splits them 80/20 into train/val, and generates `dataset/data.yaml`.

After extraction, **manually annotate** the frames in `dataset/images/` using a tool like [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com/), then export labels in YOLO format to `dataset/labels/`.

### Step 2 — Train the model

```bash
python train_model.py
```

Trains YOLO11n for 100 epochs at 640px resolution. Automatically uses GPU if available (optimised for RTX 3070 with `batch=16`), falls back to CPU otherwise.

Training artifacts are saved to `runs/detect/train/`.

## Requirements

```bash
pip install ultralytics opencv-python torch
```

GPU training requires a CUDA-compatible PyTorch installation. See [PyTorch Get Started](https://pytorch.org/get-started/locally/) to install the correct version for your CUDA version.

## Model

Built on [YOLO11n](https://docs.ultralytics.com/models/yolo11/) (nano variant) — a lightweight, fast model suitable for real-time inference. The best trained weights are saved at `runs/detect/train/weights/best.pt`.
