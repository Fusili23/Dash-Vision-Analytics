# How to Run

## Quick Start

From the project root directory:

```bash
python examples/test_with_prediction.py
```

## Prerequisites

Make sure you have installed dependencies:

```bash
pip install ultralytics opencv-python numpy
```

## Configuration

Edit `examples/test_with_prediction.py` to configure:

**Line 34** - Video file path:
```python
input_path = r"C:\path\to\your\video.mp4"
```

**Line 83** - Prediction time (seconds):
```python
PREDICT_SECONDS = 3.0  # Change to 2.0, 5.0, etc.
```

**Lines 87-90** - Feature toggles:
```python
ENABLE_EGO_MOTION = True        # Camera motion compensation
ENABLE_CONTEXT_AWARE = True     # Traffic light & zone detection
ENABLE_BEV_CALCULATION = True   # Real-world speed (km/h)
SHOW_OPTICAL_FLOW = False       # Show optical flow visualization
```

## Controls

- Press 'q' to quit
- Window shows real-time predictions overlaid on video

## Alternative: Basic Tracking Only

For basic object tracking without prediction:

```bash
python examples/test.py
```

## Troubleshooting

**CUDA Error**: Install PyTorch with CUDA support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Module Not Found**: Make sure you're in the project root directory

**Slow Performance**: Disable features or reduce resolution
```python
ENABLE_BEV_CALCULATION = False
```
