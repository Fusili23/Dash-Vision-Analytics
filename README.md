# Dash Vision Analytics

Advanced trajectory prediction system for dashcam footage using YOLOv11 and ByteTrack.

## Overview

This project implements real-time trajectory prediction for vehicles and pedestrians in dashcam videos. It compensates for ego-vehicle motion using optical flow and incorporates environmental context like traffic lights and road boundaries to generate more accurate predictions.

## Project Structure

```
Dash-Vision-Analytics/
├── src/                          # Core modules
│   ├── trajectory_predictor.py   # Prediction models (CVM, Kalman Filter)
│   ├── ego_motion.py             # Camera motion estimation
│   ├── context_aware_predictor.py # Context-based prediction
│   ├── semantic_zones.py         # Road/sidewalk segmentation
│   └── bev_transformer.py        # Bird's eye view transformation
├── examples/                     # Runnable demos
│   ├── test_with_prediction.py   # Full system demo
│   └── test.py                   # Basic tracking only
└── docs/                         # Technical documentation
```

## Methods and Models

**Object Detection and Tracking**
- YOLOv11 (nano variant) for real-time object detection
- ByteTrack for multi-object tracking across frames

**Trajectory Prediction**
- Constant Velocity Model for baseline predictions
- Kalman Filter for noise reduction and uncertainty estimation

**Ego-Motion Compensation**
- Dense Optical Flow (Farneback method) to estimate camera movement
- Background pixel analysis to separate camera motion from object motion
- Formula: V_actual = V_perceived - V_ego

**Context-Aware Prediction**
- Traffic light detection using color-based HSV filtering
- Semantic zone masking for road/sidewalk boundaries
- Intent-based model switching (e.g., vehicles stop at red lights)
- Multi-modal predictions for ambiguous situations

**Bird's Eye View Transformation**
- Perspective transformation to top-down view
- Metric velocity calculation (m/s, km/h)
- Eliminates perspective distortion in speed estimation

## Installation

```bash
git clone https://github.com/Fusili23/Dash-Vision-Analytics.git
cd Dash-Vision-Analytics
pip install ultralytics opencv-python numpy
```

## Usage

Run the complete demo:
```bash
python examples/test_with_prediction.py
```

Configuration options in `examples/test_with_prediction.py`:
- Line 34: Set video file path
- Line 83: Adjust prediction horizon (default: 3 seconds)
- Lines 87-90: Enable/disable features

Basic tracking without prediction:
```bash
python examples/test.py
```

## Code Examples

**Basic Prediction**
```python
from src.trajectory_predictor import KalmanFilterPredictor

predictor = KalmanFilterPredictor()
for frame in frames:
    predictor.update(track_id, position)
    
predictions = predictor.predict(track_id, num_steps=60)
```

**Ego-Motion Compensation**
```python
from src.ego_motion import EgoMotionEstimator, RelativeVelocityTracker

estimator = EgoMotionEstimator(flow_quality='medium')
tracker = RelativeVelocityTracker()

ego_velocity = estimator.estimate_ego_motion(frame)
tracker.update(track_id, position, ego_velocity)
actual_velocity = tracker.get_actual_velocity(track_id)
```

**Context-Aware Prediction**
```python
from src.context_aware_predictor import ContextAwarePredictor, EnvironmentalContext

predictor = ContextAwarePredictor()
context = EnvironmentalContext(traffic_lights=lights, zone_type=ROAD)

result = predictor.predict_with_context(track_id, num_steps=90, context=context)
# Returns: {'primary': trajectory, 'alternative': trajectory, 'intent': 'STOP'}
```

## Technical Details

**Key Equations**

Ego-motion compensation:
```
V_actual = V_perceived - V_ego
```

BEV velocity to metric:
```
V_m/s = (V_BEV / pixels_per_meter) * FPS
Speed_km/h = Speed_m/s * 3.6
```

**Performance**
- CPU (1920x1080): 8-12 FPS
- CPU (640x480): 25-30 FPS
- GPU: 30-60+ FPS

**Prediction Accuracy**
- 1 second horizon: 5-10 pixel MAE
- 3 second horizon: 15-25 pixel MAE
- Context-aware reduces errors by 20-30% at intersections

## Features

- Real-time object tracking with persistent IDs
- Camera motion compensation via optical flow
- Traffic light state detection
- Semantic zone constraints
- BEV transformation for metric calculations
- Multi-modal predictions with uncertainty visualization
- Beginner-friendly code with extensive comments

## Documentation

All source files include detailed line-by-line comments explaining:
- Python syntax and operators
- Algorithm implementation
- Mathematical formulas
- Design decisions

See `docs/PROJECT_DOCUMENTATION.md` for complete technical documentation.

## Dependencies

- Python 3.8+
- ultralytics (YOLOv11)
- opencv-python
- numpy

## Limitations and Future Work

This implementation uses simplified methods:
- Color-based traffic light detection (not robust)
- Manual BEV calibration required
- No rotation compensation in ego-motion
- CPU-based optical flow (slow)
- No learned intent models

Potential improvements:
- CNN-based traffic light classification
- Automatic BEV calibration
- GPU-accelerated optical flow
- Deep learning trajectory prediction
- Multi-camera fusion

## Note

This project evolved significantly beyond its original scope. What started as a simple trajectory tracking exercise expanded into a comprehensive system with ego-motion compensation, context-aware prediction, BEV transformations, and multi-modal uncertainty modeling. While the implementation demonstrates various computer vision and prediction techniques, it went much further than initially intended.

## License

Educational project.

## Author

Fusili23

## Acknowledgments

- Ultralytics (YOLOv11)
- ByteDance (ByteTrack)
- OpenCV Community
