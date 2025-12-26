# Dash Vision Analytics

고급 차량 및 보행자 궤적 예측 시스템 (Advanced Vehicle & Pedestrian Trajectory Prediction System)

## 🎯 프로젝트 개요 (Project Overview)

블랙박스 영상에서 차량과 보행자의 미래 경로를 예측하는 시스템입니다. YOLOv11과 ByteTrack를 사용한 객체 추적, 광학 흐름 기반 자차 움직임 보정, 신호등 및 도로 영역 인식을 통한 컨텍스트 기반 예측을 제공합니다.

A trajectory prediction system for dashcam footage featuring object tracking with YOLOv11 + ByteTrack, ego-motion compensation via optical flow, and context-aware prediction with traffic lights and semantic zones.

## ✨ 주요 기능 (Key Features)

- 🚗 **실시간 객체 추적** - YOLOv11 + ByteTrack
- 📹 **자차 움직임 보정** - 광학 흐름 (Optical Flow) 기반 Ego-motion 추정
- 🚦 **컨텍스트 인식 예측** - 신호등 상태, 도로/인도 구분
- 🗺️ **조감도 변환** - BEV (Bird's Eye View) 변환으로 실제 속도 계산
- 🎨 **불확실성 시각화** - 다중 모드 궤적 예측 (Multi-modal predictions)
- 📊 **초보자 친화적** - 모든 코드에 상세한 주석 포함

## 📁 프로젝트 구조 (Project Structure)

```
Dash-Vision-Analytics/
├── src/                          # 소스 코드 (Source code modules)
│   ├── __init__.py              # 패키지 초기화
│   ├── trajectory_predictor.py  # 궤적 예측 (CVM, Kalman Filter)
│   ├── ego_motion.py            # 자차 움직임 보정
│   ├── context_aware_predictor.py # 컨텍스트 인식 예측
│   ├── semantic_zones.py        # 도로/인도 마스크
│   └── bev_transformer.py       # 조감도 변환
│
├── examples/                     # 예제 스크립트 (Example scripts)
│   ├── test_with_prediction.py  # 통합 데모 (Main demo)
│   └── test.py                  # 기본 추적 (Basic tracking)
│
├── docs/                         # 문서 (Documentation)
│   ├── PROJECT_DOCUMENTATION.md # 기술 문서 (70+ pages)
│   └── COMMENTS_SUMMARY.md      # 주석 가이드
│
├── .gitignore                    # Git 제외 파일
└── README.md                     # 이 파일
```

## 🚀 빠른 시작 (Quick Start)

### 1. 설치 (Installation)

```bash
# Clone the repository
git clone https://github.com/Fusili23/Dash-Vision-Analytics.git
cd Dash-Vision-Analytics

# Install dependencies
pip install ultralytics opencv-python numpy
```

### 2. 실행 (Run)

```bash
# Run main demo with all features
python examples/test_with_prediction.py
```

**설정 변경 (Configuration):**
- `examples/test_with_prediction.py` 파일 57-60줄에서 기능 켜기/끄기
- Line 34: 비디오 파일 경로 변경
- Line 83: 예측 시간 조정 (기본 3초)

### 3. 주요 기능 토글 (Feature Toggles)

```python
ENABLE_EGO_MOTION = True        # 자차 움직임 보정
ENABLE_CONTEXT_AWARE = True     # 신호등 & 도로 인식  
ENABLE_BEV_CALCULATION = True   # 실제 속도 계산 (km/h)
SHOW_OPTICAL_FLOW = False       # 광학 흐름 시각화
```

## 📖 사용 예제 (Usage Examples)

### 기본 궤적 예측 (Basic Trajectory Prediction)

```python
from src.trajectory_predictor import KalmanFilterPredictor

predictor = KalmanFilterPredictor()

# Update with detections
for i in range(30):
    predictor.update(track_id=1, position=(x, y))

# Predict 60 frames (2 seconds @ 30fps)
predictions = predictor.predict(track_id=1, num_steps=60)
```

### Ego-Motion 보정 (Ego-Motion Compensation)

```python
from src.ego_motion import EgoMotionEstimator, RelativeVelocityTracker

ego_estimator = EgoMotionEstimator(history_size=5, flow_quality='medium')
velocity_tracker = RelativeVelocityTracker()

# Estimate camera movement
ego_velocity = ego_estimator.estimate_ego_motion(frame)

# Get ground-relative velocity
velocity_tracker.update(track_id, position, ego_velocity)
actual_velocity = velocity_tracker.get_actual_velocity(track_id)
```

### 컨텍스트 인식 예측 (Context-Aware Prediction)

```python
from src.context_aware_predictor import ContextAwarePredictor, EnvironmentalContext

predictor = ContextAwarePredictor()
context = EnvironmentalContext(
    traffic_lights=[red_light],
    zone_type=SemanticZone.ROAD,
    timestamp=frame_count
)

# Predict with context
result = predictor.predict_with_context(track_id, num_steps=90, context=context)
# Returns: {'primary': [...], 'alternative': [...], 'intent': 'STOP'}
```

## 🎓 초보자를 위한 가이드 (Beginner's Guide)

모든 Python 코드에 **초보자 친화적 주석**이 포함되어 있습니다:

- ✅ 모든 줄 설명
- ✅ 연산자 의미 (`@`, `%`, `::`, etc.)
- ✅ Python 개념 설명 (list comprehension, f-strings, etc.)
- ✅ 데이터 구조 설명
- ✅ 수학 공식 포함

📚 **상세 문서**: `docs/PROJECT_DOCUMENTATION.md` (70+ 페이지)

## 🔬 기술 스택 (Tech Stack)

- **객체 감지**: YOLOv11n (Ultralytics)
- **추적**: ByteTrack
- **예측**: Constant Velocity Model, Kalman Filter
- **Ego-Motion**: Dense Optical Flow (Farneback)
- **언어**: Python 3.13
- **라이브러리**: OpenCV, NumPy

## 📊 성능 (Performance)

| 설정 | 해상도 | CPU FPS | GPU FPS |
|------|--------|---------|---------|
| 최적화 | 640×480 | 25-30 | 60+ |
| 고품질 | 1920×1080 | 8-12 | 30+ |

## 🎯 활용 사례 (Use Cases)

- 🚗 자율주행 연구
- 📹 블랙박스 영상 분석
- 🚦 교통 흐름 연구
- ⚠️ 충돌 위험 예측
- 🏙️ 스마트시티 모니터링

## 📝 주요 알고리즘 (Key Algorithms)

### Ego-Motion 보정 공식

```
V_actual = V_perceived - V_ego
```

### BEV 속도 변환

```
V_m/s = (V_BEV / pixels_per_meter) × FPS
Speed_km/h = Speed_m/s × 3.6
```

### 정지 확률 (Stop Probability)

```
P_stop = f(distance, velocity, traffic_light_state)
```

## 🛠️ 향후 개발 계획 (Future Enhancements)

- [ ] GPU 가속 (CUDA optical flow)
- [ ] 회전 보정 (rotation compensation)
- [ ] 실시간 시맨틱 세그멘테이션
- [ ] 충돌 경고 시스템
- [ ] 다중 카메라 지원
- [ ] 학습 기반 의도 모델

## 🐛 문제 해결 (Troubleshooting)

**CUDA 에러**: PyTorch CUDA 12.4 설치
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**느린 성능**: `test_with_prediction.py`에서 기능 비활성화
```python
ENABLE_BEV_CALCULATION = False
```

**부정확한 BEV**: 캘리브레이션 실행
```python
from src.bev_transformer import calibrate_bev_interactive
calibrate_bev_interactive("your_video.mp4")
```

## 📄 라이선스 (License)

이 프로젝트는 교육 목적으로 만들어졌습니다.

## 👨‍💻 기여자 (Contributors)

- **Fusili23** - Initial work

## 🙏 감사의 말 (Acknowledgments)

- Ultralytics - YOLOv11
- ByteDance - ByteTrack
- OpenCV Community
- Gunnar Farneback - Optical Flow Algorithm

## 📧 연락처 (Contact)

GitHub: [@Fusili23](https://github.com/Fusili23)

---

⭐ 이 프로젝트가 도움이 되었다면 별표를 눌러주세요!

*Last Updated: 2025-12-26*
