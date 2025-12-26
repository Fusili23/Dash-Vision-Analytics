import cv2
from ultralytics import YOLO
import os

# CPU 모드에서는 CUDA 관련 환경 설정이 필요 없으므로 제거하거나 주석 처리합니다.
# os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0"
# os.environ['CUDA_MODULE_LOADING'] = "LAZY"

# 1. 모델 로드 (가장 가벼운 모델인 n(nano) 버전을 사용하여 CPU 부하를 최소화합니다)
model = YOLO("yolo11n.pt") 

# 2. 영상 경로 설정
input_path = r"C:\Users\seung\Desktop\ytdlp\제네시스 G90 주행영상 수원 영통 드라이브 [lxtJF6hOgIE].webm"
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("영상을 열 수 없습니다. 경로를 확인해주세요.")
    exit()

print("ByteTrack 추적을 시작합니다 (CPU 모드). 종료하려면 'q'를 누르세요.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. 추론 및 추적 수행
    # device="cpu"로 설정하여 CPU 자원만 사용합니다.
    results = model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        device="cpu"  # GPU 대신 CPU 사용
    )

    # 4. 결과 시각화 (객체 박스 및 추적 ID 표시)
    # CPU 모드에서도 시각화 결과는 동일하게 생성됩니다.
    annotated_frame = results[0].plot()

    # 화면에 출력
    cv2.imshow("G90 Drive - ByteTrack (CPU Mode)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()