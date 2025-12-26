"""
Complete Context-Aware Trajectory Prediction with Ego-Motion Compensation

Features:
- Ego-motion estimation using optical flow
- Context-aware prediction (traffic lights, semantic zones)
- Multi-modal predictions for ambiguous situations
- BEV-space relative velocity calculation
- Real-time visualization
"""

# === IMPORT STATEMENTS (bringing in tools we need) ===
# === IMPORT STATEMENTS (bringing in tools we need) ===
import cv2  # Import OpenCV - computer vision library for images/video
from ultralytics import YOLO  # Import YOLO object detection model
import numpy as np  # Import numpy - math library for arrays/matrices
from collections import defaultdict  # Import defaultdict - dictionary that creates default values
import sys  # Import sys for path management
from pathlib import Path  # Import Path for file path operations

# Add parent directory to path to import from src/
# This allows: from src.module import Class
sys.path.append(str(Path(__file__).parent.parent))  # Go up one level to project root

# Import our custom modules (files we created in src/ folder)
from src.ego_motion import EgoMotionEstimator, RelativeVelocityTracker, BEVEgoMotionIntegrator
# - EgoMotionEstimator: calculates camera movement
# - RelativeVelocityTracker: tracks object velocities with ego-motion compensation  
# - BEVEgoMotionIntegrator: handles bird's-eye view transformations

from src.context_aware_predictor import (  # Import from context_aware_predictor.py
    ContextAwarePredictor,  # Main prediction engine with context awareness
    EnvironmentalContext,  # Data structure for environment info
    TrafficLight,  # Data structure for traffic light info
    TrafficLightState,  # Enum for light states (RED, YELLOW, GREEN)
    UncertaintyVisualizer  # Tool for drawing prediction uncertainty
)

from src.semantic_zones import ZoneMaskGenerator, SemanticZone  # Tools for road/sidewalk zones
from src.bev_transformer import BEVTransformer  # Tool for perspective transformation


# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# 1. Load YOLO model
model = YOLO("yolo11n.pt")  # Create YOLO object, load weights from file
# "yolo11n.pt" is the model file (n = nano, smallest/fastest version)

# 2. Video path
# r"..." is raw string (backslashes treated as normal characters, not escape codes)
input_path = r"C:\Users\seung\Desktop\ytdlp\제네시스 G90 주행영상 수원 영통 드라이브 [lxtJF6hOgIE].webm"
cap = cv2.VideoCapture(input_path)  # Create video capture object to read video file

# Check if video opened successfully
if not cap.isOpened():  # If video couldn't be opened
    print("영상을 열 수 없습니다. 경로를 확인해주세요.")  # Print error message in Korean
    exit()  # Exit the program

# Get video properties
# cap.get() retrieves video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get width, convert to integer
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get height, convert to integer
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Get FPS, use 30 if get() returns 0
# 'or 30' means: if left side is falsy (0, None, False), use right side (30)

# 3. Initialize all systems
print("=" * 60)  # Print 60 equal signs (horizontal line)
print("고급 궤적 예측 시스템 초기화 중...")  # Print message in Korean
print("=" * 60)  # Another horizontal line

# Ego-motion estimation (camera movement detection)
ego_estimator = EgoMotionEstimator(history_size=5, flow_quality='medium')
# Create estimator object with these parameters:
# - history_size=5: smooth over 5 frames
# - flow_quality='medium': balance between speed and accuracy
velocity_tracker = RelativeVelocityTracker()  # Create velocity tracker object
print("✓ Ego-motion estimator initialized")  # Confirmation message

# Context-aware prediction (traffic lights, semantic zones)
predictor = ContextAwarePredictor()  # Create main predictor object
visualizer = UncertaintyVisualizer()  # Create visualizer object for drawing
print("✓ Context-aware predictor initialized")  # Confirmation message

# BEV transformation (bird's-eye view)
bev_transformer = BEVTransformer(frame_width, frame_height, bev_width=400, bev_height=800)
# Create transformer with:
# - Input image size: frame_width x frame_height
# - Output BEV size: 400 x 800 pixels
bev_integrator = BEVEgoMotionIntegrator(bev_transformer, pixels_per_meter=10.0)
# Create integrator that combines BEV with ego-motion
# - pixels_per_meter=10.0: calibration parameter (10 pixels = 1 meter in BEV)
print("✓ BEV transformer initialized")  # Confirmation message

# Semantic zones (road/sidewalk masks)
zone_gen = ZoneMaskGenerator(frame_width, frame_height)  # Create zone mask generator

# Create default road mask (trapezoid shape covering lower portion of image)
road_polygon = [  # List of (x, y) corner points
    (int(frame_width * 0.3), int(frame_height * 0.6)),    # Top-left corner
    # int() converts float to integer
    # frame_width * 0.3 = 30% from left edge
    # frame_height * 0.6 = 60% from top
    (int(frame_width * 0.7), int(frame_height * 0.6)),    # Top-right corner
    (int(frame_width * 0.95), frame_height),               # Bottom-right corner
    (int(frame_width * 0.05), frame_height)                # Bottom-left corner
]
zone_gen.create_road_mask_from_polygon(road_polygon)  # Create mask from this polygon
print("✓ Semantic zones initialized")  # Confirmation message

# =============================================================================
# SETTINGS SECTION
# =============================================================================

# Prediction settings
PREDICT_SECONDS = 3.0  # Predict 3 seconds into future (float number)
PREDICT_FRAMES = int(PREDICT_SECONDS * fps)  # Calculate number of frames
# Example: 3.0 seconds * 30 fps = 90 frames

# Feature toggles (True/False switches)
ENABLE_EGO_MOTION = True  # Enable camera movement compensation
ENABLE_CONTEXT_AWARE = True  # Enable traffic light & zone awareness
ENABLE_BEV_CALCULATION = True  # Enable bird's-eye view calculations
SHOW_OPTICAL_FLOW = False  # Show optical flow visualization (set True to see)

# Tracked classes (vehicles and humans only)
TRACKED_CLASSES = {  # Set literal (unordered collection of unique items)
    'person',      # Pedestrians
    'bicycle',     # Bicycles
    'car',         # Cars
    'motorcycle',  # Motorcycles
    'bus',         # Buses
    'truck'        # Trucks
}
# Sets use curly braces {}
# Fast for checking membership: if 'car' in TRACKED_CLASSES

# Colors for different object types (BGR format, not RGB!)
COLORS = {  # Dictionary mapping class names to colors
    'car': (0, 255, 0),          # Green (B=0, G=255, R=0)
    'truck': (255, 165, 0),      # Orange
    'bus': (255, 100, 100),      # Coral
    'person': (255, 0, 255),     # Magenta/Purple
    'bicycle': (0, 255, 255),    # Cyan/Aqua
    'motorcycle': (200, 0, 200)  # Purple
}

# Print configuration summary
print("=" * 60)  # Horizontal line
print(f"설정:")  # "Settings:" in Korean
# f-strings let you embed variables/expressions in strings using {}
print(f"  - 예측 시간: {PREDICT_SECONDS}초 ({PREDICT_FRAMES} 프레임)")
# {PREDICT_SECONDS} gets replaced with value of variable
print(f"  - Ego-motion 보정: {'ON' if ENABLE_EGO_MOTION else 'OFF'}")
# Conditional expression: value_if_true if condition else value_if_false
print(f"  - 컨텍스트 인식: {'ON' if ENABLE_CONTEXT_AWARE else 'OFF'}")
print(f"  - BEV 계산: {'ON' if ENABLE_BEV_CALCULATION else 'OFF'}")
print("=" * 60)  # Horizontal line
print("\n시작합니다! 종료하려면 'q'를 누르세요.\n")  # \n means newline character

# =============================================================================
# TRAFFIC LIGHT DETECTION FUNCTION
# =============================================================================

def detect_traffic_lights_simple(frame: np.ndarray) -> list:  # Define function
    # Takes frame (numpy array), returns list
    """
    Simple traffic light detection using color detection.
    
    Returns list of TrafficLight objects.
    """
    # Convert frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV = Hue, Saturation, Value (better for color detection than RGB/BGR)
    
    # Detect red regions (potential red lights)
    # Red in HSV wraps around (0-10 and 160-180)
    red_lower1 = np.array([0, 100, 100])    # Lower bound for red (H, S, V)
    red_upper1 = np.array([10, 255, 255])   # Upper bound for red
    red_lower2 = np.array([160, 100, 100])  # Second red range (wraps around)
    red_upper2 = np.array([180, 255, 255])  # Upper bound
    
    # Create binary masks (white where color is in range, black elsewhere)
    # | is bitwise OR operator (combines two masks)
    # \ at end of line means continuation on next line
    mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | \
               cv2.inRange(hsv, red_lower2, red_upper2)
    
    # Find contours (outlines) in the mask
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Returns two values: contours and hierarchy
    # _ means we ignore the second value (hierarchy)
    
    traffic_lights = []  # Create empty list to store detected lights
    
    # Loop through each contour
    for contour in contours:  # For each contour in the list
        area = cv2.contourArea(contour)  # Calculate area in pixels
        
        # Filter by size (traffic lights should be 50-500 pixels)
        if 50 < area < 500:  # If area is between 50 and 500
            # Calculate moments (for finding center)
            M = cv2.moments(contour)  # Returns dictionary of moment values
            
            # Check if moments are valid (m00 is area, should be > 0)
            if M["m00"] > 0:  # If m00 (area) is greater than 0
                # Calculate center x coordinate
                cx = int(M["m10"] / M["m00"])  # m10/m00 gives x center
                # Calculate center y coordinate  
                cy = int(M["m01"] / M["m00"])  # m01/m00 gives y center
                
                # Only accept lights in upper half of image (where they typically are)
                if cy < frame_height * 0.6:  # If y is less than 60% down
                    # Create TrafficLight object and add to list
                    traffic_lights.append(TrafficLight(  # append() adds to end of list
                        position=(cx, cy),  # (x, y) position
                        state=TrafficLightState.RED,  # Assume it's red
                        confidence=0.7,  # 70% confidence
                        detection_time=0  # Will be updated later with frame number
                    ))
    
    return traffic_lights  # Return the list of detected traffic lights

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

frame_count = 0  # Initialize frame counter to 0
ego_velocity_current = (0.0, 0.0)  # Initialize ego velocity as (0, 0) tuple

# Main loop - processes video frame by frame
while cap.isOpened():  # While video is open
    success, frame = cap.read()  # Read next frame
    # Returns two values: success (True/False) and frame (image array)
    
    if not success:  # If frame reading failed (end of video or error)
        break  # Exit the while loop
    
    frame_count += 1  # Increment frame counter by 1 (same as frame_count = frame_count + 1)
    
    # -------------------------------------------------------------------------
    # STEP 1: ESTIMATE EGO-MOTION (camera movement)
    # -------------------------------------------------------------------------
    if ENABLE_EGO_MOTION:  # If ego-motion is enabled
        # Estimate camera velocity from optical flow
        ego_velocity_current = ego_estimator.estimate_ego_motion(frame)
        # Returns (vx, vy) tuple - camera velocity in pixels/frame
        
        # Calculate speed (magnitude of velocity vector)
        ego_speed = np.linalg.norm(ego_velocity_current)
        # np.linalg.norm() calculates Euclidean distance: sqrt(vx^2 + vy^2)
    else:  # If ego-motion is disabled
        ego_velocity_current = (0.0, 0.0)  # No camera movement
        ego_speed = 0.0  # Zero speed
    
    # -------------------------------------------------------------------------
    # STEP 2: DETECT TRAFFIC LIGHTS (simple color-based detection)
    # -------------------------------------------------------------------------
    # Only detect every 5 frames to save computation
    if ENABLE_CONTEXT_AWARE and frame_count % 5 == 0:  # If context enabled AND frame is multiple of 5
        # % is modulo operator (remainder after division)
        # frame_count % 5 == 0 is True for frames 0, 5, 10, 15, ...
        
        traffic_lights = detect_traffic_lights_simple(frame)  # Call detection function
        
        # Update detection time for each light
        for light in traffic_lights:  # For each TrafficLight object in list
            light.detection_time = frame_count  # Set detection time to current frame
    else:  # If context disabled or not a detection frame
        traffic_lights = []  # Empty list (no traffic lights)
    
    # -------------------------------------------------------------------------
    # STEP 3: TRACK OBJECTS (detect and track vehicles/people)
    # -------------------------------------------------------------------------
    results = model.track(  # Call YOLO track method
        frame,  # Input image
        persist=True,  # Keep track IDs across frames
        tracker="bytetrack.yaml",  # Use ByteTrack algorithm (config file)
        device="cpu",  # Use CPU (not GPU)
        verbose=False  # Don't print information to console
    )
    # Returns Results object containing detections
    
    # -------------------------------------------------------------------------
    # STEP 4: PROCESS DETECTIONS AND PREDICT TRAJECTORIES
    # -------------------------------------------------------------------------
    prediction_frame = frame.copy()  # Create copy of frame for drawing predictions
    # .copy() creates independent copy (modifying copy won't change original)
    
    # Check if any objects were detected
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # results[0] is first (and only) image's results
        # .boxes is detection boxes
        # .id is track IDs (None if no tracking)
        
        # Extract detection data
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        # .xyxy means format with top-left (x1,y1) and bottom-right (x2,y2)
        # .cpu() moves from GPU to CPU memory
        # .numpy() converts PyTorch tensor to numpy array
        
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # Track IDs as integers
        # .astype(int) converts data type to integer
        
        classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs as integers
        
        # Get class names (convert class IDs to text labels)
        class_names = [results[0].names[cls_id] for cls_id in classes]
        # List comprehension: [expression for item in iterable]
        # results[0].names is dictionary: {0: 'person', 1: 'bicycle', 2: 'car', ...}
        
        # Process each detected object
        for box, track_id, class_name in zip(boxes, track_ids, class_names):
            # zip() combines multiple lists: [(box1,id1,name1), (box2,id2,name2), ...]
            
            # Skip if object is not a vehicle or person
            if class_name not in TRACKED_CLASSES:  # If class not in our set
                continue  # Skip to next iteration of loop
            
            # Calculate center point of bounding box
            x_center = (box[0] + box[2]) / 2  # Average of left and right x
            # box[0] is x1 (left), box[2] is x2 (right)
            y_center = (box[1] + box[3]) / 2  # Average of top and bottom y
            # box[1] is y1 (top), box[3] is y2 (bottom)
            position = (x_center, y_center)  # Create tuple
            
            # -------------------------------------------------------------
            # Update velocity tracker with ego-motion compensation
            # -------------------------------------------------------------
            velocity_tracker.update(track_id, position, ego_velocity_current)
            # This calculates: actual_velocity = perceived_velocity - ego_velocity
            
            actual_velocity = velocity_tracker.get_actual_velocity(track_id)
            # Get the ego-compensated velocity (ground-relative)
            
            # Calculate speed magnitude
            actual_speed = np.linalg.norm(actual_velocity)  # Speed in pixels/frame
            
            # -------------------------------------------------------------
            # BEV velocity calculation (convert to real-world speed)
            # -------------------------------------------------------------
            # Only calculate if BEV enabled and object is moving significantly
            # Add threshold to avoid noise from stationary objects
            if ENABLE_BEV_CALCULATION and actual_speed > 1.0:  # Increased threshold from 0.5 to 1.0
                # np.linalg.norm() gets velocity magnitude (speed)
                # > 1.0 means only process if speed > 1.0 pixels/frame
                
                try:
                    # Transform velocity from image space to BEV space
                    bev_velocity = bev_integrator.image_velocity_to_bev(
                        actual_velocity,  # Velocity in image coordinates
                        position           # Current position (needed for transformation)
                    )
                    
                    # Convert BEV velocity to metric units (m/s)
                    velocity_ms = bev_integrator.bev_velocity_to_metric(bev_velocity)
                    
                    # Convert m/s to km/h
                    speed_kmh = bev_integrator.get_speed_kmh(velocity_ms)
                    
                    # Safety check: limit unrealistic speeds
                    # Typical highway speeds: 0-150 km/h
                    if speed_kmh > 200 or speed_kmh < 0:
                        speed_kmh = 0.0  # Reset if unrealistic
                        
                except Exception as e:
                    # If BEV transformation fails, fall back to zero
                    speed_kmh = 0.0
            else:  # BEV disabled or object not moving enough
                speed_kmh = 0.0  # Zero speed
            
            # -------------------------------------------------------------
            # Create environmental context and predict trajectory
            # -------------------------------------------------------------
            if ENABLE_CONTEXT_AWARE:  # If context awareness is enabled
                # Create context object containing environmental information
                context = EnvironmentalContext(
                    traffic_lights=traffic_lights,  # List of detected traffic lights
                    zone_type=SemanticZone.ROAD,    # Assume object is on road
                    zone_mask=zone_gen.road_mask,   # Binary mask of road area
                    timestamp=frame_count            # Current frame number
                )
                
                # Update predictor with current position and context
                predictor.update(track_id, position, context)
                
                # Predict trajectory considering context
                prediction_result = predictor.predict_with_context(
                    track_id,                    # Which object to predict
                    num_steps=PREDICT_FRAMES,   # How many frames ahead
                    context=context,             # Environmental context
                    object_class=class_name      # Type of object (car, person, etc.)
                )
                # Returns dictionary with 'primary', 'alternative', 'probabilities', 'intent'
                
                # Get color for this object type
                color = COLORS.get(class_name, (255, 255, 255))
                # .get(key, default) returns value for key, or default if key not found
                # Default is white (255, 255, 255)
                
                # Draw multi-modal trajectory (primary + alternative paths)
                prediction_frame = visualizer.draw_multi_modal_trajectory(
                    prediction_frame,     # Frame to draw on
                    prediction_result,    # Prediction dictionary
                    base_color=color      # Color to use
                )
                
                # Display speed on bounding box (if moving)
                if speed_kmh > 1.0:  # If speed greater than 1 km/h
                    speed_text = f"{speed_kmh:.1f} km/h"  # Format with 1 decimal place
                    # :.1f means floating point with 1 digit after decimal
                    
                    cv2.putText(  # Draw text on image
                        prediction_frame,              # Image to draw on
                        speed_text,                    # Text string
                        (int(box[0]), int(box[1]) - 25),  # Position (above bounding box)
                        cv2.FONT_HERSHEY_SIMPLEX,      # Font type
                        0.5,                           # Font scale (size)
                        color,                         # Text color
                        2                              # Thickness
                    )
            else:  # Context awareness disabled - do basic prediction
                # Update basic predictor (without context)
                predictor.base_predictor.update(track_id, position)
                
                # Get basic prediction
                basic_pred = predictor.base_predictor.predict(track_id, PREDICT_FRAMES)
                
                # Draw simple trajectory if predictions exist
                if basic_pred:  # If list is not empty
                    color = COLORS.get(class_name, (255, 255, 255))  # Get color
                    pts = np.array(basic_pred, dtype=np.int32)  # Convert to numpy array of integers
                    # dtype=np.int32 specifies data type (32-bit integers)
                    
                    cv2.polylines(prediction_frame, [pts], False, color, 2)
                    # polylines(image, [points], is_closed, color, thickness)
                    # [pts] wrapped in list because polylines expects list of polylines
                    # False means don't close the polyline
    
    # -------------------------------------------------------------------------
    # STEP 5: DRAW BASE TRACKING RESULTS (bounding boxes, IDs)
    # -------------------------------------------------------------------------
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels
    # .plot() returns new image with visualization
    
    # Blend prediction frame with annotated frame
    final_frame = cv2.addWeighted(prediction_frame, 0.6, annotated_frame, 0.4, 0)
    # addWeighted(src1, alpha1, src2, alpha2, gamma)
    # result = src1 * 0.6 + src2 * 0.4 + 0
    # This blends predictions (60%) with detections (40%)
    
    # -------------------------------------------------------------------------
    # STEP 6: OVERLAY INFORMATION (traffic lights, info panel)
    # -------------------------------------------------------------------------
    
    # Draw traffic light markers
    for light in traffic_lights:  # For each detected traffic light
        pos = (int(light.position[0]), int(light.position[1]))  # Get position as integer tuple
        
        cv2.circle(final_frame, pos, 10, (0, 0, 255), 2)  # Draw red circle
        # circle(image, center, radius, color, thickness)
        # (0, 0, 255) is red in BGR
        # 2 is thickness (outline, not filled)
        
        cv2.putText(  # Draw "RED" label
            final_frame,                    # Image
            "RED",                          # Text
            (pos[0] + 15, pos[1]),         # Position (offset from circle)
            cv2.FONT_HERSHEY_SIMPLEX,      # Font
            0.4,                            # Scale
            (0, 0, 255),                   # Color (red)
            1                               # Thickness
        )
    
    # Calculate number of tracked objects
    # Conditional expression to avoid error if no detections
    tracked_count = sum(1 for name in class_names if name in TRACKED_CLASSES) if results[0].boxes.id is not None else 0
    # sum(1 for ...) counts how many items match condition
    # Generator expression: (value for item in iterable if condition)
    
    # Create info panel text lines
    info_lines = [  # List of strings
        f"Frame: {frame_count}",                       # Current frame number
        f"Tracked: {tracked_count} objects",           # Number tracked
        f"Ego Speed: {ego_speed:.1f} px/frame",       # Camera speed
        f"Prediction: {PREDICT_SECONDS}s ahead"        # Prediction horizon
    ]
    
    # Add traffic light count if context aware
    if ENABLE_CONTEXT_AWARE:  # If enabled
        info_lines.append(f"Traffic Lights: {len(traffic_lights)}")
        # .append() adds to end of list
        # len() gets length of list
    
    # Draw semi-transparent background for info panel
    panel_height = len(info_lines) * 25 + 20  # Calculate height based on number of lines
    # Each line is 25 pixels tall, plus 20 pixels padding
    
    overlay = final_frame.copy()  # Create copy for transparency effect
    cv2.rectangle(overlay, (5, 5), (300, panel_height), (0, 0, 0), -1)
    # rectangle(image, top_left, bottom_right, color, thickness)
    # -1 means fill the rectangle (not just outline)
    # (0, 0, 0) is black
    
    cv2.addWeighted(overlay, 0.6, final_frame, 0.4, 0, final_frame)
    # Blend black rectangle (60% opacity) with frame
    # Result stored back into final_frame
    
    # Draw info text
    y_offset = 25  # Starting y position
    for line in info_lines:  # For each line of text
        cv2.putText(  # Draw text
            final_frame,                    # Image
            line,                           # Text string
            (10, y_offset),                # Position
            cv2.FONT_HERSHEY_SIMPLEX,      # Font
            0.5,                            # Scale
            (255, 255, 255),               # White color
            1                               # Thickness
        )
        y_offset += 25  # Move down for next line
    
    # -------------------------------------------------------------------------
    # STEP 7: DISPLAY RESULTS
    # -------------------------------------------------------------------------
    
    # Show optical flow visualization (if enabled and appropriate frame)
    if SHOW_OPTICAL_FLOW and frame_count % 10 == 0:  # Every 10th frame
        # Visualize optical flow
        flow_vis = ego_estimator.visualize_flow(frame, step=20, scale=2)
        # step=20: draw arrow every 20 pixels
        # scale=2: multiply arrow length by 2
        
        cv2.imshow("Optical Flow (Ego-Motion)", flow_vis)  # Show in separate window
        # imshow(window_name, image)
    
    # Display main output window
    cv2.imshow("Advanced Trajectory Prediction", final_frame)
    # imshow creates or updates window with name and shows image
    
    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Wait 1ms for key press
        # waitKey(1) waits 1 millisecond and returns key code (or -1 if no key)
        # & 0xFF masks to get last 8 bits (for cross-platform compatibility)
        # ord("q") gets ASCII code of 'q' character
        break  # Exit the while loop

# =============================================================================
# CLEANUP SECTION (release resources)
# =============================================================================

cap.release()  # Release video capture object (closes video file)
cv2.destroyAllWindows()  # Close all OpenCV windows

# Print final statistics
print(f"\n처리 완료: 총 {frame_count} 프레임")  # "Processing complete" in Korean
# \n is newline character

# Calculate and print average ego speed
# List comprehension with np.linalg.norm to get magnitude of each velocity
print(f"평균 ego 속도: {np.mean([np.linalg.norm(v) for v in ego_estimator.ego_velocity_history]):.2f} px/frame")
# np.mean() calculates average
# :.2f formats with 2 decimal places
# [np.linalg.norm(v) for v in ...] creates list of speed magnitudes
