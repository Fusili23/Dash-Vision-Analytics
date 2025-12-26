"""
Vehicle and Pedestrian Trajectory Prediction Module

This module implements both Constant Velocity Model (CVM) and Kalman Filter
approaches for predicting future trajectories of tracked objects.
"""

# === IMPORTS (bringing in external tools/libraries we need) ===
import numpy as np  # Import numpy library, give it short name 'np' - used for math with arrays/matrices
import cv2  # Import OpenCV library - used for computer vision tasks like drawing on images
from collections import defaultdict, deque  # Import special dictionary and queue data structures
# - defaultdict: dictionary that creates default value if key doesn't exist
# - deque: double-ended queue, like a list but faster for adding/removing from ends
from typing import Dict, List, Tuple, Optional  # Import type hints to make code more readable
# - Dict: dictionary type hint (key-value pairs)
# - List: list type hint (ordered collection)
# - Tuple: tuple type hint (fixed-size collection)
# - Optional: means value can be the type OR None


# === CLASS DEFINITION for Constant Velocity Predictor ===
class ConstantVelocityPredictor:  # Define a new class (blueprint for objects)
    """
    Simple Constant Velocity Model for trajectory prediction.
    
    Assumes objects will continue moving at their current velocity.
    Good for short-term predictions (1-3 seconds) in stable motion scenarios.
    """
    
    # Constructor method - runs when you create a new object
    def __init__(self, history_size: int = 15):  # 'self' is the object itself, history_size defaults to 15
        """
        Args:
            history_size: Number of past frames to keep for velocity estimation
        """
        self.history_size = history_size  # Store history_size as an attribute (property) of this object
        # Create a defaultdict (special dictionary) that stores track histories
        # Key: track_id (int), Value: deque (queue) of positions
        self.track_histories: Dict[int, deque] = defaultdict(  # ':' specifies type hint
            lambda: deque(maxlen=history_size)  # Lambda is anonymous function that creates new deque
            # maxlen=history_size means deque will automatically remove oldest item when full
        )
    
    # Method to add new position to track history
    def update(self, track_id: int, position: Tuple[float, float]):  # Takes track ID and (x,y) position
        """Add new position to track history."""
        self.track_histories[track_id].append(position)  # Add position to the end of this track's history
        # If track_id doesn't exist, defaultdict automatically creates new deque for it
    
    # Method to predict future positions
    def predict(  # Define method named 'predict'
        self,  # The object itself
        track_id: int,  # Which track to predict
        num_steps: int = 30,  # How many future frames to predict (default 30)
        lane_constraint: Optional[float] = None  # Optional angle to constrain movement (default None)
    ) -> List[Tuple[float, float]]:  # '->' means this function returns a list of (x,y) tuples
        """
        Predict future positions using constant velocity.
        
        Args:
            track_id: ID of the track to predict
            num_steps: Number of future frames to predict (30 fps = 1 second)
            lane_constraint: Optional angle in radians to constrain movement direction
            
        Returns:
            List of predicted (x, y) coordinates
        """
        history = self.track_histories[track_id]  # Get the position history for this track
        
        # Check if we have enough data (need at least 2 points to calculate velocity)
        if len(history) < 2:  # If history has less than 2 points
            return []  # Return empty list (can't predict without movement data)
        
        # Calculate average velocity from recent history
        velocities = []  # Create empty list to store velocity vectors
        # Loop through history, starting from index 1 (second item)
        for i in range(1, len(history)):  # range(1, n) creates [1, 2, 3, ..., n-1]
            # Calculate velocity in x direction (horizontal)
            vx = history[i][0] - history[i-1][0]  # Current x minus previous x
            # Calculate velocity in y direction (vertical)
            vy = history[i][1] - history[i-1][1]  # Current y minus previous y
            velocities.append((vx, vy))  # Add this velocity vector to the list
        
        # Use weighted average (recent velocities matter more)
        # Create array of linearly spaced weights from 0.5 to 1.0
        weights = np.linspace(0.5, 1.0, len(velocities))  # Older=0.5, newer=1.0
        weights /= weights.sum()  # Normalize weights so they sum to 1.0 (divide each by total)
        
        # Calculate weighted average velocity in x direction
        # zip() combines velocities and weights: [(v1,w1), (v2,w2), ...]
        # v[0] gets x-component of velocity, multiply by weight w
        # sum() adds all weighted x-velocities together
        avg_vx = sum(v[0] * w for v, w in zip(velocities, weights))
        # Same for y direction
        avg_vy = sum(v[1] * w for v, w in zip(velocities, weights))
        
        # Apply lane constraint if provided
        if lane_constraint is not None:  # If lane_constraint was given (not None)
            # Project velocity onto lane direction
            # Create unit vector pointing in lane direction
            lane_dir = np.array([np.cos(lane_constraint), np.sin(lane_constraint)])  # Convert angle to vector
            # Convert average velocity to numpy array
            velocity = np.array([avg_vx, avg_vy])
            # Project velocity onto lane direction (dot product gives component along lane)
            # Then multiply by lane direction to get vector along lane
            projected_velocity = np.dot(velocity, lane_dir) * lane_dir
            # Extract x and y components from projected velocity
            avg_vx, avg_vy = projected_velocity  # Unpack array into two variables
        
        # Generate predictions
        predictions = []  # Create empty list for predicted positions
        current_pos = history[-1]  # Get last position in history (index -1 = last item)
        
        # Loop through number of prediction steps
        for step in range(1, num_steps + 1):  # Loop from 1 to num_steps (inclusive)
            # Predict x position: current x + (velocity_x * number of steps into future)
            pred_x = current_pos[0] + avg_vx * step
            # Predict y position: current y + (velocity_y * number of steps into future)
            pred_y = current_pos[1] + avg_vy * step
            predictions.append((pred_x, pred_y))  # Add predicted (x,y) to list
        
        return predictions  # Return list of all predicted positions


# === CLASS DEFINITION for Kalman Filter Predictor ===
class KalmanFilterPredictor:  # Define a new class
    """
    Kalman Filter for smoother trajectory prediction.
    
    Better handles noise and provides uncertainty estimates.
    Suitable for more complex motion patterns.
    """
    
    # Constructor method
    def __init__(self):  # No parameters except self
        self.filters: Dict[int, dict] = {}  # Empty dictionary to store Kalman filters for each track
        # Key: track_id, Value: dictionary containing filter matrices
        
        # Kalman filter parameters
        # State: [x, y, vx, vy]  (position and velocity)
        self.dt = 1.0  # Time step (1 frame) - stored as object attribute
        
        # Process noise (how much we trust the model)
        self.process_noise = 0.1  # Small value = trust model more
        
        # Measurement noise (how much we trust observations)
        self.measurement_noise = 0.5  # Larger value = trust measurements less
    
    # Private method (starts with _) to initialize a new Kalman filter
    def _initialize_filter(self, position: Tuple[float, float]) -> dict:  # Returns dictionary
        """Initialize a new Kalman filter for a track."""
        x, y = position  # Unpack position tuple into x and y variables
        
        # State vector: [x, y, vx, vy] - stores position and velocity
        state = np.array([x, y, 0.0, 0.0])  # Create numpy array, initially velocity is 0
        
        # State transition matrix (constant velocity model)
        # Describes how state evolves: new_state = F * old_state
        F = np.array([  # Create 4x4 matrix
            [1, 0, self.dt, 0],      # new_x = x + vx*dt
            [0, 1, 0, self.dt],      # new_y = y + vy*dt
            [0, 0, 1, 0],            # new_vx = vx (constant)
            [0, 0, 0, 1]             # new_vy = vy (constant)
        ])
        
        # Measurement matrix (we only observe position, not velocity)
        # Relates measurements to state: measurement = H * state
        H = np.array([  # Create 2x4 matrix
            [1, 0, 0, 0],  # We measure x (first element of state)
            [0, 1, 0, 0]   # We measure y (second element of state)
        ])
        
        # Covariance matrix - represents uncertainty in state estimate
        P = np.eye(4) * 100  # Create 4x4 identity matrix, multiply all by 100 (high initial uncertainty)
        # np.eye(4) creates matrix with 1s on diagonal, 0s elsewhere
        
        # Process noise covariance - uncertainty in model
        Q = np.eye(4) * self.process_noise  # 4x4 identity matrix times noise parameter
        
        # Measurement noise covariance - uncertainty in measurements
        R = np.eye(2) * self.measurement_noise  # 2x2 identity matrix times noise parameter
        
        # Return dictionary containing all filter matrices
        return {  # Create dictionary with these key-value pairs
            'state': state,  # Current state estimate
            'F': F,         # State transition matrix
            'H': H,         # Measurement matrix
            'P': P,         # Covariance matrix
            'Q': Q,         # Process noise
            'R': R          # Measurement noise
        }
    
    # Method to update filter with new measurement
    def update(self, track_id: int, position: Tuple[float, float]):  # Takes track ID and measured position
        """Update Kalman filter with new measurement."""
        # If this track doesn't have a filter yet
        if track_id not in self.filters:  # Check if track_id key exists in dictionary
            # Initialize a new filter for this track
            self.filters[track_id] = self._initialize_filter(position)
            return  # Exit function early (nothing more to do on first measurement)
        
        kf = self.filters[track_id]  # Get the filter dictionary for this track
        
        # === PREDICTION STEP ===
        # Predict next state using state transition matrix
        state_pred = kf['F'] @ kf['state']  # '@' is matrix multiplication operator
        # Predict next covariance: P = F*P*F^T + Q
        P_pred = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']  # .T means transpose (flip rows/columns)
        
        # === UPDATE STEP ===
        measurement = np.array(position)  # Convert position tuple to numpy array
        # Innovation (difference between measurement and prediction)
        y = measurement - (kf['H'] @ state_pred)  # How far off was our prediction?
        # Innovation covariance
        S = kf['H'] @ P_pred @ kf['H'].T + kf['R']  # Uncertainty in innovation
        # Kalman gain (how much to trust the measurement)
        K = P_pred @ kf['H'].T @ np.linalg.inv(S)  # np.linalg.inv() inverts matrix
        
        # Update state estimate using Kalman gain and innovation
        kf['state'] = state_pred + K @ y  # Correct prediction using measurement
        # Update covariance
        kf['P'] = (np.eye(4) - K @ kf['H']) @ P_pred  # Reduce uncertainty
    
    # Method to predict future positions
    def predict(  # Define method
        self,  # The object  
        track_id: int,  # Which track
        num_steps: int = 30,  # How many steps (default 30)
        lane_constraint: Optional[float] = None  # Optional angle constraint
    ) -> List[Tuple[float, float]]:  # Returns list of (x,y) positions
        """
        Predict future positions using Kalman filter.
        
        Args:
            track_id: ID of the track to predict
            num_steps: Number of future frames to predict
            lane_constraint: Optional angle in radians to constrain movement
            
        Returns:
            List of predicted (x, y) coordinates
        """
        # Check if filter exists for this track
        if track_id not in self.filters:  # If no filter
            return []  # Return empty list
        
        kf = self.filters[track_id]  # Get filter for this track
        state = kf['state'].copy()  # Copy current state (don't modify original)
        F = kf['F']  # Get state transition matrix
        
        # Apply lane constraint to velocity if provided
        if lane_constraint is not None:  # If constraint was given
            # Create lane direction vector
            lane_dir = np.array([np.cos(lane_constraint), np.sin(lane_constraint)])
            velocity = state[2:4]  # Extract velocity components (indices 2 and 3)
            # Project velocity onto lane direction
            projected_velocity = np.dot(velocity, lane_dir) * lane_dir
            state[2:4] = projected_velocity  # Update velocity in state
        
        predictions = []  # Create empty list for predictions
        # Loop num_steps times
        for _ in range(num_steps):  # _ means we don't use the loop variable
            state = F @ state  # Predict next state using transition matrix
            predictions.append((state[0], state[1]))  # Add predicted (x,y) to list
        
        return predictions  # Return all predicted positions


# === CLASS DEFINITION for Trajectory Visualizer ===
class TrajectoryVisualizer:  # Define class
    """Handles visualization of predicted trajectories."""
    
    @staticmethod  # Decorator marking this as a static method (doesn't need 'self')
    def draw_trajectory(  # Define static method
        frame: np.ndarray,  # Input image (numpy array)
        predictions: List[Tuple[float, float]],  # List of predicted positions
        color: Tuple[int, int, int] = (0, 255, 0),  # RGB color (default green)
        fade: bool = True,  # Whether to fade trajectory (default yes)
        draw_points: bool = True  # Whether to draw points (default yes)
    ) -> np.ndarray:  # Returns modified image
        """
        Draw predicted trajectory on frame.
        
        Args:
            frame: Input frame
            predictions: List of (x, y) coordinates
            color: RGB color for trajectory
            fade: Whether to apply fading effect
            draw_points: Whether to draw points along trajectory
            
        Returns:
            Frame with trajectory drawn
        """
        # Check if we have enough predictions to draw
        if len(predictions) < 2:  # If less than 2 points
            return frame  # Return original frame unchanged
        
        overlay = frame.copy()  # Create a copy of the frame to draw on
        
        # Draw trajectory line with fading effect
        # Loop through each pair of consecutive points
        for i in range(len(predictions) - 1):  # Stop one before end
            # Get first point, convert to integers (pixels must be whole numbers)
            pt1 = (int(predictions[i][0]), int(predictions[i][1]))
            # Get second point
            pt2 = (int(predictions[i + 1][0]), int(predictions[i + 1][1]))
            
            # Calculate alpha for fading
            if fade:  # If fading is enabled
                # Calculate fade amount (1.0 at start, 0.0 at end)
                alpha = 1.0 - (i / len(predictions))  # Divide position by total length
                # Calculate line thickness based on alpha
                thickness = max(1, int(3 * alpha))  # At least 1 pixel thick
                # Calculate faded color (multiply each RGB component by alpha)
                current_color = tuple(int(c * alpha) for c in color)  # Tuple comprehension
            else:  # No fading
                thickness = 2  # Fixed thickness
                current_color = color  # Original color
            
            # Draw line between the two points
            cv2.line(overlay, pt1, pt2, current_color, thickness)  # OpenCV line function
        
        # Draw points along trajectory
        if draw_points:  # If points should be drawn
            # Loop through every 5th prediction (to avoid cluttering)
            for i, pred in enumerate(predictions[::5]):  # [::5] means every 5th item
                # enumerate gives (index, value) pairs
                pt = (int(pred[0]), int(pred[1]))  # Convert to integer coordinates
                # Calculate alpha for this point
                alpha = 1.0 - (i * 5 / len(predictions)) if fade else 1.0  # Conditional expression
                # Calculate circle radius
                radius = max(2, int(5 * alpha))  # At least 2 pixels
                # Draw filled circle (-1 means fill)
                cv2.circle(overlay, pt, radius, color, -1)
        
        # Blend overlay with original frame
        # addWeighted: result = overlay*0.7 + frame*0.3 + 0
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)  # Result stored in 'frame'
        
        return frame  # Return modified frame
    
    @staticmethod  # Static method (no self)
    def draw_future_position(  # Define method
        frame: np.ndarray,  # Input image
        position: Tuple[float, float],  # Position to mark
        label: str = "",  # Text label (default empty)
        color: Tuple[int, int, int] = (0, 255, 255)  # Color (default cyan)
    ) -> np.ndarray:  # Returns modified image
        """Draw a marker for predicted future position."""
        pt = (int(position[0]), int(position[1]))  # Convert position to integer coordinates
        
        # Draw crosshair marker
        cv2.drawMarker(frame, pt, color, cv2.MARKER_CROSS, 15, 2)  
        # drawMarker(image, position, color, marker_type, size, thickness)
        
        # Draw label if provided
        if label:  # If label is not empty string
            # Put text on image
            cv2.putText(
                frame,  # Image to draw on
                label,  # Text string
                (pt[0] + 10, pt[1] - 10),  # Position (offset from marker)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                0.5,  # Font scale
                color,  # Text color
                1  # Thickness
            )
        
        return frame  # Return modified frame


# === STANDALONE FUNCTION (not in a class) ===
def estimate_lane_direction(  # Define function
    history: List[Tuple[float, float]],  # List of positions
    min_samples: int = 10  # Minimum required points (default 10)
) -> Optional[float]:  # Returns angle OR None
    """
    Estimate lane direction from trajectory history.
    
    Uses linear regression to find dominant direction of movement.
    
    Args:
        history: List of (x, y) positions
        min_samples: Minimum number of samples needed
        
    Returns:
        Angle in radians, or None if insufficient data
    """
    # Check if we have enough samples
    if len(history) < min_samples:  # If too few points
        return None  # Return None (can't estimate)
    
    positions = np.array(history)  # Convert list to numpy array for math operations
    
    # Fit line using SVD (Singular Value Decomposition - more robust than simple linear regression)
    mean = positions.mean(axis=0)  # Calculate mean of each column (x_mean, y_mean)
    # axis=0 means average down rows (collapse rows into single row)
    centered = positions - mean  # Subtract mean from all positions (center data at origin)
    
    # Perform Singular Value Decomposition
    _, _, vt = np.linalg.svd(centered)  # SVD returns 3 values, we only need the 3rd
    # _ means we ignore those values
    direction = vt[0]  # First principal component (dominant direction of variation)
    
    # Convert direction vector to angle
    angle = np.arctan2(direction[1], direction[0])  # arctan2(y, x) gives angle in radians
    
    return angle  # Return the angle


# === EXAMPLE/DEMO FUNCTION ===
def demonstrate_prediction():  # Define function with no parameters
    """Demonstration of how to use the prediction system."""
    
    # Initialize predictor (choose one)
    predictor = KalmanFilterPredictor()  # Create new KalmanFilterPredictor object
    # Could also use: predictor = ConstantVelocityPredictor()
    visualizer = TrajectoryVisualizer()  # Create visualizer object
    
    # Simulate tracking data
    track_id = 1  # Assign ID number 1 to this track
    
    # Example: Update with new detections
    for frame_num in range(30):  # Loop 30 times (0 to 29)
        x = 100 + frame_num * 5  # X position increases linearly (moving right)
        y = 200 + np.sin(frame_num * 0.2) * 10  # Y position follows sine wave (slight curve)
        # np.sin() calculates sine function
        
        predictor.update(track_id, (x, y))  # Add this position to prediction history
    
    # Predict next 60 frames (2 seconds at 30 fps)
    predictions = predictor.predict(track_id, num_steps=60)  # Get predictions
    
    # Print results
    print(f"Predicted {len(predictions)} future positions")  # f-string: embed variable in string
    print(f"First 5 predictions: {predictions[:5]}")  # [:5] means first 5 items
    
    return predictions  # Return the predictions list


# === MAIN EXECUTION BLOCK ===
if __name__ == "__main__":  # This block only runs if script is run directly (not imported)
    # __name__ is special variable that equals "__main__" when script is main program
    demonstrate_prediction()  # Call the demonstration function
