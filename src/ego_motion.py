"""
Ego-Motion Estimation and Compensation Module

Estimates camera (ego-vehicle) motion using optical flow on background pixels
and compensates object velocities to get true ground-relative movement.
"""

# === IMPORTS ===
import numpy as np  # Numpy - math library for arrays
import cv2  # OpenCV - computer vision library
from typing import Tuple, Optional, List, Dict  # Type hints for better code readability
# Tuple: fixed-size collection like (x, y)
# Optional: means value can be the type OR None
# List: ordered collection [item1, item2, ...]
# Dict: key-value pairs {key: value}
from collections import deque  # Double-ended queue - fast add/remove from both ends


# === CLASS: Ego-Motion Estimator ===
class EgoMotionEstimator:  # Define class (blueprint for objects)
    """
    Estimates ego-vehicle motion using optical flow on static background.
    
    The key insight: Background pixels (buildings, road markings) have no
    actual movement. Their apparent motion = camera motion.
    """
    
    # Constructor - runs when creating new object
    def __init__(  # Define initialization method
        self,  # Reference to the object itself
        history_size: int = 5,  # Number of frames to average (default 5)
        flow_quality: str = 'medium'  # Quality setting (default 'medium')
    ):
        """
        Args:
            history_size: Number of frames to average for stable estimation
            flow_quality: 'low', 'medium', 'high' - affects computational cost
        """
        self.history_size = history_size  # Store as object attribute
        # Create deque (queue) to store velocity history
        self.ego_velocity_history = deque(maxlen=history_size)  # Auto-removes oldest when full
        self.prev_gray = None  # Will store previous frame (initially None)
        
        # Optical flow parameters based on quality setting
        # These control how the algorithm works
        if flow_quality == 'low':  # Fast but less accurate
            self.pyr_scale = 0.5  # Pyramid scale (how much to downsample each level)
            self.levels = 2  # Number of pyramid levels (fewer = faster)
            self.winsize = 10  # Window size for averaging (smaller = faster)
            self.iterations = 2  # Iterations per level (fewer = faster)
        elif flow_quality == 'medium':  # Balanced
            self.pyr_scale = 0.5  # Same scale
            self.levels = 3  # More levels = better accuracy
            self.winsize = 15  # Moderate window
            self.iterations = 3  # More iterations = better accuracy
        else:  # flow_quality == 'high' - Slow but accurate
            self.pyr_scale = 0.5  # Same scale
            self.levels = 5  # Most levels
            self.winsize = 21  # Largest window
            self.iterations = 5  # Most iterations
        
        # Polynomial expansion parameters (for the algorithm)
        self.poly_n = 5  # Size of pixel neighborhood
        self.poly_sigma = 1.1  # Gaussian standard deviation
    
    # Main method - estimates camera movement
    def estimate_ego_motion(  # Method definition
        self,  # The object
        frame: np.ndarray,  # Input frame (numpy array = image)
        foreground_mask: Optional[np.ndarray] = None  # Optional mask (can be None)
    ) -> Tuple[float, float]:  # Returns tuple of two floats (vx, vy)
        """
        Estimate ego-vehicle velocity from optical flow.
        
        Args:
            frame: Current frame (BGR color image)
            foreground_mask: Optional binary mask where 1=moving object, 0=static
                           Used to exclude moving objects from calculation
        
        Returns:
            (vx, vy): Ego-vehicle velocity in pixels/frame
        """
        # Convert frame to grayscale (optical flow needs grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR to Gray conversion
        
        # On first frame, we can't calculate motion (need 2 frames to compare)
        if self.prev_gray is None:  # If no previous frame exists
            self.prev_gray = gray  # Store this frame
            return (0.0, 0.0)  # Return zero velocity
        
        # Calculate dense optical flow (motion vector for every pixel)
        flow = cv2.calcOpticalFlowFarneback(  # OpenCV's Farneback method
            self.prev_gray,  # Previous frame
            gray,  # Current frame
            None,  # Output array (None = create new)
            pyr_scale=self.pyr_scale,  # Pyramid scale factor
            levels=self.levels,  # Number of pyramid levels
            winsize=self.winsize,  # Averaging window size
            iterations=self.iterations,  # Iterations at each level
            poly_n=self.poly_n,  # Polynomial neighborhood size
            poly_sigma=self.poly_sigma,  # Gaussian standard deviation
            flags=0  # Operation flags (0 = default)
        )
        # Returns 3D array: flow[y, x] = [vx, vy] for each pixel
        
        # Create background mask (identify which pixels are static)
        if foreground_mask is not None:  # If mask was provided
            # Invert mask: we want background (static) pixels
            # 1 - foreground_mask: changes 1→0 and 0→1
            background_mask = (1 - foreground_mask).astype(np.uint8)  # Convert to 8-bit integer
        else:  # No mask provided, use simple heuristic
            # Create all-zeros mask (same size as frame)
            background_mask = np.zeros(gray.shape, dtype=np.uint8)  # All zeros (black)
            h, w = gray.shape  # Get height and width
            # Set bottom half to 1 (assume road/ground is in bottom half)
            background_mask[int(h * 0.5):, :] = 1  # From 50% down to bottom, all columns
            # [:, :] means [rows, columns]
            # int(h * 0.5): means starting from row at 50% height
            # :, : means "all columns"
        
        # Extract flow vectors from background pixels only
        flow_x = flow[:, :, 0]  # X-component of flow (all rows, all columns, channel 0)
        flow_y = flow[:, :, 1]  # Y-component of flow (channel 1)
        
        # Calculate median flow (more robust than mean)
        # Median is less affected by outliers (moving objects we missed)
        bg_flow_x = flow_x[background_mask > 0]  # Get flow_x where mask is 1
        # [background_mask > 0] is boolean indexing - selects only True positions
        bg_flow_y = flow_y[background_mask > 0]  # Get flow_y where mask is 1
        
        # Check if we have any background pixels
        if len(bg_flow_x) > 0:  # If array is not empty
            ego_vx = np.median(bg_flow_x)  # Median of X velocities
            # Median = middle value when sorted (robust to outliers)
            ego_vy = np.median(bg_flow_y)  # Median of Y velocities
        else:  # No background pixels found
            ego_vx, ego_vy = 0.0, 0.0  # Default to zero velocity
        
        # Add to history for temporal smoothing
        self.ego_velocity_history.append((ego_vx, ego_vy))  # Add tuple to queue
        # If queue is full, oldest item is automatically removed
        
        # Return smoothed estimate (average over history)
        # List comprehension to extract x velocities from history
        smoothed_vx = np.mean([v[0] for v in self.ego_velocity_history])
        # [v[0] for v in ...] creates list of first elements
        # np.mean() calculates average
        smoothed_vy = np.mean([v[1] for v in self.ego_velocity_history])
        # [v[1] for v in ...] creates list of second elements
        
        # Update previous frame for next iteration
        self.prev_gray = gray  # Current becomes previous
        
        return (smoothed_vx, smoothed_vy)  # Return smoothed velocity tuple
    
    # Helper method - compensates object velocity
    def compensate_velocity(  # Method definition
        self,  # The object
        perceived_velocity: Tuple[float, float],  # What we observe in image
        ego_velocity: Tuple[float, float]  # Camera movement
    ) -> Tuple[float, float]:  # Returns actual velocity
        """
        Calculate true ground-relative velocity.
        
        Formula: V_actual = V_perceived - V_ego
        
        Note: We subtract because ego velocity is the background motion,
        which is opposite to ego-vehicle's actual movement.
        
        Args:
            perceived_velocity: Observed object velocity in image coordinates
            ego_velocity: Estimated camera/ego-vehicle velocity
        
        Returns:
            (vx_actual, vy_actual): True ground-relative velocity
        """
        # Subtract ego-motion from perceived motion
        vx_actual = perceived_velocity[0] - ego_velocity[0]  # X component
        vy_actual = perceived_velocity[1] - ego_velocity[1]  # Y component
        
        return (vx_actual, vy_actual)  # Return compensated velocity
    
    # Getter method - retrieves current ego velocity
    def get_current_ego_velocity(self) -> Tuple[float, float]:  # Returns velocity tuple
        """Get current smoothed ego-velocity estimate."""
        # Check if history is empty
        if not self.ego_velocity_history:  # If deque is empty (False when empty)
            return (0.0, 0.0)  # Return zero velocity
        
        # Calculate average from history
        vx = np.mean([v[0] for v in self.ego_velocity_history])  # Average X velocity
        vy = np.mean([v[1] for v in self.ego_velocity_history])  # Average Y velocity
        return (vx, vy)  # Return tuple
    
    # Visualization method - draws optical flow as arrows
    def visualize_flow(  # Method definition
        self,  # The object
        frame: np.ndarray,  # Input frame
        step: int = 16,  # Spacing between arrows (default 16 pixels)
        scale: float = 3.0  # Arrow length multiplier (default 3.0)
    ) -> np.ndarray:  # Returns frame with arrows
        """
        Visualize optical flow as arrows on frame.
        
        Args:
            frame: Input frame (BGR image)
            step: Spacing between arrows in pixels (larger = fewer arrows)
            scale: Arrow length multiplier (larger = longer arrows)
        
        Returns:
            Frame with flow arrows drawn on it
        """
        # Need previous frame to calculate flow
        if self.prev_gray is None:  # No previous frame
            return frame  # Return unchanged
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR to Gray
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(  # Same method as before
            self.prev_gray, gray, None,  # Previous, current, output
            self.pyr_scale, self.levels, self.winsize,  # Parameters
            self.iterations, self.poly_n, self.poly_sigma, 0  # More parameters
        )
        
        vis = frame.copy()  # Create copy for drawing
        h, w = frame.shape[:2]  # Get height and width
        # frame.shape returns (height, width, channels)
        # [:2] takes first two elements (height, width)
        
        # Draw flow arrows at grid positions
        # range(start, stop, step) creates sequence
        for y in range(0, h, step):  # Rows: 0, step, 2*step, ..., up to h
            for x in range(0, w, step):  # Columns: 0, step, 2*step, ..., up to w
                fx, fy = flow[y, x]  # Get flow vector at this position
                # flow[y, x] returns [vx, vy]
                
                # Only draw significant flow (ignore small movements)
                if np.linalg.norm([fx, fy]) > 0.5:  # If magnitude > 0.5
                    # np.linalg.norm() calculates vector magnitude: sqrt(fx² + fy²)
                    
                    # Calculate arrow end point
                    end_x = int(x + fx * scale)  # Start x + scaled flow
                    end_y = int(y + fy * scale)  # Start y + scaled flow
                    
                    # Draw arrow
                    cv2.arrowedLine(  # OpenCV function for drawing arrows
                        vis,  # Image to draw on
                        (x, y),  # Start point
                        (end_x, end_y),  # End point
                        (0, 255, 0),  # Color (green in BGR)
                        1,  # Thickness
                        tipLength=0.3  # Arrow tip size (30% of arrow length)
                    )
        
        return vis  # Return visualized image


# === CLASS: Relative Velocity Tracker ===
class RelativeVelocityTracker:  # Define class
    """
    Tracks object velocities relative to ground (ego-motion compensated).
    """
    
    # Constructor
    def __init__(self):  # No parameters except self
        # Dictionary to store position history for each track
        self.track_histories: Dict[int, deque] = {}  # Empty dictionary
        # Key: track_id (int), Value: deque of positions
        
        # Dictionary to store perceived velocities
        self.perceived_velocities: Dict[int, Tuple[float, float]] = {}  # Empty dict
        # Key: track_id, Value: (vx, vy) before ego-motion compensation
        
        # Dictionary to store actual velocities
        self.actual_velocities: Dict[int, Tuple[float, float]] = {}  # Empty dict
        # Key: track_id, Value: (vx, vy) after ego-motion compensation
        
        self.max_history = 20  # Keep last 20 positions
    
    # Update method - adds new position and calculates velocities
    def update(  # Method definition
        self,  # The object
        track_id: int,  # Which object/track
        position: Tuple[float, float],  # Current position (x, y)
        ego_velocity: Tuple[float, float]  # Camera velocity (vx, vy)
    ):
        """
        Update track with ego-motion compensation.
        
        Args:
            track_id: Unique ID number for this object
            position: Current (x, y) position
            ego_velocity: Current ego-vehicle velocity from optical flow
        """
        # Initialize history if this is a new track
        if track_id not in self.track_histories:  # If key doesn't exist
            # Create new deque with max length
            self.track_histories[track_id] = deque(maxlen=self.max_history)
        
        # Get history for this track
        history = self.track_histories[track_id]  # Reference to the deque
        history.append(position)  # Add new position to end
        
        # Calculate perceived velocity (need at least 2 positions)
        if len(history) >= 2:  # If we have 2 or more positions
            prev_pos = history[-2]  # Second-to-last position
            # Negative indexing: -1 is last, -2 is second-to-last
            curr_pos = history[-1]  # Last position
            
            # Calculate velocity by subtracting positions
            perceived_vx = curr_pos[0] - prev_pos[0]  # Change in X
            perceived_vy = curr_pos[1] - prev_pos[1]  # Change in Y
            
            # Store perceived velocity
            self.perceived_velocities[track_id] = (perceived_vx, perceived_vy)
            
            # Compensate for ego-motion (subtract camera movement)
            actual_vx = perceived_vx - ego_velocity[0]  # Actual X velocity
            actual_vy = perceived_vy - ego_velocity[1]  # Actual Y velocity
            
            # Store actual velocity
            self.actual_velocities[track_id] = (actual_vx, actual_vy)
        else:  # Not enough history yet (only 1 position)
            # Default to zero velocity
            self.perceived_velocities[track_id] = (0.0, 0.0)
            self.actual_velocities[track_id] = (0.0, 0.0)
    
    # Getter method - returns actual velocity
    def get_actual_velocity(self, track_id: int) -> Tuple[float, float]:
        """Get ego-compensated actual velocity for track."""
        # .get(key, default) returns value or default if key not found
        return self.actual_velocities.get(track_id, (0.0, 0.0))
    
    # Getter method - returns perceived velocity
    def get_perceived_velocity(self, track_id: int) -> Tuple[float, float]:
        """Get raw perceived velocity (before ego-motion compensation)."""
        return self.perceived_velocities.get(track_id, (0.0, 0.0))
    
    # Getter method - returns speed magnitude
    def get_speed(self, track_id: int, relative: bool = True) -> float:
        """
        Get speed magnitude (scalar, not vector).
        
        Args:
            track_id: Track ID to query
            relative: If True, return ego-compensated speed
                     If False, return perceived speed
        
        Returns:
            Speed in pixels/frame (magnitude of velocity vector)
        """
        # Choose which velocity to use
        if relative:  # Ego-compensated
            vel = self.actual_velocities.get(track_id, (0.0, 0.0))
        else:  # Raw perceived
            vel = self.perceived_velocities.get(track_id, (0.0, 0.0))
        
        # Calculate magnitude (length of velocity vector)
        return np.linalg.norm(vel)  # sqrt(vx² + vy²)


# === CLASS: BEV Ego-Motion Integrator ===
class BEVEgoMotionIntegrator:  # Define class
    """
    Integrates ego-motion estimation with BEV (Bird's Eye View) transformation.
    
    Advantages:
    - Ego-motion in BEV = real-world motion (meters/second)
    - Constant scale across entire image (no perspective distortion)
    - Easier to filter unrealistic speeds
    """
    
    # Constructor
    def __init__(self, bev_transformer, pixels_per_meter: float = 10.0):
        """
        Args:
            bev_transformer: BEVTransformer object (handles coordinate transforms)
            pixels_per_meter: Calibration parameter - how many pixels = 1 meter in BEV
        """
        self.bev_transformer = bev_transformer  # Store transformer object
        self.pixels_per_meter = pixels_per_meter  # Store calibration
        self.fps = 30.0  # Assume 30 frames per second
    
    # Method - transforms velocity from image to BEV space
    def image_velocity_to_bev(  # Method definition
        self,  # The object
        velocity: Tuple[float, float],  # Velocity in image coordinates
        position: Tuple[float, float]  # Current position (needed for transform)
    ) -> Tuple[float, float]:  # Returns velocity in BEV coordinates
        """
        Transform image-space velocity to BEV velocity.
        
        This is approximate - we transform the start and end points,
        then calculate velocity from the difference.
        """
        # Transform current position to BEV
        pos_bev = self.bev_transformer.image_to_bev(position)
        
        # Calculate end position in image space
        end_pos_image = (position[0] + velocity[0], position[1] + velocity[1])
        # Add velocity vector to position
        
        # Transform end position to BEV
        end_pos_bev = self.bev_transformer.image_to_bev(end_pos_image)
        
        # BEV velocity = difference between BEV positions
        vx_bev = end_pos_bev[0] - pos_bev[0]  # X component
        vy_bev = end_pos_bev[1] - pos_bev[1]  # Y component
        
        return (vx_bev, vy_bev)  # Return BEV velocity
    
    # Method - converts BEV velocity to metric units
    def bev_velocity_to_metric(  # Method definition
        self,  # The object
        bev_velocity: Tuple[float, float]  # Velocity in BEV (pixels/frame)
    ) -> Tuple[float, float]:  # Returns velocity in meters/second
        """
        Convert BEV velocity (pixels/frame) to metric (m/s).
        
        Args:
            bev_velocity: Velocity in BEV space (pixels per frame)
        
        Returns:
            (vx, vy) in meters per second
        """
        # Convert X component
        # (pixels/frame) / (pixels/meter) * (frames/second) = meters/second
        vx_ms = (bev_velocity[0] / self.pixels_per_meter) * self.fps
        
        # Convert Y component
        vy_ms = (bev_velocity[1] / self.pixels_per_meter) * self.fps
        
        return (vx_ms, vy_ms)  # Return in m/s
    
    # Method - converts m/s to km/h
    def get_speed_kmh(self, velocity_ms: Tuple[float, float]) -> float:
        """Convert velocity in m/s to speed in km/h."""
        speed_ms = np.linalg.norm(velocity_ms)  # Calculate magnitude (m/s)
        # sqrt(vx² + vy²)
        return speed_ms * 3.6  # Convert m/s to km/h (multiply by 3.6)
        # 1 m/s = 3.6 km/h


# === EXAMPLE USAGE (runs when script is executed directly) ===
if __name__ == "__main__":  # Only runs if this file is the main program
    # Create estimator object
    ego_estimator = EgoMotionEstimator(history_size=5, flow_quality='medium')
    velocity_tracker = RelativeVelocityTracker()  # Create tracker object
    
    # Simulate video processing
    cap = cv2.VideoCapture("dashcam.mp4")  # Open video file
    
    # Main loop
    while cap.isOpened():  # While video is open
        ret, frame = cap.read()  # Read next frame
        # ret = True/False (success), frame = image array
        
        if not ret:  # If reading failed
            break  # Exit loop
        
        # Estimate ego-motion from this frame
        ego_velocity = ego_estimator.estimate_ego_motion(frame)
        
        print(f"Ego velocity: {ego_velocity}")  # Print result
        
        # For each tracked object (example with one object)
        track_id = 1  # Track ID number
        object_position = (500, 400)  # Example position
        
        # Update velocity tracker
        velocity_tracker.update(track_id, object_position, ego_velocity)
        
        # Get velocities
        actual_vel = velocity_tracker.get_actual_velocity(track_id)
        perceived_vel = velocity_tracker.get_perceived_velocity(track_id)
        
        print(f"Perceived: {perceived_vel}, Actual: {actual_vel}")
        
        # Visualize optical flow
        vis = ego_estimator.visualize_flow(frame, step=20)
        # Draw arrows every 20 pixels
        cv2.imshow("Optical Flow", vis)  # Show visualization
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait 1ms, check for 'q'
            break  # Exit loop
    
    # Cleanup
    cap.release()  # Close video file
    cv2.destroyAllWindows()  # Close all windows
