"""
Context-Aware Trajectory Prediction Module

Incorporates environmental context (traffic lights, semantic zones) into
trajectory prediction for intent-based switching and constraint-aware forecasting.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, NamedTuple
from enum import Enum
from dataclasses import dataclass
from .trajectory_predictor import KalmanFilterPredictor


class TrafficLightState(Enum):
    """Traffic light states."""
    RED = 0
    YELLOW = 1
    GREEN = 2
    UNKNOWN = 3


class ZoneType(Enum):
    """Semantic zone types."""
    ROAD = 0
    SIDEWALK = 1
    CROSSWALK = 2
    INTERSECTION = 3
    PARKING = 4
    UNKNOWN = 5


@dataclass
class TrafficLight:
    """Traffic light information."""
    position: Tuple[float, float]  # (x, y) in image coordinates
    state: TrafficLightState
    confidence: float
    detection_time: int  # Frame number
    
    def is_valid(self, current_frame: int, max_age: int = 10) -> bool:
        """Check if light detection is still valid (not too old)."""
        return (current_frame - self.detection_time) < max_age


@dataclass
class EnvironmentalContext:
    """
    Environmental context for a specific location.
    
    Stores information about traffic lights, zone types, and other
    environmental factors that affect trajectory prediction.
    """
    traffic_lights: List[TrafficLight]
    zone_type: ZoneType
    zone_mask: Optional[np.ndarray]  # Binary mask of allowed movement area
    timestamp: int  # Frame number
    
    def get_nearest_traffic_light(
        self, 
        position: Tuple[float, float],
        max_distance: float = 200.0
    ) -> Optional[TrafficLight]:
        """Find nearest traffic light within max_distance."""
        nearest = None
        min_dist = max_distance
        
        for light in self.traffic_lights:
            dist = np.linalg.norm(
                np.array(position) - np.array(light.position)
            )
            if dist < min_dist:
                min_dist = dist
                nearest = light
        
        return nearest


class IntentModel:
    """
    Models vehicle/pedestrian intent based on environmental context.
    
    Determines whether an agent will:
    - Continue at constant velocity
    - Decelerate to stop
    - Accelerate through
    - Change lanes/direction
    """
    
    @staticmethod
    def calculate_stop_probability(
        vehicle_position: Tuple[float, float],
        vehicle_velocity: Tuple[float, float],
        traffic_light: TrafficLight,
        distance_to_intersection: float
    ) -> float:
        """
        Calculate probability of vehicle stopping based on traffic light.
        
        Args:
            vehicle_position: Current (x, y) position
            vehicle_velocity: Current (vx, vy) velocity
            traffic_light: Traffic light object
            distance_to_intersection: Distance to stop line
            
        Returns:
            Probability [0, 1] that vehicle will stop
        """
        speed = np.linalg.norm(vehicle_velocity)
        
        # Red light logic
        if traffic_light.state == TrafficLightState.RED:
            # Closer vehicles are more likely to stop
            if distance_to_intersection < 50:  # Within 50 pixels
                return 0.95
            elif distance_to_intersection < 100:
                return 0.85
            else:
                return 0.7
        
        # Yellow light logic - depends on distance and speed
        elif traffic_light.state == TrafficLightState.YELLOW:
            # Calculate if vehicle can safely stop
            stopping_distance = (speed ** 2) / (2 * 0.5)  # Assume 0.5 px/frame² decel
            
            if distance_to_intersection > stopping_distance * 1.5:
                return 0.8  # Safe to stop, likely will
            elif distance_to_intersection < stopping_distance * 0.7:
                return 0.2  # Too close to stop, likely to go through
            else:
                return 0.5  # Ambiguous - could go either way
        
        # Green light
        elif traffic_light.state == TrafficLightState.GREEN:
            return 0.1  # Low probability of stopping
        
        return 0.0  # Unknown state
    
    @staticmethod
    def calculate_deceleration_profile(
        current_velocity: Tuple[float, float],
        distance_to_stop: float,
        comfort_decel: float = 0.3
    ) -> List[Tuple[float, float]]:
        """
        Generate velocity profile for smooth deceleration to stop.
        
        Args:
            current_velocity: Current (vx, vy) velocity
            distance_to_stop: Distance to stop point
            comfort_decel: Comfortable deceleration rate
            
        Returns:
            List of velocity vectors over time
        """
        speed = np.linalg.norm(current_velocity)
        if speed < 0.01:
            return [(0, 0)]
        
        # Calculate stopping time
        stop_time = speed / comfort_decel
        num_steps = int(stop_time) + 1
        
        # Direction unit vector
        direction = np.array(current_velocity) / speed
        
        # Generate deceleration profile
        velocities = []
        for t in range(num_steps):
            remaining_speed = max(0, speed - comfort_decel * t)
            vel = direction * remaining_speed
            velocities.append((vel[0], vel[1]))
        
        return velocities


class ContextAwarePredictor:
    """
    Enhanced trajectory predictor with environmental context awareness.
    """
    
    def __init__(self):
        self.base_predictor = KalmanFilterPredictor()
        self.environmental_contexts: Dict[int, EnvironmentalContext] = {}
        self.track_zones: Dict[int, ZoneType] = {}
    
    def update(
        self, 
        track_id: int, 
        position: Tuple[float, float],
        context: Optional[EnvironmentalContext] = None
    ):
        """
        Update tracker with new position and optional environmental context.
        """
        self.base_predictor.update(track_id, position)
        
        if context is not None:
            self.environmental_contexts[track_id] = context
            self.track_zones[track_id] = context.zone_type
    
    def predict_with_context(
        self,
        track_id: int,
        num_steps: int = 60,
        context: Optional[EnvironmentalContext] = None,
        object_class: str = 'car'
    ) -> Dict[str, any]:
        """
        Predict trajectory with environmental context.
        
        Returns a dictionary with:
        - 'primary': Most likely trajectory
        - 'alternative': Alternative trajectory (e.g., if yellow light)
        - 'probabilities': Probability weights for each trajectory
        - 'intent': Predicted intent (CONTINUE, STOP, ACCELERATE)
        - 'metadata': Additional prediction information
        """
        # Get base prediction
        base_prediction = self.base_predictor.predict(track_id, num_steps)
        
        if not base_prediction or context is None:
            return {
                'primary': base_prediction,
                'alternative': None,
                'probabilities': [1.0],
                'intent': 'CONTINUE',
                'metadata': {}
            }
        
        # Get current state
        if track_id not in self.base_predictor.filters:
            return {
                'primary': base_prediction,
                'alternative': None,
                'probabilities': [1.0],
                'intent': 'CONTINUE',
                'metadata': {}
            }
        
        state = self.base_predictor.filters[track_id]['state']
        position = (state[0], state[1])
        velocity = (state[2], state[3])
        
        # Check for nearby traffic light
        traffic_light = context.get_nearest_traffic_light(position)
        
        if traffic_light is None or object_class not in ['car', 'truck', 'bus', 'motorcycle']:
            # No traffic light influence or not a vehicle
            pruned = self._prune_trajectory_by_zone(
                base_prediction, 
                context.zone_mask,
                object_class
            )
            return {
                'primary': pruned,
                'alternative': None,
                'probabilities': [1.0],
                'intent': 'CONTINUE',
                'metadata': {}
            }
        
        # Calculate distance to intersection (approximate)
        distance_to_intersection = np.linalg.norm(
            np.array(position) - np.array(traffic_light.position)
        )
        
        # Calculate stop probability
        stop_prob = IntentModel.calculate_stop_probability(
            position, velocity, traffic_light, distance_to_intersection
        )
        
        # Generate predictions based on intent
        if traffic_light.state == TrafficLightState.RED and stop_prob > 0.5:
            # High probability of stopping - generate stopping trajectory
            stopping_traj = self._generate_stopping_trajectory(
                position, velocity, distance_to_intersection, num_steps
            )
            return {
                'primary': stopping_traj,
                'alternative': base_prediction,
                'probabilities': [stop_prob, 1 - stop_prob],
                'intent': 'STOP',
                'metadata': {
                    'traffic_light': traffic_light.state.name,
                    'stop_distance': distance_to_intersection
                }
            }
        
        elif traffic_light.state == TrafficLightState.YELLOW:
            # Ambiguous - generate both stopping and continuing trajectories
            stopping_traj = self._generate_stopping_trajectory(
                position, velocity, distance_to_intersection, num_steps
            )
            continue_traj = base_prediction
            
            return {
                'primary': continue_traj if stop_prob < 0.5 else stopping_traj,
                'alternative': stopping_traj if stop_prob < 0.5 else continue_traj,
                'probabilities': [1 - stop_prob, stop_prob] if stop_prob < 0.5 else [stop_prob, 1 - stop_prob],
                'intent': 'AMBIGUOUS',
                'metadata': {
                    'traffic_light': traffic_light.state.name,
                    'stop_probability': stop_prob
                }
            }
        
        else:
            # Green or unknown - continue with base prediction
            pruned = self._prune_trajectory_by_zone(
                base_prediction,
                context.zone_mask,
                object_class
            )
            return {
                'primary': pruned,
                'alternative': None,
                'probabilities': [1.0],
                'intent': 'CONTINUE',
                'metadata': {
                    'traffic_light': traffic_light.state.name if traffic_light else 'NONE'
                }
            }
    
    def _generate_stopping_trajectory(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        distance_to_stop: float,
        num_steps: int
    ) -> List[Tuple[float, float]]:
        """Generate trajectory that smoothly decelerates to a stop."""
        speed = np.linalg.norm(velocity)
        
        if speed < 0.01:
            # Already stopped
            return [position] * num_steps
        
        # Calculate deceleration needed
        # v² = u² + 2as, where v=0 (final), u=speed (initial)
        # a = -u² / (2s)
        decel = (speed ** 2) / (2 * max(distance_to_stop, 10))
        decel = min(decel, 1.0)  # Cap deceleration
        
        # Generate trajectory
        trajectory = []
        current_pos = np.array(position)
        current_vel = np.array(velocity)
        
        # Direction unit vector
        direction = current_vel / speed if speed > 0 else np.array([0, 0])
        
        for step in range(num_steps):
            # Update velocity (decelerate)
            current_speed = max(0, speed - decel * step)
            current_vel = direction * current_speed
            
            # Update position
            current_pos = current_pos + current_vel
            trajectory.append((current_pos[0], current_pos[1]))
            
            # Stop if reached target or velocity is zero
            if current_speed < 0.01:
                # Keep position constant for remaining steps
                trajectory.extend([trajectory[-1]] * (num_steps - step - 1))
                break
        
        return trajectory
    
    def _prune_trajectory_by_zone(
        self,
        trajectory: List[Tuple[float, float]],
        zone_mask: Optional[np.ndarray],
        object_class: str
    ) -> List[Tuple[float, float]]:
        """
        Prune trajectory points that fall outside valid zones.
        
        For example:
        - Cars should stay on roads
        - Pedestrians should stay on sidewalks/crosswalks
        """
        if zone_mask is None or len(trajectory) == 0:
            return trajectory
        
        mask_h, mask_w = zone_mask.shape
        pruned = []
        
        for point in trajectory:
            x, y = int(point[0]), int(point[1])
            
            # Check bounds
            if 0 <= x < mask_w and 0 <= y < mask_h:
                # Check if point is in valid zone
                if zone_mask[y, x] > 0:  # Assuming mask is binary
                    pruned.append(point)
                else:
                    # Invalid zone - stop prediction here
                    break
            else:
                # Out of bounds
                break
        
        return pruned if pruned else trajectory


class UncertaintyVisualizer:
    """Visualizes prediction uncertainty and multiple trajectories."""
    
    @staticmethod
    def draw_multi_modal_trajectory(
        frame: np.ndarray,
        prediction_result: Dict[str, any],
        base_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw primary and alternative trajectories with uncertainty.
        
        Args:
            frame: Input frame
            prediction_result: Result from predict_with_context()
            base_color: Base color for trajectories
            
        Returns:
            Frame with trajectories drawn
        """
        overlay = frame.copy()
        
        primary = prediction_result['primary']
        alternative = prediction_result['alternative']
        probabilities = prediction_result['probabilities']
        intent = prediction_result['intent']
        
        # Draw primary trajectory
        if primary and len(primary) > 1:
            prob = probabilities[0]
            color = tuple(int(c * prob) for c in base_color)
            
            # Draw line
            pts = np.array(primary, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, color, 2)
            
            # Draw end point
            end_pt = (int(primary[-1][0]), int(primary[-1][1]))
            cv2.circle(overlay, end_pt, 8, color, -1)
            
            # Label intent
            cv2.putText(
                overlay, intent, (end_pt[0] + 10, end_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Draw alternative trajectory if exists
        if alternative and len(alternative) > 1:
            prob = probabilities[1] if len(probabilities) > 1 else 0.5
            alt_color = (255, 165, 0)  # Orange for alternative
            color = tuple(int(c * prob) for c in alt_color)
            
            # Draw dashed line
            pts = np.array(alternative, dtype=np.int32)
            for i in range(0, len(pts) - 1, 3):  # Dashed effect
                if i + 1 < len(pts):
                    cv2.line(overlay, tuple(pts[i]), tuple(pts[i + 1]), color, 2)
            
            # Draw end point
            end_pt = (int(alternative[-1][0]), int(alternative[-1][1]))
            cv2.circle(overlay, end_pt, 6, color, 2)
            
            # Label probability
            prob_text = f"{prob:.0%}"
            cv2.putText(
                overlay, prob_text, (end_pt[0] + 10, end_pt[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        # Blend
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        return frame
    
    @staticmethod
    def draw_uncertainty_cone(
        frame: np.ndarray,
        trajectory: List[Tuple[float, float]],
        uncertainty_growth: float = 2.0,
        color: Tuple[int, int, int] = (200, 200, 255)
    ) -> np.ndarray:
        """
        Draw uncertainty cone that widens over prediction time.
        
        Represents growing uncertainty in future predictions.
        """
        if len(trajectory) < 2:
            return frame
        
        overlay = frame.copy()
        
        for i in range(len(trajectory) - 1):
            pt1 = trajectory[i]
            pt2 = trajectory[i + 1]
            
            # Calculate uncertainty radius (grows with time)
            radius = int(uncertainty_growth * (i + 1))
            
            # Draw circle at this point
            alpha = 1.0 - (i / len(trajectory))
            fade_color = tuple(int(c * alpha * 0.3) for c in color)
            
            cv2.circle(overlay, (int(pt2[0]), int(pt2[1])), radius, fade_color, -1)
        
        # Blend
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame


# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = ContextAwarePredictor()
    
    # Simulate vehicle approaching red light
    track_id = 1
    
    # Update position history
    for i in range(30):
        x = 100 + i * 5
        y = 200
        predictor.update(track_id, (x, y))
    
    # Create environmental context with red light
    red_light = TrafficLight(
        position=(250, 200),
        state=TrafficLightState.RED,
        confidence=0.95,
        detection_time=30
    )
    
    context = EnvironmentalContext(
        traffic_lights=[red_light],
        zone_type=ZoneType.ROAD,
        zone_mask=None,
        timestamp=30
    )
    
    # Predict with context
    result = predictor.predict_with_context(
        track_id, num_steps=60, context=context, object_class='car'
    )
    
    print(f"Intent: {result['intent']}")
    print(f"Probabilities: {result['probabilities']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Primary trajectory length: {len(result['primary'])}")
    if result['alternative']:
        print(f"Alternative trajectory length: {len(result['alternative'])}")
