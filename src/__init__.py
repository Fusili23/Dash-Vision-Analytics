"""
Dash Vision Analytics - Advanced Trajectory Prediction System

A comprehensive trajectory prediction system for dashcam footage with:
- Ego-motion compensation
- Context-aware prediction (traffic lights, semantic zones)
- Bird's-eye view transformations
- Real-time visualization
"""

__version__ = "1.0.0"
__author__ = "Fusili23"

# Import main classes for easy access
from .trajectory_predictor import (
    ConstantVelocityPredictor,
    KalmanFilterPredictor,
    TrajectoryVisualizer,
    estimate_lane_direction
)

from .ego_motion import (
    EgoMotionEstimator,
    RelativeVelocityTracker,
    BEVEgoMotionIntegrator
)

from .context_aware_predictor import (
    ContextAwarePredictor,
    EnvironmentalContext,
    TrafficLight,
    TrafficLightState,
    UncertaintyVisualizer,
    IntentModel
)

from .semantic_zones import (
    ZoneMaskGenerator,
    SemanticZone,
    InteractiveZoneAnnotator
)

from .bev_transformer import (
    BEVTransformer,
    calibrate_bev_interactive
)

__all__ = [
    # Trajectory Prediction
    'ConstantVelocityPredictor',
    'KalmanFilterPredictor',
    'TrajectoryVisualizer',
    'estimate_lane_direction',
    
    # Ego-Motion
    'EgoMotionEstimator',
    'RelativeVelocityTracker',
    'BEVEgoMotionIntegrator',
    
    # Context-Aware
    'ContextAwarePredictor',
    'EnvironmentalContext',
    'TrafficLight',
    'TrafficLightState',
    'UncertaintyVisualizer',
    'IntentModel',
    
    # Semantic Zones
    'ZoneMaskGenerator',
    'SemanticZone',
    'InteractiveZoneAnnotator',
    
    # BEV Transformation
    'BEVTransformer',
    'calibrate_bev_interactive',
]
