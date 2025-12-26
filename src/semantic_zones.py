"""
Semantic Segmentation Utilities for Zone-Based Constraints

Provides tools for:
- Creating road/sidewalk masks
- Zone-based trajectory validation
- Integration with semantic segmentation models
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from enum import Enum


class SemanticZone(Enum):
    """Semantic zone categories."""
    ROAD = 1
    SIDEWALK = 2
    CROSSWALK = 3
    INTERSECTION = 4
    PARKING = 5
    GRASS = 6
    BUILDING = 7


class ZoneMaskGenerator:
    """
    Generates and manages semantic zone masks.
    
    Can work with:
    - Manual polygon annotation
    - Pre-computed segmentation masks
    - Real-time segmentation model outputs
    """
    
    def __init__(self, image_width: int, image_height: int):
        self.width = image_width
        self.height = image_height
        self.road_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        self.sidewalk_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        self.crosswalk_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    def create_road_mask_from_polygon(
        self, 
        polygon_points: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Create road mask from polygon points.
        
        Args:
            polygon_points: List of (x, y) points defining road boundary
            
        Returns:
            Binary mask where 1 = road, 0 = not road
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        self.road_mask = mask
        return mask
    
    def create_sidewalk_mask_from_polygon(
        self,
        polygon_points: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Create sidewalk mask from polygon points."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        self.sidewalk_mask = mask
        return mask
    
    def load_from_segmentation_model(
        self,
        segmentation_output: np.ndarray,
        class_mapping: dict
    ) -> dict:
        """
        Load masks from semantic segmentation model output.
        
        Args:
            segmentation_output: Model output (H x W) with class IDs
            class_mapping: Dict mapping class IDs to zone types
                          e.g., {0: 'road', 1: 'sidewalk', 2: 'building'}
        
        Returns:
            Dictionary of zone masks
        """
        masks = {}
        
        for class_id, zone_name in class_mapping.items():
            mask = (segmentation_output == class_id).astype(np.uint8) * 255
            
            if zone_name == 'road':
                self.road_mask = mask
            elif zone_name == 'sidewalk':
                self.sidewalk_mask = mask
            elif zone_name == 'crosswalk':
                self.crosswalk_mask = mask
            
            masks[zone_name] = mask
        
        return masks
    
    def get_zone_at_point(
        self, 
        point: Tuple[float, float]
    ) -> SemanticZone:
        """
        Determine semantic zone at a given point.
        
        Priority: Crosswalk > Road > Sidewalk
        """
        x, y = int(point[0]), int(point[1])
        
        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return SemanticZone.BUILDING  # Out of bounds
        
        # Check zones in priority order
        if self.crosswalk_mask[y, x] > 0:
            return SemanticZone.CROSSWALK
        elif self.road_mask[y, x] > 0:
            return SemanticZone.ROAD
        elif self.sidewalk_mask[y, x] > 0:
            return SemanticZone.SIDEWALK
        else:
            return SemanticZone.GRASS
    
    def is_valid_position(
        self,
        point: Tuple[float, float],
        object_class: str
    ) -> bool:
        """
        Check if position is valid for given object class.
        
        Rules:
        - Vehicles: Must be on road, crosswalk, or parking
        - Pedestrians: Can be on sidewalk, crosswalk, or road
        - Bicycles: Can be on road or sidewalk
        """
        zone = self.get_zone_at_point(point)
        
        vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
        
        if object_class in vehicle_classes:
            return zone in {
                SemanticZone.ROAD, 
                SemanticZone.CROSSWALK,
                SemanticZone.INTERSECTION,
                SemanticZone.PARKING
            }
        elif object_class == 'person':
            return zone in {
                SemanticZone.SIDEWALK,
                SemanticZone.CROSSWALK,
                SemanticZone.ROAD  # Pedestrians can cross roads
            }
        elif object_class == 'bicycle':
            return zone in {
                SemanticZone.ROAD,
                SemanticZone.SIDEWALK,
                SemanticZone.CROSSWALK
            }
        
        return True  # Unknown class, allow anywhere
    
    def visualize_masks(self, base_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create visualization of all zone masks.
        
        Returns:
            RGB image with colored zones
        """
        if base_image is not None:
            vis = base_image.copy()
        else:
            vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Create colored overlay
        overlay = np.zeros_like(vis)
        
        # Road = Green
        overlay[self.road_mask > 0] = [0, 255, 0]
        
        # Sidewalk = Blue
        overlay[self.sidewalk_mask > 0] = [255, 0, 0]
        
        # Crosswalk = Yellow
        overlay[self.crosswalk_mask > 0] = [0, 255, 255]
        
        # Blend
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        return vis


class InteractiveZoneAnnotator:
    """Interactive tool for annotating zones by clicking polygons."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.points = []
        self.zones = {
            'road': [],
            'sidewalk': [],
            'crosswalk': []
        }
        self.current_zone = 'road'
    
    def annotate(self) -> dict:
        """
        Launch interactive annotation tool.
        
        Instructions:
        - Click to add points
        - Press 'r' for road, 's' for sidewalk, 'c' for crosswalk
        - Press 'f' to finish current polygon
        - Press 'q' to quit
        
        Returns:
            Dictionary of zone polygons
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Cannot read video")
            return {}
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append([x, y])
                print(f"Added point: ({x}, {y}) to {self.current_zone}")
        
        cv2.namedWindow("Zone Annotation")
        cv2.setMouseCallback("Zone Annotation", mouse_callback)
        
        zone_colors = {
            'road': (0, 255, 0),
            'sidewalk': (255, 0, 0),
            'crosswalk': (0, 255, 255)
        }
        
        print("Zone Annotation Tool")
        print("--------------------")
        print("Click to add points")
        print("'r' = Road, 's' = Sidewalk, 'c' = Crosswalk")
        print("'f' = Finish polygon")
        print("'d' = Delete last point")
        print("'q' = Quit")
        
        while True:
            display = frame.copy()
            
            # Draw existing polygons
            for zone_name, polygons in self.zones.items():
                color = zone_colors[zone_name]
                for poly in polygons:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.polylines(display, [pts], True, color, 2)
                    cv2.fillPoly(display, [pts], color + (100,))  # Semi-transparent
            
            # Draw current points
            if self.points:
                pts = np.array(self.points, dtype=np.int32)
                color = zone_colors[self.current_zone]
                cv2.polylines(display, [pts], False, color, 2)
                
                for pt in self.points:
                    cv2.circle(display, tuple(pt), 5, color, -1)
            
            # Show current zone
            cv2.putText(
                display, f"Zone: {self.current_zone.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, zone_colors[self.current_zone], 2
            )
            
            cv2.imshow("Zone Annotation", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_zone = 'road'
                print("Switched to ROAD annotation")
            elif key == ord('s'):
                self.current_zone = 'sidewalk'
                print("Switched to SIDEWALK annotation")
            elif key == ord('c'):
                self.current_zone = 'crosswalk'
                print("Switched to CROSSWALK annotation")
            elif key == ord('f') and len(self.points) >= 3:
                self.zones[self.current_zone].append(self.points.copy())
                print(f"Finished {self.current_zone} polygon with {len(self.points)} points")
                self.points = []
            elif key == ord('d') and self.points:
                removed = self.points.pop()
                print(f"Deleted point: {removed}")
        
        cv2.destroyAllWindows()
        cap.release()
        
        return self.zones


# Example usage
if __name__ == "__main__":
    # Create mask generator
    mask_gen = ZoneMaskGenerator(1920, 1080)
    
    # Example: Create road mask from polygon (trapezoid)
    road_polygon = [
        (500, 600),   # Top-left
        (1400, 600),  # Top-right
        (1800, 1080), # Bottom-right
        (100, 1080)   # Bottom-left
    ]
    
    road_mask = mask_gen.create_road_mask_from_polygon(road_polygon)
    print(f"Created road mask: {road_mask.shape}")
    
    # Test point validation
    test_point = (960, 800)  # Center-bottom
    zone = mask_gen.get_zone_at_point(test_point)
    print(f"Point {test_point} is in zone: {zone}")
    
    is_valid = mask_gen.is_valid_position(test_point, 'car')
    print(f"Valid for car: {is_valid}")
