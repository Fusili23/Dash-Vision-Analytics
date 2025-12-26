"""
Bird's Eye View (BEV) Transformation Example

Demonstrates how to perform perspective transformation for more accurate
trajectory prediction in metric space.
"""

import cv2
import numpy as np
from typing import Tuple, List


class BEVTransformer:
    """
    Handles perspective transformation between image and bird's-eye view.
    """
    
    def __init__(
        self, 
        image_width: int,
        image_height: int,
        bev_width: int = 400,
        bev_height: int = 800
    ):
        """
        Initialize BEV transformer.
        
        Args:
            image_width: Width of input video frame
            image_height: Height of input video frame
            bev_width: Width of BEV output (meters * scale)
            bev_height: Height of BEV output (meters * scale)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.bev_width = bev_width
        self.bev_height = bev_height
        
        # These should be calibrated for your specific camera
        # Default values work for typical dashcam perspective
        self.src_points = self._get_default_src_points()
        self.dst_points = self._get_default_dst_points()
        
        # Calculate transformation matrices
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
    
    def _get_default_src_points(self) -> np.ndarray:
        """
        Get default source points (image coordinates).
        
        Forms a trapezoid representing the road area in the image.
        """
        w, h = self.image_width, self.image_height
        
        return np.float32([
            [w * 0.35, h * 0.65],  # Top-left
            [w * 0.65, h * 0.65],  # Top-right
            [w * 0.95, h],         # Bottom-right
            [w * 0.05, h]          # Bottom-left
        ])
    
    def _get_default_dst_points(self) -> np.ndarray:
        """
        Get default destination points (BEV coordinates).
        
        Forms a rectangle in bird's-eye view.
        """
        return np.float32([
            [self.bev_width * 0.2, 0],
            [self.bev_width * 0.8, 0],
            [self.bev_width * 0.8, self.bev_height],
            [self.bev_width * 0.2, self.bev_height]
        ])
    
    def set_custom_region(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray = None
    ):
        """
        Set custom transformation region.
        
        Args:
            src_points: 4 points defining trapezoid in image (shape: 4x2)
            dst_points: Optional 4 points for BEV rectangle
        """
        self.src_points = src_points.astype(np.float32)
        
        if dst_points is not None:
            self.dst_points = dst_points.astype(np.float32)
        
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
    
    def image_to_bev(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform single point from image to BEV coordinates."""
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.M)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))
    
    def bev_to_image(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform single point from BEV to image coordinates."""
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.M_inv)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))
    
    def batch_image_to_bev(
        self, 
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Transform multiple points from image to BEV."""
        if not points:
            return []
        
        pts = np.array([[p] for p in points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.M)
        return [(float(p[0][0]), float(p[0][1])) for p in transformed]
    
    def batch_bev_to_image(
        self, 
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Transform multiple points from BEV to image."""
        if not points:
            return []
        
        pts = np.array([[p] for p in points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.M_inv)
        return [(float(p[0][0]), float(p[0][1])) for p in transformed]
    
    def warp_image_to_bev(self, image: np.ndarray) -> np.ndarray:
        """Warp entire image to bird's-eye view."""
        return cv2.warpPerspective(
            image, 
            self.M, 
            (self.bev_width, self.bev_height)
        )
    
    def visualize_transformation(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize the transformation region on the image.
        
        Returns:
            Image with transformation region drawn
        """
        vis_image = image.copy()
        
        # Draw source region
        pts = self.src_points.astype(np.int32)
        cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
        
        # Label corners
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, (pt, label) in enumerate(zip(pts, labels)):
            cv2.circle(vis_image, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(
                vis_image, label, (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        return vis_image


def calibrate_bev_interactive(video_path: str):
    """
    Interactive calibration tool for BEV transformation.
    
    Click 4 points on the video to define the road region:
    1. Top-left (far)
    2. Top-right (far)
    3. Bottom-right (near)
    4. Bottom-left (near)
    
    Press 's' to save, 'r' to reset, 'q' to quit.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read video")
        return None
    
    h, w = frame.shape[:2]
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")
    
    cv2.namedWindow("BEV Calibration")
    cv2.setMouseCallback("BEV Calibration", mouse_callback)
    
    print("Click 4 corners of the road region:")
    print("1. Top-left (far), 2. Top-right (far)")
    print("3. Bottom-right (near), 4. Bottom-left (near)")
    print("Press 's' to save, 'r' to reset, 'q' to quit")
    
    while True:
        display = frame.copy()
        
        # Draw existing points
        for i, pt in enumerate(points):
            cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(
                display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        # Draw lines if we have multiple points
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(display, [pts], len(points) == 4, (0, 255, 0), 2)
        
        cv2.imshow("BEV Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            points = []
            print("Reset points")
        elif key == ord('s') and len(points) == 4:
            src_points = np.float32(points)
            print("\nCalibration complete!")
            print(f"Source points:\n{src_points}")
            
            cv2.destroyAllWindows()
            cap.release()
            return src_points
    
    cv2.destroyAllWindows()
    cap.release()
    return None


# Example usage
if __name__ == "__main__":
    # Create transformer for 1920x1080 video
    transformer = BEVTransformer(
        image_width=1920,
        image_height=1080,
        bev_width=400,
        bev_height=800
    )
    
    # Example point transformation
    image_point = (960, 540)  # Center of image
    bev_point = transformer.image_to_bev(image_point)
    back_to_image = transformer.bev_to_image(bev_point)
    
    print(f"Image point: {image_point}")
    print(f"BEV point: {bev_point}")
    print(f"Back to image: {back_to_image}")
    
    # Example trajectory transformation
    trajectory = [(100, 500), (150, 520), (200, 540), (250, 560)]
    bev_trajectory = transformer.batch_image_to_bev(trajectory)
    
    print(f"\nOriginal trajectory: {trajectory}")
    print(f"BEV trajectory: {bev_trajectory}")
