"""
Face detection module using OpenCV.
Provides functionality to detect face bounding boxes, eyes, and mouth positions.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class FaceDetectionResult:
    """Container for face detection results."""
    face_bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    left_eye: Optional[Tuple[int, int]] = None  # (x, y)
    right_eye: Optional[Tuple[int, int]] = None  # (x, y)
    mouth: Optional[Tuple[int, int]] = None  # (x, y)
    confidence: float = 0.0
    algorithm: str = "OpenCV"
    
    # Manual/OFIQ specific fields
    left_eye_center: Optional[Tuple[int, int]] = None  # Manual/OFIQ left eye center (x, y)
    right_eye_center: Optional[Tuple[int, int]] = None  # Manual/OFIQ right eye center (x, y)
    mouth_center: Optional[Tuple[int, int]] = None  # Manual/OFIQ mouth center (x, y)
    data_source: str = "detected"  # "detected", "manual", "ofiq"
    manual_confidence: float = 1.0  # Confidence for manual/OFIQ entries (0.0-1.0)
    
    # Additional facial landmarks (for future 98-point support)
    landmarks: Optional[List[Tuple[int, int]]] = None  # List of (x, y) coordinates
    
    def has_manual_data(self) -> bool:
        """Check if this result contains manual or OFIQ data."""
        return (self.left_eye_center is not None or 
                self.right_eye_center is not None or 
                self.mouth_center is not None or
                self.data_source in ["manual", "ofiq"])
    
    def get_effective_eyes(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Get eye coordinates, preferring manual/OFIQ data if available."""
        left = self.left_eye_center if self.left_eye_center is not None else self.left_eye
        right = self.right_eye_center if self.right_eye_center is not None else self.right_eye
        return left, right
    
    def get_effective_mouth(self) -> Optional[Tuple[int, int]]:
        """Get mouth coordinate, preferring manual/OFIQ data if available."""
        return self.mouth_center if self.mouth_center is not None else self.mouth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = {
            "face_bbox": self.face_bbox,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "mouth": self.mouth,
            "confidence": self.confidence,
            "algorithm": self.algorithm,
            "left_eye_center": self.left_eye_center,
            "right_eye_center": self.right_eye_center,
            "mouth_center": self.mouth_center,
            "data_source": self.data_source,
            "manual_confidence": self.manual_confidence,
            "landmarks": self.landmarks
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceDetectionResult":
        """Create result from dictionary."""
        return cls(
            face_bbox=data.get("face_bbox", (0, 0, 0, 0)),
            left_eye=data.get("left_eye"),
            right_eye=data.get("right_eye"),
            mouth=data.get("mouth"),
            confidence=data.get("confidence", 0.0),
            algorithm=data.get("algorithm", "OpenCV"),
            left_eye_center=data.get("left_eye_center"),
            right_eye_center=data.get("right_eye_center"),
            mouth_center=data.get("mouth_center"),
            data_source=data.get("data_source", "detected"),
            manual_confidence=data.get("manual_confidence", 1.0),
            landmarks=data.get("landmarks")
        )


class FaceDetector:
    """Face detector using OpenCV's Haar cascades and DNN models."""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.mouth_cascade = None
        self.face_dnn = None
        self.initialize_cascades()
        
    def initialize_cascades(self):
        """Initialize OpenCV cascade classifiers."""
        try:
            # Try to load Haar cascades
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            # Try to load mouth/smile cascade (may not be available in all OpenCV installations)
            try:
                # Try smile cascade first (more commonly available)
                self.mouth_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_smile.xml'
                )
                if self.mouth_cascade.empty():
                    # Try alternative mouth cascade
                    self.mouth_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
                    )
                    if self.mouth_cascade.empty():
                        print("Mouth cascade not available, mouth detection will be disabled")
                        self.mouth_cascade = None
            except Exception as e:
                print(f"Mouth cascade not available ({e}), mouth detection will be disabled")
                self.mouth_cascade = None
            
            # Try to load DNN model for better accuracy
            try:
                # This would require downloading model files
                # For now, we'll use cascades as fallback
                pass
            except:
                print("DNN model not available, using Haar cascades")
                
        except Exception as e:
            print(f"Error loading cascade classifiers: {e}")
            raise
    
    def detect_faces_haar(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces using Haar cascades.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of FaceDetectionResult objects
        """
        if self.face_cascade is None:
            raise RuntimeError("Face cascade not initialized")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            result = FaceDetectionResult(
                face_bbox=(x, y, w, h),
                confidence=1.0,  # Haar cascades don't provide confidence
                algorithm="OpenCV-Haar"
            )
            
            # Region of interest for eyes and mouth
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            # Detect eyes
            if self.eye_cascade:
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                
                if len(eyes) >= 2:
                    # Sort eyes by x-coordinate (left to right)
                    eyes = sorted(eyes, key=lambda e: e[0])
                    left_eye = eyes[0]
                    right_eye = eyes[1] if len(eyes) > 1 else eyes[0]
                    
                    # Calculate eye centers
                    result.left_eye = (
                        x + left_eye[0] + left_eye[2] // 2,
                        y + left_eye[1] + left_eye[3] // 2
                    )
                    result.right_eye = (
                        x + right_eye[0] + right_eye[2] // 2,
                        y + right_eye[1] + right_eye[3] // 2
                    )
            
            # Detect mouth
            if self.mouth_cascade:
                # Mouth is typically in lower half of face
                mouth_roi_gray = gray[y + h//2:y + h, x:x + w]
                mouths = self.mouth_cascade.detectMultiScale(
                    mouth_roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 20)
                )
                
                if len(mouths) > 0:
                    # Take the largest mouth detection
                    mouth = max(mouths, key=lambda m: m[2] * m[3])
                    result.mouth = (
                        x + mouth[0] + mouth[2] // 2,
                        y + h//2 + mouth[1] + mouth[3] // 2
                    )
            
            results.append(result)
        
        return results
    
    def detect_faces_dnn(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces using DNN model (placeholder for future implementation).
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of FaceDetectionResult objects
        """
        # This is a placeholder for DNN-based detection
        # In a real implementation, you would load a pre-trained DNN model
        # For now, fall back to Haar cascades
        return self.detect_faces_haar(image)
    
    def detect_faces(self, image: np.ndarray, method: str = "haar") -> List[FaceDetectionResult]:
        """
        Main face detection method.
        
        Args:
            image: Input image in BGR format
            method: Detection method ("haar" or "dnn")
            
        Returns:
            List of FaceDetectionResult objects
        """
        if method.lower() == "dnn":
            return self.detect_faces_dnn(image)
        else:
            return self.detect_faces_haar(image)
    
    def draw_detections(self, image: np.ndarray, results: List[FaceDetectionResult]) -> np.ndarray:
        """
        Draw detection results on the image.
        
        Args:
            image: Input image in BGR format
            results: List of detection results
            
        Returns:
            Image with detections drawn
        """
        output = image.copy()
        
        for result in results:
            # Draw face bounding box (green)
            x, y, w, h = result.face_bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get effective coordinates (prefer manual/OFIQ if available)
            left_eye, right_eye = result.get_effective_eyes()
            mouth = result.get_effective_mouth()
            
            # Draw left eye - different style for manual/OFIQ
            if left_eye:
                if result.left_eye_center is not None:  # Manual/OFIQ data
                    # Draw as a square with thicker border for manual data
                    cv2.rectangle(output, 
                                 (left_eye[0] - 6, left_eye[1] - 6),
                                 (left_eye[0] + 6, left_eye[1] + 6),
                                 (255, 165, 0), 3)  # Orange for manual
                    cv2.putText(output, "L*", (left_eye[0] - 15, left_eye[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                else:  # Detected data
                    cv2.circle(output, left_eye, 5, (255, 0, 0), -1)
                    cv2.putText(output, "L", (left_eye[0] - 10, left_eye[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw right eye - different style for manual/OFIQ
            if right_eye:
                if result.right_eye_center is not None:  # Manual/OFIQ data
                    # Draw as a square with thicker border for manual data
                    cv2.rectangle(output, 
                                 (right_eye[0] - 6, right_eye[1] - 6),
                                 (right_eye[0] + 6, right_eye[1] + 6),
                                 (255, 165, 0), 3)  # Orange for manual
                    cv2.putText(output, "R*", (right_eye[0] - 15, right_eye[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                else:  # Detected data
                    cv2.circle(output, right_eye, 5, (0, 0, 255), -1)
                    cv2.putText(output, "R", (right_eye[0] - 10, right_eye[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw mouth - different style for manual/OFIQ
            if mouth:
                if result.mouth_center is not None:  # Manual/OFIQ data
                    # Draw as a diamond for manual data
                    pts = np.array([
                        [mouth[0], mouth[1] - 7],
                        [mouth[0] + 7, mouth[1]],
                        [mouth[0], mouth[1] + 7],
                        [mouth[0] - 7, mouth[1]]
                    ], np.int32)
                    cv2.polylines(output, [pts], True, (255, 20, 147), 3)  # Deep pink for manual
                    cv2.putText(output, "M*", (mouth[0] - 15, mouth[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 20, 147), 2)
                else:  # Detected data
                    cv2.circle(output, mouth, 5, (0, 255, 255), -1)
                    cv2.putText(output, "M", (mouth[0] - 10, mouth[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Add info text with data source
            if result.has_manual_data():
                info_text = f"{result.algorithm} + Manual: {result.confidence:.2f}"
                color = (255, 165, 0)  # Orange for mixed data
            else:
                info_text = f"{result.algorithm}: {result.confidence:.2f}"
                color = (0, 255, 0)  # Green for detected only
            
            cv2.putText(output, info_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks if available (for future 98-point support)
            if result.landmarks:
                for landmark in result.landmarks:
                    cv2.circle(output, landmark, 2, (255, 255, 255), -1)
        
        return output


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Loaded image or None if failed
    """
    try:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            return None
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None


def save_image(image: np.ndarray, file_path: str) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image to save
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(file_path, image)
        return True
    except Exception as e:
        print(f"Error saving image to {file_path}: {e}")
        return False