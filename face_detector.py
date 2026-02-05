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
            
            # Draw left eye (blue)
            if result.left_eye:
                cv2.circle(output, result.left_eye, 5, (255, 0, 0), -1)
                cv2.putText(output, "L", (result.left_eye[0] - 10, result.left_eye[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw right eye (red)
            if result.right_eye:
                cv2.circle(output, result.right_eye, 5, (0, 0, 255), -1)
                cv2.putText(output, "R", (result.right_eye[0] - 10, result.right_eye[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw mouth (yellow)
            if result.mouth:
                cv2.circle(output, result.mouth, 5, (0, 255, 255), -1)
                cv2.putText(output, "M", (result.mouth[0] - 10, result.mouth[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Add confidence and algorithm info
            info_text = f"{result.algorithm}: {result.confidence:.2f}"
            cv2.putText(output, info_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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