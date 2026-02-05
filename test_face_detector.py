"""
Test script for face detector module.
"""

import cv2
import numpy as np
from face_detector import FaceDetector, load_image, save_image


def test_face_detector():
    """Test the face detector with a sample image."""
    print("Testing Face Detector...")
    
    # Create a simple test image with a face-like pattern
    # In a real test, you would use an actual image file
    print("Creating test image...")
    
    # Create a simple colored image
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Draw a simple face-like pattern
    # Face oval
    cv2.ellipse(test_image, (200, 150), (80, 100), 0, 0, 360, (200, 200, 200), -1)
    
    # Eyes
    cv2.circle(test_image, (170, 120), 15, (255, 255, 255), -1)
    cv2.circle(test_image, (230, 120), 15, (255, 255, 255), -1)
    cv2.circle(test_image, (170, 120), 5, (0, 0, 0), -1)
    cv2.circle(test_image, (230, 120), 5, (0, 0, 0), -1)
    
    # Mouth
    cv2.ellipse(test_image, (200, 190), (40, 20), 0, 0, 180, (0, 0, 0), 3)
    
    # Save test image
    save_image(test_image, "test_face.jpg")
    print("Saved test image: test_face.jpg")
    
    # Load the test image
    image = load_image("test_face.jpg")
    if image is None:
        print("Failed to load test image")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # Create face detector
    detector = FaceDetector()
    
    # Test Haar cascade detection
    print("\nTesting Haar cascade detection...")
    results = detector.detect_faces(image, method="haar")
    
    print(f"Detected {len(results)} face(s)")
    
    for i, result in enumerate(results):
        print(f"\nFace {i+1}:")
        print(f"  Bounding Box: {result.face_bbox}")
        print(f"  Left Eye: {result.left_eye}")
        print(f"  Right Eye: {result.right_eye}")
        print(f"  Mouth: {result.mouth}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Algorithm: {result.algorithm}")
    
    # Draw detections on image
    annotated_image = detector.draw_detections(image, results)
    save_image(annotated_image, "test_face_detected.jpg")
    print("\nSaved annotated image: test_face_detected.jpg")
    
    # Test DNN detection (placeholder)
    print("\nTesting DNN detection (placeholder)...")
    try:
        results_dnn = detector.detect_faces(image, method="dnn")
        print(f"DNN detected {len(results_dnn)} face(s)")
    except Exception as e:
        print(f"DNN detection not implemented: {e}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_face_detector()