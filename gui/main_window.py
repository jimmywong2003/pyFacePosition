"""
Main window for the pyFacePosition application.
"""

import sys
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QStatusBar, QSplitter, QScrollArea,
    QToolBar, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon, QPainter, QPen, QColor

from face_detector import FaceDetector, load_image, FaceDetectionResult
import cv2
import numpy as np


class ImageDisplayWidget(QWidget):
    """Widget for displaying images with detection results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: Optional[np.ndarray] = None
        self.detection_results: list[FaceDetectionResult] = []
        self.setMinimumSize(400, 300)
        
    def set_image(self, image: np.ndarray):
        """Set the image to display."""
        self.image = image.copy()
        self.update()
        
    def set_detection_results(self, results: list[FaceDetectionResult]):
        """Set detection results to display."""
        self.detection_results = results
        self.update()
        
    def paintEvent(self, event):
        """Paint the image and detection results."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        
        if self.image is not None:
            # Convert OpenCV image to QImage
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.image.data, width, height, bytes_per_line, 
                           QImage.Format.Format_BGR888)
            
            # Scale image to fit widget while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Center the image
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # Calculate scaling factor for drawing detections
            if self.image.shape[1] > 0 and scaled_pixmap.width() > 0:
                scale_x = scaled_pixmap.width() / self.image.shape[1]
                scale_y = scaled_pixmap.height() / self.image.shape[0]
                
                # Draw detection results
                for result in self.detection_results:
                    # Scale coordinates
                    x_face = x + result.face_bbox[0] * scale_x
                    y_face = y + result.face_bbox[1] * scale_y
                    w_face = result.face_bbox[2] * scale_x
                    h_face = result.face_bbox[3] * scale_y
                    
                    # Draw face bounding box (green)
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawRect(int(x_face), int(y_face), int(w_face), int(h_face))
                    
                    # Draw left eye (blue)
                    if result.left_eye:
                        eye_x = x + result.left_eye[0] * scale_x
                        eye_y = y + result.left_eye[1] * scale_y
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.setBrush(QColor(255, 0, 0, 100))
                        painter.drawEllipse(int(eye_x - 5), int(eye_y - 5), 10, 10)
                        painter.drawText(int(eye_x - 10), int(eye_y - 10), "L")
                    
                    # Draw right eye (red)
                    if result.right_eye:
                        eye_x = x + result.right_eye[0] * scale_x
                        eye_y = y + result.right_eye[1] * scale_y
                        painter.setPen(QPen(QColor(0, 0, 255), 2))
                        painter.setBrush(QColor(0, 0, 255, 100))
                        painter.drawEllipse(int(eye_x - 5), int(eye_y - 5), 10, 10)
                        painter.drawText(int(eye_x - 10), int(eye_y - 10), "R")
                    
                    # Draw mouth (yellow)
                    if result.mouth:
                        mouth_x = x + result.mouth[0] * scale_x
                        mouth_y = y + result.mouth[1] * scale_y
                        painter.setPen(QPen(QColor(255, 255, 0), 2))
                        painter.setBrush(QColor(255, 255, 0, 100))
                        painter.drawEllipse(int(mouth_x - 5), int(mouth_y - 5), 10, 10)
                        painter.drawText(int(mouth_x - 10), int(mouth_y - 10), "M")


class DetectionThread(QThread):
    """Thread for running face detection to keep UI responsive."""
    
    detection_finished = pyqtSignal(list)
    detection_error = pyqtSignal(str)
    
    def __init__(self, detector: FaceDetector, image: np.ndarray, method: str):
        super().__init__()
        self.detector = detector
        self.image = image
        self.method = method
        
    def run(self):
        """Run face detection in the thread."""
        try:
            results = self.detector.detect_faces(self.image, self.method)
            self.detection_finished.emit(results)
        except Exception as e:
            self.detection_error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.image: Optional[np.ndarray] = None
        self.detector = FaceDetector()
        self.detection_thread: Optional[DetectionThread] = None
        self.detection_results: list[FaceDetectionResult] = []
        
        self.init_ui()
        self.setWindowTitle("pyFacePosition - Face Detection Tool")
        self.resize(1200, 800)
        
    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Image display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image display widget
        self.image_display = ImageDisplayWidget()
        left_layout.addWidget(self.image_display)
        
        # Image info label
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.image_info_label)
        
        # Right panel - Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)
        
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        file_layout.addWidget(self.save_button)
        
        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)
        
        # Detection settings group
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout()
        
        # Algorithm selection
        detection_layout.addWidget(QLabel("Detection Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Haar Cascades", "DNN (Future)"])
        detection_layout.addWidget(self.algorithm_combo)
        
        # Detection parameters
        detection_layout.addWidget(QLabel("Scale Factor:"))
        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(1.01, 2.0)
        self.scale_factor_spin.setValue(1.1)
        self.scale_factor_spin.setSingleStep(0.05)
        detection_layout.addWidget(self.scale_factor_spin)
        
        detection_layout.addWidget(QLabel("Min Neighbors:"))
        self.min_neighbors_spin = QSpinBox()
        self.min_neighbors_spin.setRange(1, 20)
        self.min_neighbors_spin.setValue(5)
        detection_layout.addWidget(self.min_neighbors_spin)
        
        detection_layout.addWidget(QLabel("Min Size:"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 200)
        self.min_size_spin.setValue(30)
        detection_layout.addWidget(self.min_size_spin)
        
        detection_group.setLayout(detection_layout)
        right_layout.addWidget(detection_group)
        
        # Detection control group
        control_group = QGroupBox("Detection Control")
        control_layout = QVBoxLayout()
        
        self.detect_button = QPushButton("Detect Faces")
        self.detect_button.clicked.connect(self.detect_faces)
        self.detect_button.setEnabled(False)
        control_layout.addWidget(self.detect_button)
        
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self.clear_results)
        self.clear_button.setEnabled(False)
        control_layout.addWidget(self.clear_button)
        
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        
        # Results group
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        
        self.results_label = QLabel("No detection results")
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)
        
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)
        
        # Add stretch to push everything up
        right_layout.addStretch()
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def create_toolbar(self):
        """Create the application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # Load action
        load_action = QAction("Load", self)
        load_action.triggered.connect(self.load_image)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        # Detect action
        detect_action = QAction("Detect", self)
        detect_action.triggered.connect(self.detect_faces)
        toolbar.addAction(detect_action)
        
        # Clear action
        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.clear_results)
        toolbar.addAction(clear_action)
        
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            self.status_bar.showMessage(f"Loading image: {file_path}")
            QApplication.processEvents()
            
            image = load_image(file_path)
            if image is not None:
                self.image = image
                self.image_display.set_image(image)
                self.detect_button.setEnabled(True)
                self.clear_button.setEnabled(True)
                
                # Update image info
                height, width, channels = image.shape
                self.image_info_label.setText(
                    f"{Path(file_path).name} - {width}x{height} - {channels} channels"
                )
                self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
            else:
                QMessageBox.critical(self, "Error", f"Failed to load image: {file_path}")
                self.status_bar.showMessage("Failed to load image")
                
    def detect_faces(self):
        """Run face detection on the current image."""
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
            
        if self.detection_thread and self.detection_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Detection is already in progress")
            return
            
        # Get detection parameters
        method = "haar" if self.algorithm_combo.currentText() == "Haar Cascades" else "dnn"
        
        # Update UI
        self.detect_button.setEnabled(False)
        self.status_bar.showMessage("Detecting faces...")
        QApplication.processEvents()
        
        # Create and start detection thread
        self.detection_thread = DetectionThread(self.detector, self.image, method)
        self.detection_thread.detection_finished.connect(self.on_detection_finished)
        self.detection_thread.detection_error.connect(self.on_detection_error)
        self.detection_thread.finished.connect(self.on_detection_thread_finished)
        self.detection_thread.start()
        
    @pyqtSlot(list)
    def on_detection_finished(self, results: list[FaceDetectionResult]):
        """Handle detection completion."""
        self.detection_results = results
        self.image_display.set_detection_results(results)
        
        # Update results label
        if results:
            result_text = f"Detected {len(results)} face(s):\n"
            for i, result in enumerate(results):
                result_text += f"\nFace {i+1}:\n"
                result_text += f"  Bounding Box: {result.face_bbox}\n"
                if result.left_eye:
                    result_text += f"  Left Eye: {result.left_eye}\n"
                if result.right_eye:
                    result_text += f"  Right Eye: {result.right_eye}\n"
                if result.mouth:
                    result_text += f"  Mouth: {result.mouth}\n"
                result_text += f"  Confidence: {result.confidence:.2f}\n"
                result_text += f"  Algorithm: {result.algorithm}\n"
        else:
            result_text = "No faces detected"
            
        self.results_label.setText(result_text)
        self.save_button.setEnabled(True)
        self.status_bar.showMessage(f"Detection complete: {len(results)} face(s) found")
        
    @pyqtSlot(str)
    def on_detection_error(self, error_message: str):
        """Handle detection error."""
        QMessageBox.critical(self, "Detection Error", f"Face detection failed:\n{error_message}")
        self.status_bar.showMessage("Detection failed")
        
    @pyqtSlot()
    def on_detection_thread_finished(self):
        """Handle detection thread completion."""
        self.detect_button.setEnabled(True)
        
    def clear_results(self):
        """Clear detection results."""
        self.detection_results = []
        self.image_display.set_detection_results([])
        self.results_label.setText("No detection results")
        self.save_button.setEnabled(False)
        self.status_bar.showMessage("Results cleared")
        
    def save_results(self):
        """Save detection results and annotated image."""
        if self.image is None or not self.detection_results:
            QMessageBox.warning(self, "Warning", "No detection results to save")
            return
            
        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            # Draw detections on image
            annotated_image = self.detector.draw_detections(self.image, self.detection_results)
            
            # Save image
            if cv2.imwrite(file_path, annotated_image):
                self.status_bar.showMessage(f"Saved results to {Path(file_path).name}")
                
                # Also save detection data as text file
                txt_file_path = Path(file_path).with_suffix('.txt')
                with open(txt_file_path, 'w') as f:
                    f.write(f"Face Detection Results\n")
                    f.write(f"=====================\n\n")
                    f.write(f"Image: {file_path}\n")
                    f.write(f"Faces Detected: {len(self.detection_results)}\n\n")
                    
                    for i, result in enumerate(self.detection_results):
                        f.write(f"Face {i+1}:\n")
                        f.write(f"  Bounding Box: {result.face_bbox}\n")
                        if result.left_eye:
                            f.write(f"  Left Eye: {result.left_eye}\n")
                        if result.right_eye:
                            f.write(f"  Right Eye: {result.right_eye}\n")
                        if result.mouth:
                            f.write(f"  Mouth: {result.mouth}\n")
                        f.write(f"  Confidence: {result.confidence:.2f}\n")
                        f.write(f"  Algorithm: {result.algorithm}\n\n")
                        
                QMessageBox.information(
                    self,
                    "Save Complete",
                    f"Saved annotated image to:\n{file_path}\n\n"
                    f"Detection data saved to:\n{txt_file_path}"
                )
