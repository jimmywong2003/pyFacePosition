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
    
    # Signals
    face_selected = pyqtSignal(int)  # Emits face index when clicked
    point_edited = pyqtSignal(int, str, tuple)  # Emits (face_index, point_type, coordinates)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: Optional[np.ndarray] = None
        self.detection_results: list[FaceDetectionResult] = []
        self.setMinimumSize(400, 300)
        
        # Interactive editing state
        self.editing_mode = "none"  # "none", "face", "left_eye", "right_eye", "mouth"
        self.selected_face_index = -1
        self.selected_point_type = ""  # "face", "left_eye", "right_eye", "mouth"
        self.is_dragging = False
        self.drag_start_pos = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
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
                    
                    # Get effective coordinates (prefer manual/OFIQ if available)
                    left_eye, right_eye = result.get_effective_eyes()
                    mouth = result.get_effective_mouth()
                    
                    # Draw left eye - different style for manual/OFIQ
                    if left_eye:
                        eye_x = x + left_eye[0] * scale_x
                        eye_y = y + left_eye[1] * scale_y
                        if result.left_eye_center is not None:  # Manual/OFIQ data
                            # Draw as a square with thicker border for manual data
                            painter.setPen(QPen(QColor(255, 165, 0), 3))  # Orange for manual
                            painter.setBrush(QColor(255, 165, 0, 100))
                            painter.drawRect(int(eye_x - 6), int(eye_y - 6), 12, 12)
                            painter.drawText(int(eye_x - 15), int(eye_y - 15), "L*")
                        else:  # Detected data
                            painter.setPen(QPen(QColor(255, 0, 0), 2))
                            painter.setBrush(QColor(255, 0, 0, 100))
                            painter.drawEllipse(int(eye_x - 5), int(eye_y - 5), 10, 10)
                            painter.drawText(int(eye_x - 10), int(eye_y - 10), "L")
                    
                    # Draw right eye - different style for manual/OFIQ
                    if right_eye:
                        eye_x = x + right_eye[0] * scale_x
                        eye_y = y + right_eye[1] * scale_y
                        if result.right_eye_center is not None:  # Manual/OFIQ data
                            # Draw as a square with thicker border for manual data
                            painter.setPen(QPen(QColor(255, 165, 0), 3))  # Orange for manual
                            painter.setBrush(QColor(255, 165, 0, 100))
                            painter.drawRect(int(eye_x - 6), int(eye_y - 6), 12, 12)
                            painter.drawText(int(eye_x - 15), int(eye_y - 15), "R*")
                        else:  # Detected data
                            painter.setPen(QPen(QColor(0, 0, 255), 2))
                            painter.setBrush(QColor(0, 0, 255, 100))
                            painter.drawEllipse(int(eye_x - 5), int(eye_y - 5), 10, 10)
                            painter.drawText(int(eye_x - 10), int(eye_y - 10), "R")
                    
                    # Draw mouth - different style for manual/OFIQ
                    if mouth:
                        mouth_x = x + mouth[0] * scale_x
                        mouth_y = y + mouth[1] * scale_y
                        if result.mouth_center is not None:  # Manual/OFIQ data
                            # Draw as a diamond for manual data
                            painter.setPen(QPen(QColor(255, 20, 147), 3))  # Deep pink for manual
                            painter.setBrush(QColor(255, 20, 147, 100))
                            # Draw diamond shape
                            from PyQt6.QtCore import QPoint
                            points = [
                                QPoint(int(mouth_x), int(mouth_y - 7)),
                                QPoint(int(mouth_x + 7), int(mouth_y)),
                                QPoint(int(mouth_x), int(mouth_y + 7)),
                                QPoint(int(mouth_x - 7), int(mouth_y))
                            ]
                            from PyQt6.QtGui import QPolygon
                            polygon = QPolygon(points)
                            painter.drawPolygon(polygon)
                            painter.drawText(int(mouth_x - 15), int(mouth_y - 15), "M*")
                        else:  # Detected data
                            painter.setPen(QPen(QColor(255, 255, 0), 2))
                            painter.setBrush(QColor(255, 255, 0, 100))
                            painter.drawEllipse(int(mouth_x - 5), int(mouth_y - 5), 10, 10)
                            painter.drawText(int(mouth_x - 10), int(mouth_y - 10), "M")
                    
                    # Draw landmarks if available (for future 98-point support)
                    if result.landmarks:
                        for landmark in result.landmarks:
                            landmark_x = x + landmark[0] * scale_x
                            landmark_y = y + landmark[1] * scale_y
                            painter.setPen(QPen(QColor(255, 255, 255), 1))
                            painter.setBrush(QColor(255, 255, 255, 150))
                            painter.drawEllipse(int(landmark_x - 2), int(landmark_y - 2), 4, 4)
        
        # Draw visual feedback for interactive editing
        if self.is_dragging and self.selected_face_index >= 0:
            painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            if self.selected_point_type == "face":
                # Draw dragging rectangle for face bounding box
                x1, y1 = self.drag_start_pos
                x2, y2 = event.pos().x(), event.pos().y()
                painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            else:
                # Draw dragging line for point editing
                x1, y1 = self.drag_start_pos
                x2, y2 = event.pos().x(), event.pos().y()
                painter.drawLine(x1, y1, x2, y2)
                painter.drawEllipse(x2 - 8, y2 - 8, 16, 16)
    
    def mousePressEvent(self, event):
        """Handle mouse press events for interactive editing."""
        if event.button() == Qt.MouseButton.LeftButton and self.image is not None:
            # Calculate image scaling and offset
            height, width, _ = self.image.shape
            scaled_pixmap = QPixmap.fromImage(
                QImage(self.image.data, width, height, 3 * width, QImage.Format.Format_BGR888)
            ).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
            
            x_offset = (self.width() - scaled_pixmap.width()) // 2
            y_offset = (self.height() - scaled_pixmap.height()) // 2
            scale_x = scaled_pixmap.width() / width
            scale_y = scaled_pixmap.height() / height
            
            # Convert mouse position to image coordinates
            mouse_x = event.pos().x()
            mouse_y = event.pos().y()
            img_x = int((mouse_x - x_offset) / scale_x) if scale_x > 0 else 0
            img_y = int((mouse_y - y_offset) / scale_y) if scale_y > 0 else 0
            
            # Check if clicking on a face or facial feature
            for i, result in enumerate(self.detection_results):
                # Check face bounding box
                x, y, w, h = result.face_bbox
                if x <= img_x <= x + w and y <= img_y <= y + h:
                    self.selected_face_index = i
                    self.face_selected.emit(i)
                    
                    # Check which feature was clicked
                    left_eye, right_eye = result.get_effective_eyes()
                    mouth = result.get_effective_mouth()
                    
                    # Define click tolerance (in image coordinates)
                    tolerance = 15 / min(scale_x, scale_y) if min(scale_x, scale_y) > 0 else 15
                    
                    if left_eye and abs(img_x - left_eye[0]) <= tolerance and abs(img_y - left_eye[1]) <= tolerance:
                        self.selected_point_type = "left_eye"
                        self.editing_mode = "left_eye"
                    elif right_eye and abs(img_x - right_eye[0]) <= tolerance and abs(img_y - right_eye[1]) <= tolerance:
                        self.selected_point_type = "right_eye"
                        self.editing_mode = "right_eye"
                    elif mouth and abs(img_x - mouth[0]) <= tolerance and abs(img_y - mouth[1]) <= tolerance:
                        self.selected_point_type = "mouth"
                        self.editing_mode = "mouth"
                    else:
                        # Clicked inside face but not on a specific feature
                        self.selected_point_type = "face"
                        self.editing_mode = "face"
                    
                    # Start dragging
                    self.is_dragging = True
                    self.drag_start_pos = (mouse_x, mouse_y)
                    self.update()
                    return
            
            # If no face was clicked, clear selection
            self.selected_face_index = -1
            self.selected_point_type = ""
            self.editing_mode = "none"
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for interactive editing."""
        if self.is_dragging:
            self.update()  # Trigger repaint for visual feedback
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for interactive editing."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            
            if self.selected_face_index >= 0 and self.drag_start_pos:
                # Calculate image scaling and offset
                height, width, _ = self.image.shape
                scaled_pixmap = QPixmap.fromImage(
                    QImage(self.image.data, width, height, 3 * width, QImage.Format.Format_BGR888)
                ).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                
                x_offset = (self.width() - scaled_pixmap.width()) // 2
                y_offset = (self.height() - scaled_pixmap.height()) // 2
                scale_x = scaled_pixmap.width() / width
                scale_y = scaled_pixmap.height() / height
                
                # Convert mouse position to image coordinates
                mouse_x = event.pos().x()
                mouse_y = event.pos().y()
                img_x = int((mouse_x - x_offset) / scale_x) if scale_x > 0 else 0
                img_y = int((mouse_y - y_offset) / scale_y) if scale_y > 0 else 0
                
                # Update the selected feature
                result = self.detection_results[self.selected_face_index]
                
                if self.selected_point_type == "face":
                    # Update face bounding box
                    start_x = int((self.drag_start_pos[0] - x_offset) / scale_x) if scale_x > 0 else 0
                    start_y = int((self.drag_start_pos[1] - y_offset) / scale_y) if scale_y > 0 else 0
                    
                    x = min(start_x, img_x)
                    y = min(start_y, img_y)
                    w = abs(img_x - start_x)
                    h = abs(img_y - start_y)
                    
                    result.face_bbox = (x, y, w, h)
                    self.point_edited.emit(self.selected_face_index, "face", (x, y, w, h))
                    
                elif self.selected_point_type == "left_eye":
                    result.left_eye_center = (img_x, img_y)
                    self.point_edited.emit(self.selected_face_index, "left_eye", (img_x, img_y))
                    
                elif self.selected_point_type == "right_eye":
                    result.right_eye_center = (img_x, img_y)
                    self.point_edited.emit(self.selected_face_index, "right_eye", (img_x, img_y))
                    
                elif self.selected_point_type == "mouth":
                    result.mouth_center = (img_x, img_y)
                    self.point_edited.emit(self.selected_face_index, "mouth", (img_x, img_y))
                
                # Update data source to manual
                result.data_source = "manual"
                result.manual_confidence = 1.0
                
                # Update display
                self.update()
                
                # Emit signal to update GUI
                self.point_edited.emit(self.selected_face_index, self.selected_point_type, (img_x, img_y))
            
            self.drag_start_pos = None
            self.update()
    
    def set_editing_mode(self, mode: str):
        """Set the current editing mode."""
        self.editing_mode = mode
        self.update()
    
    def get_scaled_coordinates(self, img_x: int, img_y: int) -> tuple:
        """Convert image coordinates to widget coordinates."""
        if self.image is None:
            return (img_x, img_y)
        
        height, width, _ = self.image.shape
        scaled_pixmap = QPixmap.fromImage(
            QImage(self.image.data, width, height, 3 * width, QImage.Format.Format_BGR888)
        ).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
        
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        y_offset = (self.height() - scaled_pixmap.height()) // 2
        scale_x = scaled_pixmap.width() / width
        scale_y = scaled_pixmap.height() / height
        
        widget_x = x_offset + img_x * scale_x
        widget_y = y_offset + img_y * scale_y
        
        return (int(widget_x), int(widget_y))


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
        self.image_display.face_selected.connect(self.on_face_selected)
        self.image_display.point_edited.connect(self.on_point_edited)
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
        
        # Manual Editing group
        manual_group = QGroupBox("Manual Editing")
        manual_layout = QVBoxLayout()
        
        # Face selection
        manual_layout.addWidget(QLabel("Select Face:"))
        self.face_selection_combo = QComboBox()
        self.face_selection_combo.currentIndexChanged.connect(self.on_face_selection_changed)
        manual_layout.addWidget(self.face_selection_combo)
        
        # Bounding box coordinates
        manual_layout.addWidget(QLabel("Face Bounding Box:"))
        bbox_layout = QHBoxLayout()
        bbox_layout.addWidget(QLabel("X:"))
        self.bbox_x_spin = QSpinBox()
        self.bbox_x_spin.setRange(0, 10000)
        self.bbox_x_spin.valueChanged.connect(self.on_manual_data_changed)
        bbox_layout.addWidget(self.bbox_x_spin)
        
        bbox_layout.addWidget(QLabel("Y:"))
        self.bbox_y_spin = QSpinBox()
        self.bbox_y_spin.setRange(0, 10000)
        self.bbox_y_spin.valueChanged.connect(self.on_manual_data_changed)
        bbox_layout.addWidget(self.bbox_y_spin)
        
        bbox_layout.addWidget(QLabel("W:"))
        self.bbox_w_spin = QSpinBox()
        self.bbox_w_spin.setRange(1, 10000)
        self.bbox_w_spin.valueChanged.connect(self.on_manual_data_changed)
        bbox_layout.addWidget(self.bbox_w_spin)
        
        bbox_layout.addWidget(QLabel("H:"))
        self.bbox_h_spin = QSpinBox()
        self.bbox_h_spin.setRange(1, 10000)
        self.bbox_h_spin.valueChanged.connect(self.on_manual_data_changed)
        bbox_layout.addWidget(self.bbox_h_spin)
        manual_layout.addLayout(bbox_layout)
        
        # Left eye coordinates
        manual_layout.addWidget(QLabel("Left Eye Center:"))
        left_eye_layout = QHBoxLayout()
        left_eye_layout.addWidget(QLabel("X:"))
        self.left_eye_x_spin = QSpinBox()
        self.left_eye_x_spin.setRange(0, 10000)
        self.left_eye_x_spin.valueChanged.connect(self.on_manual_data_changed)
        left_eye_layout.addWidget(self.left_eye_x_spin)
        
        left_eye_layout.addWidget(QLabel("Y:"))
        self.left_eye_y_spin = QSpinBox()
        self.left_eye_y_spin.setRange(0, 10000)
        self.left_eye_y_spin.valueChanged.connect(self.on_manual_data_changed)
        left_eye_layout.addWidget(self.left_eye_y_spin)
        manual_layout.addLayout(left_eye_layout)
        
        # Right eye coordinates
        manual_layout.addWidget(QLabel("Right Eye Center:"))
        right_eye_layout = QHBoxLayout()
        right_eye_layout.addWidget(QLabel("X:"))
        self.right_eye_x_spin = QSpinBox()
        self.right_eye_x_spin.setRange(0, 10000)
        self.right_eye_x_spin.valueChanged.connect(self.on_manual_data_changed)
        right_eye_layout.addWidget(self.right_eye_x_spin)
        
        right_eye_layout.addWidget(QLabel("Y:"))
        self.right_eye_y_spin = QSpinBox()
        self.right_eye_y_spin.setRange(0, 10000)
        self.right_eye_y_spin.valueChanged.connect(self.on_manual_data_changed)
        right_eye_layout.addWidget(self.right_eye_y_spin)
        manual_layout.addLayout(right_eye_layout)
        
        # Mouth coordinates
        manual_layout.addWidget(QLabel("Mouth Center:"))
        mouth_layout = QHBoxLayout()
        mouth_layout.addWidget(QLabel("X:"))
        self.mouth_x_spin = QSpinBox()
        self.mouth_x_spin.setRange(0, 10000)
        self.mouth_x_spin.valueChanged.connect(self.on_manual_data_changed)
        mouth_layout.addWidget(self.mouth_x_spin)
        
        mouth_layout.addWidget(QLabel("Y:"))
        self.mouth_y_spin = QSpinBox()
        self.mouth_y_spin.setRange(0, 10000)
        self.mouth_y_spin.valueChanged.connect(self.on_manual_data_changed)
        mouth_layout.addWidget(self.mouth_y_spin)
        manual_layout.addLayout(mouth_layout)
        
        # Data source selection
        manual_layout.addWidget(QLabel("Data Source:"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Detected", "Manual", "OFIQ"])
        self.data_source_combo.currentTextChanged.connect(self.on_manual_data_changed)
        manual_layout.addWidget(self.data_source_combo)
        
        # Manual confidence
        manual_layout.addWidget(QLabel("Manual Confidence:"))
        self.manual_confidence_spin = QDoubleSpinBox()
        self.manual_confidence_spin.setRange(0.0, 1.0)
        self.manual_confidence_spin.setSingleStep(0.1)
        self.manual_confidence_spin.setValue(1.0)
        self.manual_confidence_spin.valueChanged.connect(self.on_manual_data_changed)
        manual_layout.addWidget(self.manual_confidence_spin)
        
        # Update button
        self.update_manual_button = QPushButton("Update Manual Data")
        self.update_manual_button.clicked.connect(self.update_manual_data)
        self.update_manual_button.setEnabled(False)
        manual_layout.addWidget(self.update_manual_button)
        
        manual_group.setLayout(manual_layout)
        right_layout.addWidget(manual_group)
        
        # Visualization Options group
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout()
        
        # Show centers only toggle
        self.show_centers_only_check = QCheckBox("Show Centers Only")
        self.show_centers_only_check.setChecked(False)
        self.show_centers_only_check.stateChanged.connect(self.on_visualization_changed)
        viz_layout.addWidget(self.show_centers_only_check)
        
        # Show detected points toggle
        self.show_detected_check = QCheckBox("Show Detected Points")
        self.show_detected_check.setChecked(True)
        self.show_detected_check.stateChanged.connect(self.on_visualization_changed)
        viz_layout.addWidget(self.show_detected_check)
        
        # Show manual points toggle
        self.show_manual_check = QCheckBox("Show Manual Points")
        self.show_manual_check.setChecked(True)
        self.show_manual_check.stateChanged.connect(self.on_visualization_changed)
        viz_layout.addWidget(self.show_manual_check)
        
        # Comparison mode toggle
        self.comparison_mode_check = QCheckBox("Comparison Mode")
        self.comparison_mode_check.setChecked(False)
        self.comparison_mode_check.stateChanged.connect(self.on_comparison_mode_changed)
        viz_layout.addWidget(self.comparison_mode_check)
        
        viz_group.setLayout(viz_layout)
        right_layout.addWidget(viz_group)
        
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
                
                # Update spin box ranges based on image dimensions
                max_width = width - 1
                max_height = height - 1
                
                # Bounding box spin boxes
                self.bbox_x_spin.setMaximum(max_width)
                self.bbox_y_spin.setMaximum(max_height)
                self.bbox_w_spin.setMaximum(width)
                self.bbox_h_spin.setMaximum(height)
                
                # Eye and mouth spin boxes
                self.left_eye_x_spin.setMaximum(max_width)
                self.left_eye_y_spin.setMaximum(max_height)
                self.right_eye_x_spin.setMaximum(max_width)
                self.right_eye_y_spin.setMaximum(max_height)
                self.mouth_x_spin.setMaximum(max_width)
                self.mouth_y_spin.setMaximum(max_height)
                
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
        
        # Update face selection combo and results label
        self.update_face_selection_combo()
        self.update_results_label()
        
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
        self.update_face_selection_combo()  # Clear the face selection combo
        self.status_bar.showMessage("Results cleared")
        
    def update_face_selection_combo(self):
        """Update the face selection combo box with current detection results."""
        self.face_selection_combo.clear()
        if self.detection_results:
            for i in range(len(self.detection_results)):
                self.face_selection_combo.addItem(f"Face {i+1}")
            self.face_selection_combo.setEnabled(True)
            self.update_manual_button.setEnabled(True)
            self.on_face_selection_changed(0)  # Select first face
        else:
            self.face_selection_combo.setEnabled(False)
            self.update_manual_button.setEnabled(False)
            
    @pyqtSlot(int)
    def on_face_selection_changed(self, index: int):
        """Handle face selection change."""
        if 0 <= index < len(self.detection_results):
            result = self.detection_results[index]
            
            # Update bounding box fields
            x, y, w, h = result.face_bbox
            self.bbox_x_spin.setValue(x)
            self.bbox_y_spin.setValue(y)
            self.bbox_w_spin.setValue(w)
            self.bbox_h_spin.setValue(h)
            
            # Update left eye fields
            if result.left_eye_center:
                self.left_eye_x_spin.setValue(result.left_eye_center[0])
                self.left_eye_y_spin.setValue(result.left_eye_center[1])
            elif result.left_eye:
                self.left_eye_x_spin.setValue(result.left_eye[0])
                self.left_eye_y_spin.setValue(result.left_eye[1])
            else:
                self.left_eye_x_spin.setValue(0)
                self.left_eye_y_spin.setValue(0)
                
            # Update right eye fields
            if result.right_eye_center:
                self.right_eye_x_spin.setValue(result.right_eye_center[0])
                self.right_eye_y_spin.setValue(result.right_eye_center[1])
            elif result.right_eye:
                self.right_eye_x_spin.setValue(result.right_eye[0])
                self.right_eye_y_spin.setValue(result.right_eye[1])
            else:
                self.right_eye_x_spin.setValue(0)
                self.right_eye_y_spin.setValue(0)
                
            # Update mouth fields
            if result.mouth_center:
                self.mouth_x_spin.setValue(result.mouth_center[0])
                self.mouth_y_spin.setValue(result.mouth_center[1])
            elif result.mouth:
                self.mouth_x_spin.setValue(result.mouth[0])
                self.mouth_y_spin.setValue(result.mouth[1])
            else:
                self.mouth_x_spin.setValue(0)
                self.mouth_y_spin.setValue(0)
                
            # Update data source
            if result.data_source == "manual":
                self.data_source_combo.setCurrentText("Manual")
            elif result.data_source == "ofiq":
                self.data_source_combo.setCurrentText("OFIQ")
            else:
                self.data_source_combo.setCurrentText("Detected")
                
            # Update manual confidence
            self.manual_confidence_spin.setValue(result.manual_confidence)
            
    @pyqtSlot()
    def on_manual_data_changed(self):
        """Handle manual data changes."""
        # Enable update button when data changes
        self.update_manual_button.setEnabled(True)
        
    def update_manual_data(self):
        """Update the selected face with manual data."""
        index = self.face_selection_combo.currentIndex()
        if 0 <= index < len(self.detection_results):
            result = self.detection_results[index]
            
            # Update bounding box
            result.face_bbox = (
                self.bbox_x_spin.value(),
                self.bbox_y_spin.value(),
                self.bbox_w_spin.value(),
                self.bbox_h_spin.value()
            )
            
            # Update left eye center (manual data)
            left_eye_x = self.left_eye_x_spin.value()
            left_eye_y = self.left_eye_y_spin.value()
            if left_eye_x > 0 and left_eye_y > 0:
                result.left_eye_center = (left_eye_x, left_eye_y)
            else:
                result.left_eye_center = None
                
            # Update right eye center (manual data)
            right_eye_x = self.right_eye_x_spin.value()
            right_eye_y = self.right_eye_y_spin.value()
            if right_eye_x > 0 and right_eye_y > 0:
                result.right_eye_center = (right_eye_x, right_eye_y)
            else:
                result.right_eye_center = None
                
            # Update mouth center (manual data)
            mouth_x = self.mouth_x_spin.value()
            mouth_y = self.mouth_y_spin.value()
            if mouth_x > 0 and mouth_y > 0:
                result.mouth_center = (mouth_x, mouth_y)
            else:
                result.mouth_center = None
                
            # Update data source
            source_text = self.data_source_combo.currentText().lower()
            result.data_source = source_text
            
            # Update manual confidence
            result.manual_confidence = self.manual_confidence_spin.value()
            
            # Update display
            self.image_display.set_detection_results(self.detection_results)
            
            # Update results label
            self.update_results_label()
            
            # Disable update button
            self.update_manual_button.setEnabled(False)
            
            self.status_bar.showMessage(f"Updated manual data for Face {index + 1}")
            
    def update_results_label(self):
        """Update the results label with current detection data."""
        if self.detection_results:
            result_text = f"Detected {len(self.detection_results)} face(s):\n"
            for i, result in enumerate(self.detection_results):
                result_text += f"\nFace {i+1}:\n"
                result_text += f"  Bounding Box: {result.face_bbox}\n"
                
                # Show effective coordinates
                left_eye, right_eye = result.get_effective_eyes()
                mouth = result.get_effective_mouth()
                
                if left_eye:
                    source = "Manual" if result.left_eye_center else "Detected"
                    result_text += f"  Left Eye: {left_eye} ({source})\n"
                if right_eye:
                    source = "Manual" if result.right_eye_center else "Detected"
                    result_text += f"  Right Eye: {right_eye} ({source})\n"
                if mouth:
                    source = "Manual" if result.mouth_center else "Detected"
                    result_text += f"  Mouth: {mouth} ({source})\n"
                    
                result_text += f"  Confidence: {result.confidence:.2f}\n"
                result_text += f"  Algorithm: {result.algorithm}\n"
                result_text += f"  Data Source: {result.data_source}\n"
                if result.has_manual_data():
                    result_text += f"  Manual Confidence: {result.manual_confidence:.2f}\n"
        else:
            result_text = "No faces detected"
            
        self.results_label.setText(result_text)
        
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
                        
                        # Show effective coordinates
                        left_eye, right_eye = result.get_effective_eyes()
                        mouth = result.get_effective_mouth()
                        
                        if left_eye:
                            source = "Manual" if result.left_eye_center else "Detected"
                            f.write(f"  Left Eye: {left_eye} ({source})\n")
                        if right_eye:
                            source = "Manual" if result.right_eye_center else "Detected"
                            f.write(f"  Right Eye: {right_eye} ({source})\n")
                        if mouth:
                            source = "Manual" if result.mouth_center else "Detected"
                            f.write(f"  Mouth: {mouth} ({source})\n")
                            
                        f.write(f"  Confidence: {result.confidence:.2f}\n")
                        f.write(f"  Algorithm: {result.algorithm}\n")
                        f.write(f"  Data Source: {result.data_source}\n")
                        if result.has_manual_data():
                            f.write(f"  Manual Confidence: {result.manual_confidence:.2f}\n")
                        f.write("\n")
                
                # Also save as JSON for programmatic use
                json_file_path = Path(file_path).with_suffix('.json')
                import json
                with open(json_file_path, 'w') as f:
                    results_data = []
                    for result in self.detection_results:
                        results_data.append(result.to_dict())
                    json.dump(results_data, f, indent=2)
                        
                QMessageBox.information(
                    self,
                    "Save Complete",
                    f"Saved annotated image to:\n{file_path}\n\n"
                    f"Detection data saved to:\n{txt_file_path}\n"
                    f"JSON data saved to:\n{json_file_path}"
                )
    
    @pyqtSlot(int)
    def on_face_selected(self, face_index: int):
        """Handle face selection from image display."""
        if 0 <= face_index < len(self.detection_results):
            self.face_selection_combo.setCurrentIndex(face_index)
            self.status_bar.showMessage(f"Selected Face {face_index + 1}")
    
    @pyqtSlot(int, str, tuple)
    def on_point_edited(self, face_index: int, point_type: str, coordinates: tuple):
        """Handle point editing from image display."""
        if 0 <= face_index < len(self.detection_results):
            result = self.detection_results[face_index]
            
            if point_type == "face":
                # Update face bounding box
                x, y, w, h = coordinates
                result.face_bbox = (x, y, w, h)
                
                # Update spin boxes
                self.bbox_x_spin.setValue(x)
                self.bbox_y_spin.setValue(y)
                self.bbox_w_spin.setValue(w)
                self.bbox_h_spin.setValue(h)
                
            elif point_type == "left_eye":
                # Update left eye center
                x, y = coordinates
                result.left_eye_center = (x, y)
                
                # Update spin boxes
                self.left_eye_x_spin.setValue(x)
                self.left_eye_y_spin.setValue(y)
                
            elif point_type == "right_eye":
                # Update right eye center
                x, y = coordinates
                result.right_eye_center = (x, y)
                
                # Update spin boxes
                self.right_eye_x_spin.setValue(x)
                self.right_eye_y_spin.setValue(y)
                
            elif point_type == "mouth":
                # Update mouth center
                x, y = coordinates
                result.mouth_center = (x, y)
                
                # Update spin boxes
                self.mouth_x_spin.setValue(x)
                self.mouth_y_spin.setValue(y)
            
            # Update data source to manual
            result.data_source = "manual"
            result.manual_confidence = 1.0
            
            # Update data source combo
            self.data_source_combo.setCurrentText("Manual")
            
            # Update manual confidence
            self.manual_confidence_spin.setValue(1.0)
            
            # Update display
            self.image_display.set_detection_results(self.detection_results)
            
            # Update results label
            self.update_results_label()
            
            # Disable update button (changes already applied)
            self.update_manual_button.setEnabled(False)
            
            self.status_bar.showMessage(f"Updated {point_type} for Face {face_index + 1}")
    
    @pyqtSlot(int)
    def on_visualization_changed(self, state: int):
        """Handle visualization option changes."""
        # For now, just update the display
        self.image_display.update()
        self.status_bar.showMessage("Visualization options updated")
    
    @pyqtSlot(int)
    def on_comparison_mode_changed(self, state: int):
        """Handle comparison mode toggle."""
        if state == Qt.CheckState.Checked.value:
            self.status_bar.showMessage("Comparison mode enabled - showing detected vs manual")
        else:
            self.status_bar.showMessage("Comparison mode disabled")
        self.image_display.update()
