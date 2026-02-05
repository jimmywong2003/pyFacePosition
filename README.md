# pyFacePosition - Face Detection Application

A Python-based application for facial feature detection and visualization with a PyQt6 graphical user interface.

## Features

- **Image Loading**: Support for common image formats (JPG, PNG, BMP, TIFF)
- **Face Detection**: Detect faces using OpenCV Haar cascades
- **Facial Feature Detection**: Identify eyes and mouth positions
- **Visualization**: Color-coded annotations for different facial features
- **Interactive GUI**: User-friendly interface with real-time feedback
- **Results Export**: Save annotated images and detection data
- **Algorithm Comparison**: Framework for comparing different detection algorithms

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd pyFacePosition
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the GUI application**:
   ```bash
   python main.py
   ```

2. **Using the application**:
   - Click "Load Image" to select an image file
   - Adjust detection parameters if needed
   - Click "Detect Faces" to run face detection
   - View results in the display area and results panel
   - Use "Save Results" to export annotated images and detection data
   - Use "Clear Results" to reset the current detection

### Command Line Testing

Run the test script to verify the face detector module:
```bash
python test_face_detector.py
```

## Project Structure

```
pyFacePosition/
├── main.py                 # Application entry point
├── face_detector.py        # Core face detection logic
├── gui/
│   └── main_window.py      # Main GUI window
├── test_face_detector.py   # Test script
├── requirements.txt        # Python dependencies
├── design.md              # Project design document
├── README.md              # This file
└── LICENSE                # License file
```

## Key Components

### Face Detector Module (`face_detector.py`)
- `FaceDetector` class: Main detection engine
- `FaceDetectionResult` dataclass: Container for detection results
- Support for Haar cascade detection
- Placeholder for DNN-based detection
- Image loading and saving utilities

### GUI Application (`gui/main_window.py`)
- `MainWindow` class: Main application window
- `ImageDisplayWidget`: Custom widget for image display
- `DetectionThread`: Background thread for face detection
- Interactive controls for image loading, detection, and results management

## Detection Algorithms

### Currently Implemented
1. **OpenCV Haar Cascades**
   - Uses pre-trained Haar cascade classifiers
   - Detects faces, eyes, and mouth positions
   - Configurable parameters (scale factor, min neighbors, min size)

### Planned for Future Implementation
2. **OFIQ (Open Face Image Quality)**
   - Advanced facial analysis algorithm
   - Higher accuracy for facial landmark detection
   - Quality assessment capabilities

## Color Coding

The application uses the following color scheme for visualization:
- **Green**: Face bounding boxes
- **Blue**: Left eye positions (marked with "L")
- **Red**: Right eye positions (marked with "R")
- **Yellow**: Mouth positions (marked with "M")

## Development

### Adding New Detection Algorithms

To add a new detection algorithm:

1. Extend the `FaceDetector` class in `face_detector.py`
2. Implement a new detection method (e.g., `detect_faces_ofiq()`)
3. Update the GUI to include the new algorithm option
4. Add appropriate error handling and result formatting

### Customizing the GUI

The GUI is built with PyQt6 and follows a modular design:
- `MainWindow` handles the main application logic
- `ImageDisplayWidget` manages image rendering
- Control panels are organized in groups for clarity
- Threading is used to keep the UI responsive during detection

## Testing

The project includes a test script (`test_face_detector.py`) that:
- Creates a synthetic test image with a face-like pattern
- Tests the Haar cascade detection
- Saves annotated results for visual verification
- Provides console output of detection results

## Troubleshooting

### Common Issues

1. **OpenCV cascade classifiers not found**:
   - Ensure OpenCV is properly installed
   - Check that the Haar cascade XML files are in the expected location

2. **PyQt6 import errors**:
   - Verify PyQt6 is installed: `pip show PyQt6`
   - Reinstall if necessary: `pip install --force-reinstall PyQt6`

3. **Image loading failures**:
   - Check file path and permissions
   - Verify image format is supported
   - Ensure Pillow is installed correctly

4. **Detection performance issues**:
   - Adjust detection parameters (scale factor, min neighbors)
   - Use smaller images for faster processing
   - Consider implementing DNN-based detection for better accuracy

### Debug Mode

For debugging, you can modify the application to:
- Print detailed detection information to console
- Save intermediate processing steps
- Display detection confidence scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for the computer vision libraries
- PyQt developers for the GUI framework
- Contributors and testers

## Future Enhancements

See the [design document](design.md) for detailed plans on future features and improvements.

---

**Note**: This is a development version. Some features (like OFIQ integration) are planned for future implementation.