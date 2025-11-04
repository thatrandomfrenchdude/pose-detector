# Pose Detection Application

A simple, real-time human pose detection application using MediaPipe and OpenCV. This application can process live camera feeds or static images to detect and visualize human pose keypoints.

## Features

- **Real-time pose detection** from webcam feed
- **Single image processing** for static pose analysis
- **Multiple camera support** with automatic detection
- **FPS monitoring** for performance tracking
- **Mirror mode** for natural camera interaction
- **Cross-platform compatibility** (Windows, macOS, Linux)

## Installation

### Prerequisites

- Python 3.7 or higher
- A webcam (for real-time detection)

### Setup

1. **Clone or download this repository**
   ```bash
   cd pose-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Real-time Camera Detection

Run the application with your default camera:
```bash
python main.py
```

Specify a different camera:
```bash
python main.py --camera 1
```

### List Available Cameras

Check which cameras are available on your system:
```bash
python main.py --list-cameras
```

### Process a Single Image

Detect pose in a static image:
```bash
python main.py --image path/to/your/image.jpg
```

Save the result to a file:
```bash
python main.py --image input.jpg --output output.jpg
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--camera` | Camera index to use | 0 |
| `--image` | Path to input image file | None |
| `--output` | Path to save processed image | None |
| `--list-cameras` | List available cameras | False |

## Controls

When running real-time detection:
- **Press 'q'** to quit the application
- **Ctrl+C** to force stop

## Architecture

### Core Components

1. **PoseDetector Class** (`PoseDetector`)
   - Wraps MediaPipe pose estimation
   - Handles landmark detection and visualization
   - Configurable detection parameters

2. **CameraProcessor Class** (`CameraProcessor`)
   - Manages camera input and video processing
   - Handles both real-time and static image processing
   - Provides FPS monitoring and resource management

### Key Features of the Implementation

- **MediaPipe Integration**: Uses Google's MediaPipe for robust pose detection
- **Real-time Performance**: Optimized for smooth video processing
- **Error Handling**: Comprehensive error handling for camera and file operations
- **Resource Management**: Proper cleanup of camera and MediaPipe resources
- **Flexible Input**: Supports both camera streams and static images

## Pose Landmarks

The application detects 33 body landmarks following the MediaPipe pose model:

- **Face**: Nose, eyes, ears, mouth corners
- **Arms**: Shoulders, elbows, wrists, hand landmarks
- **Torso**: Shoulder, hip connections
- **Legs**: Hips, knees, ankles, foot landmarks

Each landmark includes:
- X, Y coordinates (normalized to image dimensions)
- Visibility score (confidence that the landmark is visible)

## Customization

### Adjusting Detection Parameters

Modify the `PoseDetector` initialization in `main.py`:

```python
self.pose = self.mp_pose.Pose(
    static_image_mode=False,           # For video stream
    model_complexity=1,                # 0=Light, 1=Full, 2=Heavy
    smooth_landmarks=True,             # Temporal smoothing
    enable_segmentation=False,         # Person segmentation
    min_detection_confidence=0.5,      # Detection threshold
    min_tracking_confidence=0.5        # Tracking threshold
)
```

### Visualization Customization

Modify drawing parameters in the `detect_pose` method:

```python
landmark_drawing_spec=self.mp_drawing.DrawingSpec(
    color=(0, 255, 0),    # Green landmarks
    thickness=2, 
    circle_radius=2
),
connection_drawing_spec=self.mp_drawing.DrawingSpec(
    color=(0, 0, 255),    # Red connections
    thickness=2
)
```

## Performance Tips

1. **Camera Resolution**: Lower resolution improves FPS
   ```python
   self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduce for better performance
   self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

2. **Model Complexity**: Use lighter model for faster processing
   ```python
   model_complexity=0  # Fastest option
   ```

3. **Smoothing**: Disable smoothing for slightly better performance
   ```python
   smooth_landmarks=False
   ```

## Troubleshooting

### Common Issues

1. **Camera not found**
   ```
   Error: Cannot open camera 0. Available cameras: []
   ```
   - Run `python main.py --list-cameras` to check available cameras
   - Try different camera indices with `--camera 1`, `--camera 2`, etc.
   - Ensure no other applications are using the camera

2. **Import errors**
   ```
   Import "mediapipe" could not be resolved
   ```
   - Install dependencies: `pip install -r requirements.txt`
   - Ensure you're using the correct Python environment

3. **Low FPS performance**
   - Reduce camera resolution
   - Use `model_complexity=0` for faster processing
   - Close other applications using CPU/camera

4. **Image file not found**
   ```
   Could not load image: path/to/image.jpg
   ```
   - Check the file path is correct
   - Ensure the image format is supported (jpg, png, bmp, etc.)

## Dependencies

- **OpenCV** (`opencv-python`): Computer vision and camera handling
- **MediaPipe** (`mediapipe`): Google's pose detection framework
- **NumPy** (`numpy`): Numerical operations and array handling
- **Pillow** (`pillow`): Image processing support

## Development Roadmap

Potential future enhancements:

1. **Multiple Person Detection**: Extend to detect multiple people simultaneously
2. **Pose Classification**: Add gesture/pose recognition capabilities
3. **3D Pose Estimation**: Include depth information for 3D pose
4. **Performance Analytics**: Add pose analysis and movement tracking
5. **Video File Processing**: Support for processing video files
6. **Custom Model Integration**: Support for custom trained pose models
7. **REST API**: Web service interface for pose detection
8. **Mobile Support**: Android/iOS mobile app versions

## Contributing

When extending this application:

1. **Follow the existing architecture** with separate classes for detection and processing
2. **Add comprehensive error handling** for new features
3. **Update the README** with new functionality
4. **Test across different platforms** and camera configurations
5. **Maintain backward compatibility** with existing command-line options

## License

This project is open source and available under the MIT License.