# API Reference

Complete API documentation for HRNet Pose Detection components.

## Main Application

### `PoseDetectionApp`

Main application class that coordinates NPU and MediaPipe detectors.

```python
from src.pose_detection import PoseDetectionApp
```

#### Constructor
```python
PoseDetectionApp(model_path="model/model.onnx", force_mediapipe=False)
```

**Parameters:**
- `model_path` (str): Path to ONNX model file
- `force_mediapipe` (bool): Force MediaPipe mode, skip NPU

**Example:**
```python
# Default NPU mode
app = PoseDetectionApp()

# Force MediaPipe
app = PoseDetectionApp(force_mediapipe=True)

# Custom model
app = PoseDetectionApp(model_path="custom/model.onnx")
```

#### Methods

##### `process_camera(camera_id=0)`
Process real-time camera feed.

**Parameters:**
- `camera_id` (int): Camera device index

**Behavior:**
- Opens camera window
- Displays real-time pose detection
- Shows FPS and keypoint count
- Exits on 'q' key press

**Example:**
```python
app = PoseDetectionApp()
app.process_camera(camera_id=0)  # Blocks until quit
```

##### `process_image(image_path, output_path=None)`
Process single image file.

**Parameters:**
- `image_path` (str): Path to input image
- `output_path` (str, optional): Path to save result

**Returns:**
- `tuple`: (annotated_image, keypoint_count)

**Example:**
```python
# Display result
result_img, count = app.process_image("photo.jpg")

# Save result
result_img, count = app.process_image("photo.jpg", "result.jpg")
```

##### `get_info()`
Get detector and system information.

**Returns:**
- `dict`: System information including:
  - `detector_type`: "NPU-ONNX" or "MediaPipe"
  - `available_cameras`: List of camera indices
  - Additional model info if NPU mode

**Example:**
```python
info = app.get_info()
print(f"Using: {info['detector_type']}")
print(f"Cameras: {info['available_cameras']}")
```

##### `release()`
Release detector resources.

**Example:**
```python
app.release()  # Always call when done
```

## NPU Detector

### `ONNXPoseDetector`

NPU-accelerated pose detection using ONNX Runtime.

```python
from src.pose_detection.detectors.onnx_detector import ONNXPoseDetector
```

#### Constructor
```python
ONNXPoseDetector(model_path)
```

**Parameters:**
- `model_path` (str): Path to ONNX model file

**Raises:**
- `ImportError`: If ONNX Runtime not available
- `FileNotFoundError`: If model file not found

#### Methods

##### `detect_pose(image)`
Detect pose keypoints in image.

**Parameters:**
- `image` (np.ndarray): Input image (BGR format)

**Returns:**
- `tuple`: (annotated_image, keypoints)
  - `annotated_image` (np.ndarray): Image with keypoints drawn
  - `keypoints` (List[Tuple[int, int]]): List of (x, y) coordinates

**Example:**
```python
import cv2

detector = ONNXPoseDetector("model/model.onnx")
image = cv2.imread("photo.jpg")
result_img, keypoints = detector.detect_pose(image)

print(f"Found {len(keypoints)} keypoints")
for i, (x, y) in enumerate(keypoints):
    print(f"Keypoint {i}: ({x}, {y})")
```

##### `get_model_info()`
Get model and session information.

**Returns:**
- `dict`: Model information including:
  - `model_path`: Path to loaded model
  - `providers`: ONNX Runtime providers
  - `input_shape`: Model input dimensions
  - `output_shape`: Model output dimensions

##### `preprocess_image(image)`
Preprocess image for model inference.

**Parameters:**
- `image` (np.ndarray): Input image

**Returns:**
- `np.ndarray`: Preprocessed tensor

##### `postprocess_output(output, original_shape)`
Convert model output to keypoints.

**Parameters:**
- `output` (np.ndarray): Model output tensor
- `original_shape` (tuple): Original image shape

**Returns:**
- `List[Tuple[int, int]]`: Keypoint coordinates

## MediaPipe Detector

### `MediaPipePoseDetector`

MediaPipe pose detection fallback.

```python
from src.pose_detection.detectors.mediapipe_detector import MediaPipePoseDetector
```

#### Constructor
```python
MediaPipePoseDetector()
```

**Raises:**
- `ImportError`: If MediaPipe not available

#### Methods

##### `detect_pose(image)`
Detect pose using MediaPipe.

**Parameters:**
- `image` (np.ndarray): Input image (BGR format)

**Returns:**
- `tuple`: (annotated_image, landmarks)
  - `annotated_image` (np.ndarray): Image with pose drawn
  - `landmarks`: MediaPipe pose landmarks object

**Example:**
```python
detector = MediaPipePoseDetector()
image = cv2.imread("photo.jpg")
result_img, landmarks = detector.detect_pose(image)

if landmarks:
    print("Pose detected!")
```

##### `get_landmarks_array(landmarks, image_shape)`
Convert landmarks to coordinate array.

**Parameters:**
- `landmarks`: MediaPipe landmarks object
- `image_shape` (tuple): Image dimensions

**Returns:**
- `np.ndarray`: Array of [x, y] coordinates

**Example:**
```python
coords = detector.get_landmarks_array(landmarks, image.shape)
print(f"Landmark coordinates: {coords.shape}")
```

## Utility Functions

### Model Utilities

```python
from src.pose_detection.utils.model_utils import generate_npu_context, check_model_files
```

#### `generate_npu_context(model_path)`
Generate NPU context model for faster loading.

**Parameters:**
- `model_path` (str): Path to original ONNX model

**Returns:**
- `bool`: True if successful

**Example:**
```python
success = generate_npu_context("model/model.onnx")
if success:
    print("Context generated! Future loads will be faster.")
```

#### `check_model_files(model_path)`
Check model file availability and status.

**Parameters:**
- `model_path` (str): Path to model file

**Returns:**
- `dict`: File information

### Camera Utilities

```python
from src.pose_detection.utils.camera_utils import get_available_cameras, setup_camera
```

#### `get_available_cameras(max_cameras=5)`
Get list of available camera indices.

**Parameters:**
- `max_cameras` (int): Maximum cameras to check

**Returns:**
- `List[int]`: Available camera indices

**Example:**
```python
cameras = get_available_cameras()
print(f"Available cameras: {cameras}")
```

#### `setup_camera(camera_id, width=640, height=480, fps=30)`
Setup camera with optimal settings.

**Parameters:**
- `camera_id` (int): Camera index
- `width` (int): Frame width
- `height` (int): Frame height
- `fps` (int): Target FPS

**Returns:**
- `cv2.VideoCapture`: Configured camera object

**Example:**
```python
cap = setup_camera(0, width=1280, height=720)
# Use camera...
cap.release()
```

## Error Handling

### Common Exceptions

#### `ImportError`
Raised when required dependencies unavailable.

```python
try:
    app = PoseDetectionApp()
except ImportError as e:
    print(f"Missing dependency: {e}")
    # Install missing packages
```

#### `FileNotFoundError`
Raised when model files not found.

```python
try:
    detector = ONNXPoseDetector("missing_model.onnx")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
    # Download or specify correct path
```

#### `ValueError`
Raised for invalid parameters or camera issues.

```python
try:
    app.process_camera(camera_id=99)
except ValueError as e:
    print(f"Camera error: {e}")
    # Try different camera or check availability
```

## Best Practices

### Resource Management
```python
# Always use try/finally or context managers
app = PoseDetectionApp()
try:
    app.process_image("photo.jpg")
finally:
    app.release()
```

### Error Handling
```python
try:
    app = PoseDetectionApp()
    result = app.process_image("photo.jpg")
except ImportError:
    print("Dependencies missing - install requirements")
except FileNotFoundError:
    print("Model or image file not found")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if 'app' in locals():
        app.release()
```

### Performance Optimization
```python
# Generate context for faster loading
generate_npu_context("model/model.onnx")

# Reuse detector for multiple images
app = PoseDetectionApp()
for image_path in image_list:
    app.process_image(image_path)
app.release()
```

## Integration Examples

### Web API Integration
```python
from flask import Flask, request, jsonify
from src.pose_detection import PoseDetectionApp
import base64
import cv2

app_flask = Flask(__name__)
pose_app = PoseDetectionApp()

@app_flask.route('/detect', methods=['POST'])
def detect_pose():
    try:
        # Decode image from base64
        image_data = request.json['image']
        # ... decode and process ...
        
        result_img, count = pose_app.process_image(image_path)
        return jsonify({'keypoints': count, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Batch Processing
```python
import os
from src.pose_detection import PoseDetectionApp

def process_directory(input_dir, output_dir):
    app = PoseDetectionApp()
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                app.process_image(input_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Failed {filename}: {e}")
    
    app.release()
```

### Custom Analysis
```python
from src.pose_detection.detectors.onnx_detector import ONNXPoseDetector
import numpy as np

def analyze_pose_symmetry(image_path):
    detector = ONNXPoseDetector("model/model.onnx")
    
    image = cv2.imread(image_path)
    _, keypoints = detector.detect_pose(image)
    
    # Custom analysis logic
    left_points = keypoints[:len(keypoints)//2]
    right_points = keypoints[len(keypoints)//2:]
    
    # Calculate symmetry score
    symmetry = calculate_symmetry(left_points, right_points)
    
    detector.release()
    return symmetry
```