# Configuration settings for the pose detection application

# Camera settings
DEFAULT_CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Pose detection parameters
MODEL_COMPLEXITY = 1          # 0=Light, 1=Full, 2=Heavy
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
ENABLE_SMOOTHING = True
ENABLE_SEGMENTATION = False

# Visualization settings
LANDMARK_COLOR = (0, 255, 0)     # Green (B, G, R)
CONNECTION_COLOR = (0, 0, 255)   # Red (B, G, R)
LANDMARK_THICKNESS = 2
CONNECTION_THICKNESS = 2
LANDMARK_RADIUS = 2

# Performance settings
ENABLE_FPS_DISPLAY = True
ENABLE_MIRROR_MODE = True        # Flip camera horizontally

# Output settings
DEFAULT_OUTPUT_DIR = "output"
IMAGE_OUTPUT_FORMAT = "jpg"
IMAGE_QUALITY = 95