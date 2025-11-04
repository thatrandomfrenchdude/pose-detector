import cv2
import numpy as np

def create_sample_image():
    """Create a simple sample image for testing pose detection."""
    # Create a 480x640 white image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a simple stick figure
    # Head
    cv2.circle(img, (320, 100), 30, (0, 0, 0), 2)
    
    # Body
    cv2.line(img, (320, 130), (320, 300), (0, 0, 0), 3)
    
    # Arms
    cv2.line(img, (320, 180), (250, 220), (0, 0, 0), 3)  # Left arm
    cv2.line(img, (320, 180), (390, 220), (0, 0, 0), 3)  # Right arm
    
    # Legs
    cv2.line(img, (320, 300), (270, 400), (0, 0, 0), 3)  # Left leg
    cv2.line(img, (320, 300), (370, 400), (0, 0, 0), 3)  # Right leg
    
    # Add some text
    cv2.putText(img, "Sample Person", (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

if __name__ == "__main__":
    sample_img = create_sample_image()
    cv2.imwrite("sample_person.jpg", sample_img)
    print("Sample image created: sample_person.jpg")