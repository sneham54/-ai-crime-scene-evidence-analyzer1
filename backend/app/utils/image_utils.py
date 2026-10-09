import cv2
import numpy as np
from PIL import Image
import io
import base64

def cv2_to_base64(img):
    """Convert OpenCV image (BGR) to base64 PNG string."""
    if img is None:
        return None
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def base64_to_cv2(b64_string):
    data = base64.b64decode(b64_string)
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def save_annotated_image(annotated_np, output_path):
    """Save numpy annotated image from YOLO."""
    cv2.imwrite(output_path, annotated_np)
    return output_path
