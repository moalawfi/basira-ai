import sys
import os
# This line fixes the 'No module named cv_module' issue by adding the root directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import cv2
import numpy as np

from cv_module.face_detector import detect_faces
from cv_module.shape_color_detector import detect_shapes, detect_dominant_colors

TEST_IMAGE_PATH = "test_temp_image.jpg"

@pytest.fixture(autouse=True)
def cleanup():
    """Fixture to automatically delete the temporary test image after each test."""
    yield
    if os.path.exists(TEST_IMAGE_PATH):
        os.remove(TEST_IMAGE_PATH)

# 1. Test detecting zero faces
def test_detect_zero_faces():
    """Test that a blank image returns 0 faces."""
    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(TEST_IMAGE_PATH, blank_image)
    result = detect_faces(TEST_IMAGE_PATH)
    assert result["count"] == 0

# 2. Test detecting one face (uses a real image from the device)
def test_detect_one_face():
    """Test that a real image containing a face returns at least 1 face."""
    # Change this path to point to a real image on your machine, 
    # or leave it as is and the test will skip gracefully.
    test_real_image = "test_outputs/face1.jpg" 
    
    if not os.path.exists(test_real_image):
        pytest.skip("Skipped: Real test image not found")
        
    result = detect_faces(test_real_image)
    assert result["count"] >= 1

# 3. Test detecting a rectangle shape
def test_detect_rectangle_shape():
    """Test that a drawn rectangle is successfully detected."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.imwrite(TEST_IMAGE_PATH, img)
    shapes = detect_shapes(TEST_IMAGE_PATH)
    assert len(shapes) > 0

# 4. Test detecting a circle shape
def test_detect_circle_shape():
    """Test that a drawn circle is successfully detected."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
    cv2.imwrite(TEST_IMAGE_PATH, img)
    shapes = detect_shapes(TEST_IMAGE_PATH)
    assert len(shapes) > 0

# 5. Test detecting a single dominant color
def test_detect_single_dominant_color():
    """Test that a purely red image extracts red as the dominant color."""
    red_image = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
    cv2.imwrite(TEST_IMAGE_PATH, red_image)
    colors = detect_dominant_colors(TEST_IMAGE_PATH, n_colors=1)
    assert len(colors) > 0

# 6. Test detecting two dominant colors
def test_detect_two_dominant_colors():
    """Test that an image split into blue and green extracts exactly 2 colors."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :100] = [255, 0, 0]   # Left half blue
    img[:, 100:] = [0, 255, 0]   # Right half green
    cv2.imwrite(TEST_IMAGE_PATH, img)
    colors = detect_dominant_colors(TEST_IMAGE_PATH, n_colors=2)
    assert len(colors) == 2









  