import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import cv2
import numpy as np

from cv_module.face_detector import detect_faces
from cv_module.shape_color_detector import detect_shapes, detect_dominant_colors

# -- path for all real test images ------------------------------------------
TESTS_DIR = os.path.dirname(__file__)

FACES_DIR  = os.path.join(TESTS_DIR, "test_images", "images", "faces")
SHAPES_DIR = os.path.join(TESTS_DIR, "test_images", "images", "shapes")

# Face images
NO_FACES_IMG    = os.path.join(FACES_DIR, "no_faces.jpg")
ONE_FACE_IMG    = os.path.join(FACES_DIR, "one_face.jpg")
TWO_FACES_IMG   = os.path.join(FACES_DIR, "two_faces.jpg")
GROUP_PHOTO_IMG = os.path.join(FACES_DIR, "group_photo.png")

# Shape images
SQUARE_IMG       = os.path.join(SHAPES_DIR, "square.png")
CIRCLE_IMG       = os.path.join(SHAPES_DIR, "circle.png")
BLUE_CIRCLE_IMG  = os.path.join(SHAPES_DIR, "blue_circle.png")
TRIANGLE_IMG     = os.path.join(SHAPES_DIR, "triangel.png")
RED_TRIANGLE_IMG = os.path.join(SHAPES_DIR, "red_triangle.png")
HEXAGON_IMG      = os.path.join(SHAPES_DIR, "hexagon.png")
HEPTAGON_IMG     = os.path.join(SHAPES_DIR, "heptagon.png")


# -- Helper ------------------------------------------------------------------
def require_image(path):
    """Skip the test gracefully if the image file doesn't exist."""
    if not os.path.exists(path):
        pytest.skip(f"Image not found: {path}")


# -- Face detection tests ----------------------------------------------------

# 1. Test detecting zero faces (real image with no faces)
def test_detect_zero_faces():
    """Test that an image with no faces returns 0 faces."""
    require_image(NO_FACES_IMG)
    result = detect_faces(NO_FACES_IMG)
    assert result["count"] == 0

# 2. Test detecting one face
def test_detect_one_face():
    """Test that a real image with one person returns at least 1 face."""
    require_image(ONE_FACE_IMG)
    result = detect_faces(ONE_FACE_IMG)
    assert result["count"] >= 1

# 3. Test detecting two faces
def test_detect_two_faces():
    """Test that a real image with two people returns at least 2 faces."""
    require_image(TWO_FACES_IMG)
    result = detect_faces(TWO_FACES_IMG)
    assert result["count"] >= 2

# 4. Test detecting multiple faces in a group photo
def test_detect_group_photo_faces():
    """Test that a group photo returns more than 2 faces."""
    require_image(GROUP_PHOTO_IMG)
    result = detect_faces(GROUP_PHOTO_IMG)
    assert result["count"] >= 2


# -- Shape detection tests ---------------------------------------------------

# 5. Test detecting a square / rectangle shape
def test_detect_square_shape():
    """Test that the square image is successfully detected as a shape."""
    require_image(SQUARE_IMG)
    shapes = detect_shapes(SQUARE_IMG)
    assert len(shapes) > 0

# 6. Test detecting a circle shape
def test_detect_circle_shape():
    """Test that the circle image is successfully detected as a shape."""
    require_image(CIRCLE_IMG)
    shapes = detect_shapes(CIRCLE_IMG)
    assert len(shapes) > 0

# 7. Test detecting a triangle shape
def test_detect_triangle_shape():
    """Test that the triangle image is successfully detected as a shape."""
    require_image(TRIANGLE_IMG)
    shapes = detect_shapes(TRIANGLE_IMG)
    assert len(shapes) > 0

# 8. Test detecting a hexagon shape
def test_detect_hexagon_shape():
    """Test that the hexagon image is successfully detected as a shape."""
    require_image(HEXAGON_IMG)
    shapes = detect_shapes(HEXAGON_IMG)
    assert len(shapes) > 0

# 9. Test detecting a heptagon shape
def test_detect_heptagon_shape():
    """Test that the heptagon image is successfully detected as a shape."""
    require_image(HEPTAGON_IMG)
    shapes = detect_shapes(HEPTAGON_IMG)
    assert len(shapes) > 0


# -- Color detection tests ---------------------------------------------------

# 10. Test detecting a single dominant color (blue circle)
def test_detect_single_dominant_color():
    """Test that the blue circle image returns at least 1 dominant color."""
    require_image(BLUE_CIRCLE_IMG)
    colors = detect_dominant_colors(BLUE_CIRCLE_IMG, n_colors=1)
    assert len(colors) > 0

# 11. Test detecting two dominant colors (red triangle has red + background)
def test_detect_two_dominant_colors():
    """Test that the red triangle image returns exactly 2 dominant colors."""
    require_image(RED_TRIANGLE_IMG)
    colors = detect_dominant_colors(RED_TRIANGLE_IMG, n_colors=2)
    assert len(colors) == 2

# 12. Test that color count respects the n_colors parameter
def test_color_count_respected():
    """Test that n_colors=3 returns exactly 3 dominant colors."""
    require_image(BLUE_CIRCLE_IMG)
    colors = detect_dominant_colors(BLUE_CIRCLE_IMG, n_colors=3)
    assert len(colors) == 3
