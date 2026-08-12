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


# -- Integration tests -------------------------------------------------------

import json
import tempfile
import sqlite3
from cv_module.shape_color_detector import analyze_image
from cv_module.ocr_extractor import extract_text
from services import image_analysis_pipeline

# 13. Test analyze_image full pipeline returns all expected keys
def test_analyze_image_full_pipeline():
    """Test that analyze_image returns a result with all expected keys."""
    require_image(ONE_FACE_IMG)
    result = analyze_image(ONE_FACE_IMG, save_to_db=False)
    assert "image_path" in result
    assert "faces" in result
    assert "shapes" in result
    assert "colors" in result
    assert "summary" in result
    summary = result["summary"]
    assert "num_faces" in summary
    assert "num_shapes" in summary
    assert "num_colors" in summary


def test_image_pipeline_passes_ocr_text_to_shared_nlp(monkeypatch):
    """Real CV and NLP plus mocked OCR verify the complete image pipeline."""
    require_image(SQUARE_IMG)
    extracted_text = "مرحبا هذا نص مستخرج من الصورة"

    monkeypatch.setattr(
        image_analysis_pipeline,
        "extract_text",
        lambda path: {"success": True, "text": extracted_text, "error": None},
    )

    calls = []
    for name in ("classify_intent", "analyze_sentiment", "extract_keywords", "detect_language", "summarize_text"):
        original = getattr(image_analysis_pipeline.nlp_service, name)

        def track_text(text, _original=original, _name=name):
            calls.append((_name, text))
            return _original(text)

        monkeypatch.setattr(image_analysis_pipeline.nlp_service, name, track_text)

    result = image_analysis_pipeline.run_image_analysis_pipeline(SQUARE_IMG)

    assert result["summary"]["num_shapes"] >= 1
    assert result["ocr"]["text"] == extracted_text
    assert {name for name, text in calls if text == extracted_text} == {
        "classify_intent", "analyze_sentiment", "extract_keywords",
        "detect_language", "summarize_text"
    }
    assert set(result["nlp"]) == {"intent", "sentiment", "keywords", "language", "summary"}


def test_image_api_runs_shared_ocr_to_nlp_pipeline(monkeypatch):
    """The image API exposes OCR and NLP results from the shared pipeline."""
    require_image(SQUARE_IMG)
    extracted_text = "مرحبا هذا نص الصورة"
    monkeypatch.setattr(
        image_analysis_pipeline,
        "extract_text",
        lambda path: {"success": True, "text": extracted_text, "error": None},
    )

    import app as api_app

    response = api_app.app.test_client().post(
        "/analyze/image",
        json={"image_path": SQUARE_IMG, "save_to_db": False},
    )

    assert response.status_code == 200
    analysis = response.get_json()["analysis"]
    assert analysis["ocr"]["text"] == extracted_text
    assert set(analysis["nlp"]) == {"intent", "sentiment", "keywords", "language", "summary"}


# 14. Test saving CV result to a temporary database (not the real one)
def test_cv_save_to_temp_db(monkeypatch, tmp_path):
    """Test that save_cv_result writes a row to the database."""
    require_image(SQUARE_IMG)

    # Create a temp DB file
    temp_db = str(tmp_path / "test_analyzer.db")

    # Monkeypatch DB_PATH in both db_setup and results_store
    import storage.db_setup as db_setup_mod
    import storage.results_store as results_store_mod
    monkeypatch.setattr(db_setup_mod, "DB_PATH", temp_db)
    monkeypatch.setattr(results_store_mod, "DB_PATH", temp_db)

    # Initialize the temp DB schema
    db_setup_mod.init_db(temp_db)

    # Save a result
    from storage.results_store import save_cv_result
    test_data = {"faces": [], "shapes": [], "colors": []}
    saved = save_cv_result(SQUARE_IMG, test_data)
    assert saved is True

    # Verify it was actually written
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM results WHERE type='cv'")
    count = cursor.fetchone()[0]
    conn.close()
    assert count >= 1


# 15. Test OCR fallback when Tesseract is not found (uses monkeypatch)
def test_ocr_fallback_tesseract_not_found(monkeypatch):
    """Test that extract_text returns a proper error schema when Tesseract is missing."""
    require_image(SQUARE_IMG)

    # Create a fake pytesseract module that raises TesseractNotFoundError
    import types
    fake_pytesseract = types.ModuleType("pytesseract")

    class FakeTesseractNotFoundError(Exception):
        pass

    fake_pytesseract.TesseractNotFoundError = FakeTesseractNotFoundError

    def fake_image_to_string(*args, **kwargs):
        raise FakeTesseractNotFoundError()

    fake_pytesseract.image_to_string = fake_image_to_string

    # Monkeypatch the import inside extract_text
    import builtins
    real_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "pytesseract":
            return fake_pytesseract
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)

    result = extract_text(SQUARE_IMG)

    # Verify stable schema
    assert "success" in result
    assert "text" in result
    assert "error" in result
    assert result["success"] is False
    assert result["text"] == ""
    assert result["error"] is not None
