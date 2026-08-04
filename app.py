# app.py - Week 6: Real CV Integration
from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)

# Make sure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Supported image formats
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Import real CV functions
from cv_module.face_detector import detect_faces, draw_faces, detect_faces_with_eyes
from cv_module.shape_color_detector import detect_shapes, detect_dominant_colors, analyze_image

# Safe database import
try:
    from storage.results_store import save_cv_result, get_results
except Exception as e:
    print(f"Warning: Database modules not fully ready. Using mocks. Error: {e}")

    def save_cv_result(img_path, data):
        print(f"[DB MOCK] Saved result for: {img_path}")
        return True

    def get_results(limit=10):
        print(f"[DB MOCK] Fetching {limit} results.")
        return []


# Helper: validate uploaded image
def _validate_image_path(image_path):
    """Returns (is_valid, error_message)."""
    if not image_path:
        return False, "No image path provided"
    if not os.path.exists(image_path):
        return False, f"Image not found: {image_path}"
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    return True, None


# 1. Full image analysis (faces + shapes + colors)
@app.route('/analyze/image', methods=['POST'])
def analyze_image_endpoint():
    """Accepts an image file upload or a JSON image_path, runs analyze_image, returns JSON."""
    image_path = None

    # Case 1: image sent as a file upload (Form-Data)
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

    # Case 2: image path sent as JSON
    elif request.is_json:
        data = request.get_json()
        image_path = data.get("image_path")

    else:
        return jsonify({"error": "Send an image file or an image_path in JSON"}), 400

    # Validate path and format
    valid, err = _validate_image_path(image_path)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        # Run full CV analysis (faces + shapes + colors)
        analysis_result = analyze_image(image_path)

        # Save to database
        db_id = save_cv_result(image_path, analysis_result)
        analysis_result["db_id"] = db_id

        return jsonify({
            "message": "Image analyzed successfully",
            "analysis": analysis_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 2. Face detection only
@app.route('/analyze/faces', methods=['POST'])
def analyze_faces_endpoint():
    """Runs detect_faces on the given image and returns the result."""
    image_path = None

    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
    elif request.is_json:
        image_path = request.get_json().get("image_path")

    valid, err = _validate_image_path(image_path)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        result = detect_faces(image_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 3. Shape detection only
@app.route('/analyze/shapes', methods=['POST'])
def analyze_shapes_endpoint():
    """Runs detect_shapes on the given image and returns the result."""
    image_path = None

    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
    elif request.is_json:
        image_path = request.get_json().get("image_path")

    valid, err = _validate_image_path(image_path)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        shapes = detect_shapes(image_path)
        return jsonify({"shapes": shapes, "count": len(shapes)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 4. Color detection only
@app.route('/analyze/colors', methods=['POST'])
def analyze_colors_endpoint():
    """Runs detect_dominant_colors on the given image and returns the result."""
    image_path = None
    n_colors = 3

    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        n_colors = int(request.form.get("n_colors", 3))
    elif request.is_json:
        data = request.get_json()
        image_path = data.get("image_path")
        n_colors = data.get("n_colors", 3)

    valid, err = _validate_image_path(image_path)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        colors = detect_dominant_colors(image_path, n_colors=n_colors)
        return jsonify({"colors": colors, "count": len(colors)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 5. Get last results from the database
@app.route('/results', methods=['GET'])
def get_results_endpoint():
    """Returns the last 10 results from the database."""
    try:
        results = get_results(limit=10)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# NLP endpoints (mocks - waiting for team integration)

@app.route('/analyze/intent', methods=['POST'])
def analyze_intent():
    data = request.json
    return jsonify({"intent": "mock_intent", "input": data.get("text", "")}), 200

@app.route('/analyze/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    return jsonify({"sentiment": "positive", "confidence": 0.95}), 200

@app.route('/analyze/keywords', methods=['POST'])
def analyze_keywords():
    data = request.json
    return jsonify({"keywords": ["mock", "keyword", "test"]}), 200

@app.route('/analyze/language', methods=['POST'])
def analyze_language():
    data = request.json
    return jsonify({"language": "ar", "confidence": 0.99}), 200

@app.route('/analyze/summary', methods=['POST'])
def analyze_summary():
    data = request.json
    return jsonify({"summary": "This is a mocked summary of the text."}), 200

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "total_analyses": 100,
        "cv_analyses": 40,
        "text_analyses": 60
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)