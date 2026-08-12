# main.py - Week 7: Full integration with real NLP, CV, OCR, and shared service layer
import os
import sys
import time
import json

# CV imports
from cv_module.face_detector import detect_faces, draw_faces
from cv_module.shape_color_detector import detect_shapes, detect_dominant_colors, analyze_image
from cv_module.ocr_extractor import extract_text
from services.image_analysis_pipeline import run_image_analysis_pipeline

# NLP imports via shared service layer
from services.nlp_service import (
    classify_intent,
    analyze_sentiment,
    extract_keywords,
    detect_language,
    summarize_text,
    get_analytics,
)

# Safe DB import
try:
    from storage.results_store import save_cv_result, get_results
    from storage.db_setup import init_db
    _DB_AVAILABLE = True
except Exception as e:
    _DB_AVAILABLE = False
    print(f"[Warning] Database not available: {e}")

    def save_cv_result(img_path, data):
        return None

    def get_results(limit=10):
        return []

# Constants
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# Helper functions

def print_header(title):
    """Print a styled section header."""
    width = 50
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_divider():
    print("-" * 50)


def progress_bar(task_name, steps=20, delay=0.03):
    """Display a simple progress bar for long operations."""
    print(f"\n  [..] {task_name}...", end="")
    sys.stdout.flush()
    for i in range(steps):
        time.sleep(delay)
        pct = int((i + 1) / steps * 100)
        bar = "=" * (i + 1) + "-" * (steps - i - 1)
        print(f"\r  [..] {task_name}... [{bar}] {pct}%", end="")
        sys.stdout.flush()
    print(f"\r  [OK] {task_name}... [{'=' * steps}] 100%  ")


def validate_image_path(path):
    """Validate image path exists and has a supported format. Returns (ok, result)."""
    if not path or path.strip() == "":
        return False, "[Error] Image path cannot be empty."

    path = path.strip().strip('"').strip("'")

    if not os.path.exists(path):
        return False, f"[Error] File not found: {path}"

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"[Error] Unsupported format '{ext}'. Supported formats: {supported}"

    return True, path


def validate_text(text):
    """Validate that text input is not empty."""
    if not text or text.strip() == "":
        return False, "[Error] Text cannot be empty. Please enter some text to analyze."
    return True, text.strip()


def display_face_results(result):
    """Print face detection results."""
    count = result.get("count", 0)
    print(f"\n  Faces detected: {count}")
    if count > 0:
        for i, face in enumerate(result.get("boxes", []), 1):
            box = face.get("box", [])
            print(f"     Face {i}: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}")


def display_shape_results(shapes):
    """Print shape detection results."""
    print(f"\n  Shapes detected: {len(shapes)}")
    if shapes:
        for i, s in enumerate(shapes, 1):
            print(f"     Shape {i}: {s['type']} | Sides: {s['sides']} | Area: {s['area']:.0f}")


def display_color_results(colors):
    """Print color detection results."""
    print(f"\n  Dominant colors: {len(colors)}")
    if colors:
        for i, c in enumerate(colors, 1):
            rgb = c.get("color", (0, 0, 0))
            pct = c.get("percentage", 0)
            print(f"     Color {i}: RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) -> {pct}%")


def display_db_results(rows):
    """Print database results."""
    if not rows:
        print("\n  No stored results found.")
        return

    print(f"\n  Found {len(rows)} result(s):\n")
    for row in rows:
        # row = (id, type, input_text, output_json, timestamp)
        rid = row[0]
        rtype = row[1]
        rinput = row[2]
        rtimestamp = row[4] if len(row) > 4 else "N/A"

        print(f"  +-- Result #{rid}")
        print(f"  |   Type:      {rtype}")
        print(f"  |   Input:     {rinput[:60]}{'...' if len(str(rinput)) > 60 else ''}")
        print(f"  |   Timestamp: {rtimestamp}")
        print(f"  +" + "-" * 40)


# Menu

def show_menu():
    print("\n==============================================")
    print("        Smart Content Analyzer  (CLI)         ")
    print("==============================================")
    print("  1. Text Analysis (intent + sentiment)")
    print("  2. Extended Text Analysis (keywords + language + summary)")
    print("  3. Image Analysis")
    print("  4. View All Stored Results")
    print("  5. View Statistics")
    print("  6. Exit")
    print("==============================================")


# Option handlers

def handle_text_analysis():
    """Option 1: Basic text analysis (intent + sentiment)."""
    print_header("Text Analysis (Intent + Sentiment)")
    text = input("  Enter text to analyze: ")

    ok, result = validate_text(text)
    if not ok:
        print(f"  {result}")
        return

    # Intent
    progress_bar("Classifying intent")
    intent_result = classify_intent(result)
    if intent_result["success"]:
        r = intent_result["result"]
        print(f"\n  Intent:     {r['label']}")
        if r.get("confidence") is not None:
            print(f"  Confidence: {r['confidence']:.2f}")
    else:
        print(f"\n  [Unavailable] Intent: {intent_result['error']}")

    # Sentiment
    progress_bar("Analyzing sentiment")
    sent_result = analyze_sentiment(result)
    if sent_result["success"]:
        print(f"  Sentiment:  {sent_result['result']['label']}")
    else:
        print(f"  [Unavailable] Sentiment: {sent_result['error']}")


def handle_extended_text_analysis():
    """Option 2: Extended text analysis (keywords + language + summary)."""
    print_header("Extended Text Analysis")
    text = input("  Enter text to analyze: ")

    ok, result = validate_text(text)
    if not ok:
        print(f"  {result}")
        return

    # Keywords
    progress_bar("Extracting keywords")
    kw_result = extract_keywords(result)
    if kw_result["success"]:
        print(f"\n  Keywords: {kw_result['result']['keywords']}")
    else:
        print(f"\n  [Unavailable] Keywords: {kw_result['error']}")

    # Language detection
    progress_bar("Detecting language")
    lang_result = detect_language(result)
    if lang_result["success"]:
        r = lang_result["result"]
        print(f"  Language: {r['language']} ({r['confidence']})")
    else:
        print(f"  [Unavailable] Language: {lang_result['error']}")

    # Summarize
    progress_bar("Summarizing text")
    sum_result = summarize_text(result)
    if sum_result["success"]:
        print(f"  Summary:  {sum_result['result']['summary']}")
    else:
        print(f"  [Unavailable] Summary: {sum_result['error']}")


def handle_image_analysis():
    """Option 3: Shared CV -> OCR -> NLP image analysis pipeline."""
    print_header("Image Analysis")
    image_path = input("  Enter image path: ")

    # Validate
    ok, result = validate_image_path(image_path)
    if not ok:
        print(f"  {result}")
        return
    image_path = result  # cleaned path

    print(f"\n  Analyzing: {image_path}")
    print_divider()

    progress_bar("Running CV, OCR, and NLP analysis")
    combined_result = run_image_analysis_pipeline(image_path, save_to_db=_DB_AVAILABLE)

    face_result = combined_result["faces"]
    shapes = combined_result["shapes"]
    colors = combined_result["colors"]
    display_face_results(face_result)
    display_shape_results(shapes)
    display_color_results(colors)

    ocr_result = combined_result["ocr"]
    if ocr_result["success"] and ocr_result["text"]:
        print(f"\n  Extracted text: {ocr_result['text'][:100]}")
        print("  NLP analysis:")
        for name, nlp_result in combined_result["nlp"].items():
            if nlp_result["success"]:
                print(f"    {name}: {nlp_result['result']}")
            else:
                print(f"    {name}: unavailable ({nlp_result['error']})")
    elif ocr_result["error"]:
        print(f"\n  [OCR] {ocr_result['error']}")
    else:
        print(f"\n  No text found in image.")

    print_divider()
    if "db_id" in combined_result:
        print(f"  Saved to database (ID: {combined_result['db_id']})")
    elif "db_error" in combined_result:
        print(f"  [Warning] {combined_result['db_error']}")
    else:
        print("  [Warning] Database not available - results not saved.")

    # Step 6: Save drawn output
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base}_faces_output.jpg")

    if face_result.get("count", 0) > 0:
        try:
            draw_faces(image_path, output_path)
            print(f"  Face output saved: {output_path}")
        except Exception as e:
            print(f"  [Warning] Could not save drawn faces: {e}")

    print_divider()
    print("  [OK] Image analysis complete!")


def handle_view_results():
    """Option 4: View all stored results from the database."""
    print_header("Stored Results")

    if not _DB_AVAILABLE:
        print("  [Warning] Database not available.")
        return

    progress_bar("Fetching results")
    try:
        rows = get_results(limit=10)
        display_db_results(rows)
    except Exception as e:
        print(f"  [Error] Error fetching results: {e}")


def handle_statistics():
    """Option 5: View statistics."""
    print_header("Analytics & Statistics")

    progress_bar("Computing statistics")

    analytics_result = get_analytics()
    if analytics_result["success"]:
        stats = analytics_result["result"]
        print(f"\n  Analyses by type:")
        for type_name, count in stats.items():
            print(f"    {type_name}: {count}")
    else:
        print(f"  [Unavailable] Analytics: {analytics_result['error']}")


# Main loop

def main():
    # Initialize database on startup
    if _DB_AVAILABLE:
        try:
            init_db()
        except Exception:
            pass

    while True:
        show_menu()
        choice = input("\n  Choose an option (1-6): ").strip()

        if choice == "1":
            handle_text_analysis()
        elif choice == "2":
            handle_extended_text_analysis()
        elif choice == "3":
            handle_image_analysis()
        elif choice == "4":
            handle_view_results()
        elif choice == "5":
            handle_statistics()
        elif choice == "6":
            print("\n  Goodbye!\n")
            break
        else:
            print("\n  [Error] Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()
