# main.py - Week 6: Improved CLI with real CV, error handling, and progress bar
import os
import sys
import time
import json

# Imports
from cv_module.face_detector import detect_faces, draw_faces
from cv_module.shape_color_detector import detect_shapes, detect_dominant_colors, analyze_image
from cv_module.ocr_extractor import extract_text

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
    """Option 1: Basic text analysis (mock)."""
    print_header("Text Analysis (Intent + Sentiment)")
    text = input("  Enter text to analyze: ")

    ok, result = validate_text(text)
    if not ok:
        print(f"  {result}")
        return

    progress_bar("Analyzing text")
    print(f"\n  Intent:    mock_intent")
    print(f"  Sentiment: positive (0.95)")
    print(f"\n  (NLP functions are mocks - waiting for team integration)")


def handle_extended_text_analysis():
    """Option 2: Extended text analysis (mock)."""
    print_header("Extended Text Analysis")
    text = input("  Enter text to analyze: ")

    ok, result = validate_text(text)
    if not ok:
        print(f"  {result}")
        return

    progress_bar("Extracting keywords")
    print(f"\n  Keywords: ['mock', 'keyword', 'test']")

    progress_bar("Detecting language")
    print(f"  Language: ar (0.99)")

    progress_bar("Summarizing text")
    print(f"  Summary:  This is a mocked summary of the text.")
    print(f"\n  (NLP functions are mocks - waiting for team integration)")


def handle_image_analysis():
    """Option 3: Full image analysis using real CV functions -> DB -> display."""
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

    # Step 1: Face detection
    progress_bar("Detecting faces")
    try:
        face_result = detect_faces(image_path)
        display_face_results(face_result)
    except Exception as e:
        print(f"  [Error] Face detection failed: {e}")
        face_result = {"count": 0, "boxes": []}

    # Step 2: Shape detection
    progress_bar("Detecting shapes")
    try:
        shapes = detect_shapes(image_path)
        display_shape_results(shapes)
    except Exception as e:
        print(f"  [Error] Shape detection failed: {e}")
        shapes = []

    # Step 3: Color detection
    progress_bar("Extracting dominant colors")
    try:
        colors = detect_dominant_colors(image_path, n_colors=3)
        display_color_results(colors)
    except Exception as e:
        print(f"  [Error] Color detection failed: {e}")
        colors = []

    # Step 4: OCR
    progress_bar("Extracting text (OCR)")
    try:
        ocr_result = extract_text(image_path)
        extracted = ocr_result.get("text", "")
        if extracted:
            print(f"\n  Extracted text: {extracted[:100]}")
        else:
            print(f"\n  No text found in image.")
    except Exception as e:
        print(f"  [Error] OCR failed: {e}")

    # Step 5: Save to DB
    print_divider()
    combined_result = {
        "image_path": image_path,
        "faces": face_result,
        "shapes": shapes,
        "colors": colors,
        "summary": {
            "num_faces": face_result.get("count", 0),
            "num_shapes": len(shapes),
            "num_colors": len(colors),
        }
    }

    if _DB_AVAILABLE:
        progress_bar("Saving to database")
        try:
            db_id = save_cv_result(image_path, combined_result)
            print(f"  Saved to database (ID: {db_id})")
        except Exception as e:
            print(f"  [Warning] Database save failed: {e}")
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

    if not _DB_AVAILABLE:
        print("  [Warning] Database not available.")
        return

    progress_bar("Computing statistics")

    try:
        all_results = get_results(limit=1000)
        cv_count = sum(1 for r in all_results if r[1] == 'cv')
        nlp_count = sum(1 for r in all_results if r[1] == 'nlp')
        total = len(all_results)

        print(f"\n  Total analyses: {total}")
        print(f"  CV analyses:    {cv_count}")
        print(f"  NLP analyses:   {nlp_count}")
    except Exception as e:
        print(f"  [Error] Error computing stats: {e}")


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
