import os
import cv2


def extract_text(image_path):
    """
    Extract text from an image using Tesseract OCR.

    Returns a dict with a stable schema:
      - success (bool): whether OCR ran and found text
      - text (str): the extracted string, empty if unavailable
      - error (str or None): error description if something went wrong
    """
    if not os.path.exists(image_path):
        return {"success": False, "text": "", "error": f"Image not found: {image_path}"}

    try:
        import pytesseract
    except ImportError:
        return {
            "success": False,
            "text": "",
            "error": "pytesseract package is not installed"
        }

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "text": "", "error": f"Could not read image: {image_path}"}

        # Convert to grayscale for better OCR accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply threshold to clean up the image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        extracted = pytesseract.image_to_string(thresh).strip()

        return {
            "success": True,
            "text": extracted,
            "error": None
        }

    except pytesseract.TesseractNotFoundError:
        return {
            "success": False,
            "text": "",
            "error": "Tesseract binary is not installed or not in PATH. "
                     "See docs/LIMITATIONS.md for details."
        }
    except Exception as e:
        return {"success": False, "text": "", "error": f"OCR error: {str(e)}"}

