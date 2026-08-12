"""Shared orchestration for CV, OCR, NLP, and optional result storage."""

from cv_module.ocr_extractor import extract_text
from cv_module.shape_color_detector import analyze_image
from services import nlp_service


def analyze_extracted_text(text):
    """Run the shared NLP service on non-empty OCR text."""
    return {
        "intent": nlp_service.classify_intent(text),
        "sentiment": nlp_service.analyze_sentiment(text),
        "keywords": nlp_service.extract_keywords(text),
        "language": nlp_service.detect_language(text),
        "summary": nlp_service.summarize_text(text),
    }


def run_image_analysis_pipeline(image_path, save_to_db=False):
    """Analyze an image through CV, OCR, NLP, and optional database storage.

    OCR failures are returned in the result and do not prevent the CV result from
    being returned. NLP runs only when OCR produces non-empty text.
    """
    result = analyze_image(image_path, save_to_db=False)
    ocr_result = extract_text(image_path)
    result["ocr"] = ocr_result
    result["nlp"] = None

    if ocr_result.get("success") and ocr_result.get("text", "").strip():
        result["nlp"] = analyze_extracted_text(ocr_result["text"])

    if save_to_db:
        try:
            from storage.results_store import save_cv_result

            result["db_id"] = save_cv_result(image_path, result)
        except Exception as exc:
            result["db_error"] = f"Failed to save image analysis: {exc}"

    return result
