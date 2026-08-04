# services/nlp_service.py
# Shared NLP integration layer used by both app.py and main.py.
# Uses importlib to load nlp-core modules (hyphen prevents normal imports).

import sys
import os
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NLP_CORE_DIR = PROJECT_ROOT / "nlp-core"


# -- Dynamic loader for nlp-core (hyphen in folder name) --------------------

def _load_nlp_core_module(module_name, filename):
    """Load a module from the nlp-core directory using importlib."""
    file_path = NLP_CORE_DIR / filename
    if not file_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


# -- Load NLP modules -------------------------------------------------------

_intent_module = _load_nlp_core_module("intent_classifier", "intent_classifier.py")
_sentiment_module = _load_nlp_core_module("sentiment_analyzer", "sentiment_analyzer.py")

# nlp_extensions uses underscores, so normal imports work
_language_detector = None
_keyword_extractor = None
_summarizer = None
_analytics = None

try:
    from nlp_extensions.language_detector import detect_language_with_confidence as _detect_lang
    _language_detector = _detect_lang
except ImportError:
    pass

try:
    from nlp_extensions.keyword_extractor import extract_by_frequency as _extract_kw
    _keyword_extractor = _extract_kw
except ImportError:
    pass

try:
    from nlp_extensions.summarizer import summarize as _summarize_fn
    _summarizer = _summarize_fn
except ImportError:
    pass

try:
    from storage.analytics import total_analyses_by_type as _analytics_fn
    _analytics = _analytics_fn
except ImportError:
    pass


# -- Consistent result schema -----------------------------------------------
# Every NLP function returns: {"success": bool, "result": ..., "error": str|None}

def _unavailable(name):
    """Return a standard unavailable-error result."""
    return {
        "success": False,
        "result": None,
        "error": f"{name} module is not available"
    }


def classify_intent(text):
    """Classify the intent of Arabic text."""
    if _intent_module is None:
        return _unavailable("intent_classifier")
    try:
        raw = _intent_module.classify_intent(text)
        return {
            "success": True,
            "result": {
                "label": raw.get("result", {}).get("label"),
                "confidence": raw.get("result", {}).get("confidence"),
                "language": raw.get("language"),
            },
            "error": None
        }
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


def analyze_sentiment(text):
    """Analyze sentiment of Arabic text."""
    if _sentiment_module is None:
        return _unavailable("sentiment_analyzer")
    try:
        prediction = _sentiment_module.analyze_sentiment(text)
        return {
            "success": True,
            "result": {
                "label": prediction,
            },
            "error": None
        }
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


def detect_language(text):
    """Detect the language of the input text."""
    if _language_detector is None:
        return _unavailable("language_detector")
    try:
        raw = _language_detector(text)
        return {
            "success": True,
            "result": {
                "language": raw.get("language"),
                "confidence": raw.get("confidence"),
            },
            "error": None
        }
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


def extract_keywords(text, top_n=5):
    """Extract keywords using frequency-based method."""
    if _keyword_extractor is None:
        return _unavailable("keyword_extractor")
    try:
        keywords = _keyword_extractor(text, top_n=top_n)
        return {
            "success": True,
            "result": {"keywords": keywords},
            "error": None
        }
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


def summarize_text(text, n_sentences=2):
    """Summarize the given text."""
    if _summarizer is None:
        return _unavailable("summarizer")
    try:
        summary = _summarizer(text, n_sentences=n_sentences)
        return {
            "success": True,
            "result": {"summary": summary},
            "error": None
        }
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


def get_analytics():
    """Get analysis statistics from the database."""
    if _analytics is None:
        return _unavailable("analytics")
    try:
        stats = _analytics()
        return {
            "success": True,
            "result": stats,
            "error": None
        }
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
