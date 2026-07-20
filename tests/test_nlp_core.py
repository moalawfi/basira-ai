# -*- coding: utf-8 -*-

import importlib.util
import sys
from pathlib import Path

import pytest


# ============================================================
# UTF-8 support for Arabic output on Windows
# ============================================================

def configure_utf8_output():
    """
    Configure terminal output to use UTF-8 when supported.

    This provides the in-code equivalent needed for Arabic output.
    The command `python -X utf8` itself cannot be placed inside
    a Python file because it is a terminal option.
    """

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)

        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


configure_utf8_output()


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NLP_CORE_DIR = PROJECT_ROOT / "nlp-core"

# Make project modules, including storage, importable.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Dynamic module loader
# ============================================================

def load_module(module_name, file_path):
    """
    Load a Python module directly from its file path.

    This is needed because the nlp-core directory contains
    a hyphen and cannot be imported as a normal Python package.
    """

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module from {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


intent_classifier = load_module(
    "intent_classifier_for_tests",
    NLP_CORE_DIR / "intent_classifier.py",
)

sentiment_analyzer = load_module(
    "sentiment_analyzer_for_tests",
    NLP_CORE_DIR / "sentiment_analyzer.py",
)


# ============================================================
# Expected classes
# ============================================================

INTENT_CLASSES = {
    "أمر",
    "تحية",
    "سؤال",
    "شكوى",
}

SENTIMENT_CLASSES = {
    "إيجابي",
    "سلبي",
    "محايد",
}


# ============================================================
# Intent classification tests
# ============================================================

def test_intent_command():
    """A normal command sentence should be classified as أمر."""

    result = intent_classifier.classify_intent(
        "حلل هذا النص الآن"
    )

    assert result["type"] == "intent"
    assert result["result"]["label"] == "أمر"


def test_intent_greeting():
    """A normal greeting should be classified as تحية."""

    result = intent_classifier.classify_intent(
        "السلام عليكم"
    )

    assert result["type"] == "intent"
    assert result["result"]["label"] == "تحية"


def test_intent_question():
    """A normal question should be classified as سؤال."""

    result = intent_classifier.classify_intent(
        "هل التطبيق يدعم اللغة العربية؟"
    )

    assert result["type"] == "intent"
    assert result["result"]["label"] == "سؤال"


def test_intent_complaint():
    """A normal complaint should be classified as شكوى."""

    result = intent_classifier.classify_intent(
        "التطبيق بطيء جدًا"
    )

    assert result["type"] == "intent"
    assert result["result"]["label"] == "شكوى"


def test_intent_ambiguous_text_returns_valid_class():
    """
    A difficult or ambiguous sentence should still return
    one of the supported intent classes.
    """

    result = intent_classifier.classify_intent(
        "ممكن تفتح صفحة النتائج؟"
    )

    assert result["type"] == "intent"
    assert result["result"]["label"] in INTENT_CLASSES


def test_intent_empty_text_raises_value_error():
    """Whitespace-only input should be rejected."""

    with pytest.raises(
        ValueError,
        match="text cannot be empty",
    ):
        intent_classifier.classify_intent("   ")


def test_intent_invalid_text_type_raises_type_error():
    """A non-string input should be rejected."""

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        intent_classifier.classify_intent(None)


def test_intent_can_save_result_to_database(monkeypatch):
    """
    When save_to_db=True, classify_intent should call
    save_nlp_result with the cleaned text and result dictionary.
    """

    saved_data = {}

    def fake_save_nlp_result(input_text, results_dict):
        saved_data["input_text"] = input_text
        saved_data["results_dict"] = results_dict
        return True

    from storage import results_store

    monkeypatch.setattr(
        results_store,
        "save_nlp_result",
        fake_save_nlp_result,
    )

    result = intent_classifier.classify_intent(
        "  السلام عليكم  ",
        save_to_db=True,
    )

    assert saved_data["input_text"] == "السلام عليكم"
    assert saved_data["results_dict"] == result


def test_intent_does_not_save_by_default(monkeypatch):
    """
    Database saving should remain optional and should not occur
    when save_to_db is not enabled.
    """

    def fail_if_called(*args, **kwargs):
        pytest.fail(
            "save_nlp_result should not be called by default"
        )

    from storage import results_store

    monkeypatch.setattr(
        results_store,
        "save_nlp_result",
        fail_if_called,
    )

    result = intent_classifier.classify_intent(
        "السلام عليكم"
    )

    assert result["result"]["label"] == "تحية"


# ============================================================
# Sentiment analysis tests
# ============================================================

def test_sentiment_positive():
    """A positive sentence should be classified as إيجابي."""

    result = sentiment_analyzer.analyze_sentiment(
        "الخدمة ممتازة والموظفون متعاونون جدًا"
    )

    assert result == "إيجابي"


def test_sentiment_negative():
    """A negative sentence should be classified as سلبي."""

    result = sentiment_analyzer.analyze_sentiment(
        "الخدمة بطيئة والموظفون غير متعاونين"
    )

    assert result == "سلبي"


def test_sentiment_neutral():
    """A factual sentence should be classified as محايد."""

    result = sentiment_analyzer.analyze_sentiment(
        "بدأ الاجتماع في الساعة التاسعة صباحًا"
    )

    assert result == "محايد"


def test_sentiment_ambiguous_text_returns_valid_class():
    """
    A difficult or mixed sentence should still return
    one of the supported sentiment classes.
    """

    result = sentiment_analyzer.analyze_sentiment(
        "الخدمة كانت عادية، لا ممتازة ولا سيئة"
    )

    assert result in SENTIMENT_CLASSES


def test_sentiment_empty_text_raises_value_error():
    """An empty sentiment input should be rejected."""

    with pytest.raises(
        ValueError,
        match="text cannot be empty",
    ):
        sentiment_analyzer.analyze_sentiment("")


def test_sentiment_invalid_text_type_raises_type_error():
    """A non-string sentiment input should be rejected."""

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        sentiment_analyzer.analyze_sentiment(123)


def test_sentiment_can_save_result_to_database(monkeypatch):
    """
    When save_to_db=True, analyze_sentiment should call
    save_nlp_result with the cleaned text and result dictionary.
    """

    saved_data = {}

    def fake_save_nlp_result(input_text, results_dict):
        saved_data["input_text"] = input_text
        saved_data["results_dict"] = results_dict
        return True

    from storage import results_store

    monkeypatch.setattr(
        results_store,
        "save_nlp_result",
        fake_save_nlp_result,
    )

    result = sentiment_analyzer.analyze_sentiment(
        "  الخدمة ممتازة جدًا  ",
        save_to_db=True,
    )

    assert saved_data["input_text"] == "الخدمة ممتازة جدًا"

    assert saved_data["results_dict"] == {
        "type": "sentiment",
        "result": {
            "label": result,
        },
    }


def test_sentiment_does_not_save_by_default(monkeypatch):
    """
    Database saving should remain optional and should not occur
    when save_to_db is not enabled.
    """

    def fail_if_called(*args, **kwargs):
        pytest.fail(
            "save_nlp_result should not be called by default"
        )

    from storage import results_store

    monkeypatch.setattr(
        results_store,
        "save_nlp_result",
        fail_if_called,
    )

    result = sentiment_analyzer.analyze_sentiment(
        "الخدمة ممتازة جدًا"
    )

    assert result == "إيجابي"