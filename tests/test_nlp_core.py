# -*- coding: utf-8 -*-

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# ============================================================
# UTF-8 support for Arabic output on Windows
# ============================================================

def configure_utf8_output():
    """
    Configure terminal output to use UTF-8 when supported.

    The command ``python -X utf8`` is still recommended when
    running the tests from Windows or Git Bash.
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
EVALUATION_PATH = NLP_CORE_DIR / "EVALUATION.md"

# Make project modules, including storage and nlp_extensions,
# importable during test execution.
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

# Use Abdulrahman's detector without modifying his file.
from nlp_extensions.language_detector import (
    detect_language_with_confidence,
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
    """A normal Arabic command should be classified as أمر."""

    result = intent_classifier.classify_intent(
        "حلل هذا النص الآن"
    )

    assert result["type"] == "intent"
    assert result["language"]["language"] == "ar"
    assert result["result"]["label"] == "أمر"


def test_intent_greeting():
    """A normal Arabic greeting should be classified as تحية."""

    result = intent_classifier.classify_intent(
        "السلام عليكم"
    )

    assert result["type"] == "intent"
    assert result["language"]["language"] == "ar"
    assert result["result"]["label"] == "تحية"


def test_intent_question():
    """A normal Arabic question should be classified as سؤال."""

    result = intent_classifier.classify_intent(
        "هل التطبيق يدعم اللغة العربية؟"
    )

    assert result["type"] == "intent"
    assert result["language"]["language"] == "ar"
    assert result["result"]["label"] == "سؤال"


def test_intent_complaint():
    """A normal Arabic complaint should be classified as شكوى."""

    result = intent_classifier.classify_intent(
        "التطبيق بطيء جدًا"
    )

    assert result["type"] == "intent"
    assert result["language"]["language"] == "ar"
    assert result["result"]["label"] == "شكوى"


def test_intent_ambiguous_text_returns_valid_class():
    """
    A difficult or ambiguous Arabic sentence should still return
    one of the supported intent classes.
    """

    result = intent_classifier.classify_intent(
        "ممكن تفتح صفحة النتائج؟"
    )

    assert result["type"] == "intent"
    assert result["language"]["language"] == "ar"
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


def test_intent_english_text_is_rejected():
    """English text should not be sent to the Arabic intent model."""

    with pytest.raises(
        ValueError,
        match="supports Arabic text only",
    ):
        intent_classifier.classify_intent(
            "Hello and welcome"
        )


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
    """A positive Arabic sentence should be classified as إيجابي."""

    result = sentiment_analyzer.analyze_sentiment(
        "الخدمة ممتازة والموظفون متعاونون جدًا"
    )

    assert result == "إيجابي"


def test_sentiment_negative():
    """A negative Arabic sentence should be classified as سلبي."""

    result = sentiment_analyzer.analyze_sentiment(
        "الخدمة بطيئة والموظفون غير متعاونين"
    )

    assert result == "سلبي"


def test_sentiment_neutral():
    """A factual Arabic sentence should be classified as محايد."""

    result = sentiment_analyzer.analyze_sentiment(
        "بدأ الاجتماع في الساعة التاسعة صباحًا"
    )

    assert result == "محايد"


def test_sentiment_ambiguous_text_returns_valid_class():
    """
    A difficult or mixed Arabic sentence should still return
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


def test_sentiment_english_text_is_rejected():
    """English text should not be sent to the Arabic sentiment model."""

    with pytest.raises(
        ValueError,
        match="supports Arabic text only",
    ):
        sentiment_analyzer.analyze_sentiment(
            "The service is excellent"
        )


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


# ============================================================
# Week 7: Arabic evaluation on 10 sentences
# ============================================================

ARABIC_EVALUATION_CASES = [
    {
        "text": "هلا والله، حياكم الله",
        "intent": "تحية",
        "sentiment": "محايد",
    },
    {
        "text": "أهلين، كيف الحال؟",
        "intent": "تحية",
        "sentiment": "محايد",
    },
    {
        "text": "هل أقدر أطلع نتائج التحليل السابقة؟",
        "intent": "سؤال",
        "sentiment": "محايد",
    },
    {
        "text": "متى يكتمل رفع المستند؟",
        "intent": "سؤال",
        "sentiment": "محايد",
    },
    {
        "text": "ورني التقارير الموجودة",
        "intent": "أمر",
        "sentiment": "محايد",
    },
    {
        "text": "سجل النتيجة في قاعدة البيانات",
        "intent": "أمر",
        "sentiment": "محايد",
    },
    {
        "text": "نفذ تحليل المحتوى لهذا الملف",
        "intent": "أمر",
        "sentiment": "محايد",
    },
    {
        "text": "النظام يتأخر كثير في إظهار النتيجة",
        "intent": "شكوى",
        "sentiment": "سلبي",
    },
    {
        "text": "كل ما أحاول أرسل الملف يفشل",
        "intent": "شكوى",
        "sentiment": "سلبي",
    },
    {
        "text": "التصنيف الذي ظهر لي خاطئ تمامًا",
        "intent": "شكوى",
        "sentiment": "سلبي",
    },
]


def evaluate_arabic_cases():
    """
    Run language detection, intent classification, and sentiment
    analysis on the ten fixed Arabic evaluation sentences.
    """

    rows = []

    for case in ARABIC_EVALUATION_CASES:
        text = case["text"]

        language_result = (
            detect_language_with_confidence(text)
        )

        intent_result = (
            intent_classifier.classify_intent(text)
        )

        sentiment_prediction = (
            sentiment_analyzer.analyze_sentiment(text)
        )

        predicted_intent = (
            intent_result["result"]["label"]
        )

        rows.append(
            {
                "text": text,
                "expected_intent": case["intent"],
                "predicted_intent": predicted_intent,
                "intent_correct": (
                    predicted_intent == case["intent"]
                ),
                "expected_sentiment": case["sentiment"],
                "predicted_sentiment": sentiment_prediction,
                "sentiment_correct": (
                    sentiment_prediction
                    == case["sentiment"]
                ),
                "predicted_language": (
                    language_result["language"]
                ),
                "language_correct": (
                    language_result["language"] == "ar"
                ),
            }
        )

    total = len(rows)

    metrics = {
        "language_accuracy": (
            sum(row["language_correct"] for row in rows)
            / total
        ),
        "intent_accuracy": (
            sum(row["intent_correct"] for row in rows)
            / total
        ),
        "sentiment_accuracy": (
            sum(row["sentiment_correct"] for row in rows)
            / total
        ),
        "complete_e2e_accuracy": (
            sum(
                row["language_correct"]
                and row["intent_correct"]
                and row["sentiment_correct"]
                for row in rows
            )
            / total
        ),
    }

    return metrics, rows


def markdown_safe(value):
    """Escape Markdown table separators."""

    return str(value).replace("|", "\\|")


def update_week7_evaluation(metrics, rows):
    """
    Add or replace the Week 7 Arabic evaluation section in
    nlp-core/EVALUATION.md.
    """

    start_marker = (
        "<!-- WEEK7_ARABIC_EVALUATION_START -->"
    )

    end_marker = (
        "<!-- WEEK7_ARABIC_EVALUATION_END -->"
    )

    total = len(rows)

    language_correct = sum(
        row["language_correct"] for row in rows
    )
    intent_correct = sum(
        row["intent_correct"] for row in rows
    )
    sentiment_correct = sum(
        row["sentiment_correct"] for row in rows
    )
    complete_correct = sum(
        row["language_correct"]
        and row["intent_correct"]
        and row["sentiment_correct"]
        for row in rows
    )

    section_lines = [
        start_marker,
        "",
        "# Week 7: Arabic Language Support and E2E Testing",
        "",
        (
            "Ten fixed Arabic sentences were processed using "
            "`language_detector`, the intent classifier, and "
            "the sentiment analyzer."
        ),
        "",
        "## Accuracy Summary",
        "",
        "| Metric | Accuracy | Correct | Total |",
        "|---|---:|---:|---:|",
        (
            "| Language detection "
            f"| {metrics['language_accuracy']:.4f} "
            f"| {language_correct} | {total} |"
        ),
        (
            "| Intent classification "
            f"| {metrics['intent_accuracy']:.4f} "
            f"| {intent_correct} | {total} |"
        ),
        (
            "| Sentiment analysis "
            f"| {metrics['sentiment_accuracy']:.4f} "
            f"| {sentiment_correct} | {total} |"
        ),
        (
            "| Complete E2E result "
            f"| {metrics['complete_e2e_accuracy']:.4f} "
            f"| {complete_correct} | {total} |"
        ),
        "",
        "## Detailed Results",
        "",
        (
            "| Text | Language | Expected Intent | Predicted Intent "
            "| Expected Sentiment | Predicted Sentiment |"
        ),
        "|---|---|---|---|---|---|",
    ]

    for row in rows:
        section_lines.append(
            "| "
            + markdown_safe(row["text"])
            + " | "
            + markdown_safe(row["predicted_language"])
            + " | "
            + markdown_safe(row["expected_intent"])
            + " | "
            + markdown_safe(row["predicted_intent"])
            + " | "
            + markdown_safe(row["expected_sentiment"])
            + " | "
            + markdown_safe(row["predicted_sentiment"])
            + " |"
        )

    section_lines.extend(
        [
            "",
            (
                "A model score below `0.80` should be reviewed "
                "before deciding whether more Arabic training "
                "examples and retraining are required."
            ),
            "",
            end_marker,
        ]
    )

    new_section = "\n".join(section_lines)

    if EVALUATION_PATH.exists():
        current_content = EVALUATION_PATH.read_text(
            encoding="utf-8"
        )
    else:
        current_content = "# NLP Model Evaluation\n"

    if (
        start_marker in current_content
        and end_marker in current_content
    ):
        before = current_content.split(
            start_marker,
            1,
        )[0].rstrip()

        after = current_content.split(
            end_marker,
            1,
        )[1].lstrip()

        updated_content = (
            before
            + "\n\n"
            + new_section
        )

        if after:
            updated_content += "\n\n" + after
    else:
        updated_content = (
            current_content.rstrip()
            + "\n\n"
            + new_section
        )

    EVALUATION_PATH.write_text(
        updated_content.rstrip() + "\n",
        encoding="utf-8",
    )


def test_ten_arabic_sentences_and_document_accuracy():
    """
    Evaluate ten Arabic sentences and document the measured
    accuracy in EVALUATION.md.

    This test records the real model performance. It does not
    hide a weak score by changing the expected labels.
    """

    metrics, rows = evaluate_arabic_cases()

    update_week7_evaluation(
        metrics,
        rows,
    )

    print("\n" + "=" * 60)
    print("WEEK 7 ARABIC EVALUATION")
    print("=" * 60)
    print(
        "Language detection accuracy:",
        f"{metrics['language_accuracy']:.2%}",
    )
    print(
        "Intent accuracy:",
        f"{metrics['intent_accuracy']:.2%}",
    )
    print(
        "Sentiment accuracy:",
        f"{metrics['sentiment_accuracy']:.2%}",
    )
    print(
        "Complete E2E accuracy:",
        f"{metrics['complete_e2e_accuracy']:.2%}",
    )

    for row in rows:
        print(
            "\n"
            f"Text: {row['text']}\n"
            f"Language: {row['predicted_language']}\n"
            f"Intent: {row['predicted_intent']} "
            f"(expected {row['expected_intent']})\n"
            f"Sentiment: {row['predicted_sentiment']} "
            f"(expected {row['expected_sentiment']})"
        )

    # The detector must recognize every case as Arabic.
    assert metrics["language_accuracy"] == 1.0

    # Model predictions must remain inside their supported classes.
    assert all(
        row["predicted_intent"] in INTENT_CLASSES
        for row in rows
    )

    assert all(
        row["predicted_sentiment"] in SENTIMENT_CLASSES
        for row in rows
    )


# ============================================================
# Week 7: Complete Arabic E2E integration test
# ============================================================

def test_complete_arabic_e2e_flow(
    tmp_path,
    monkeypatch,
    capsys,
):
    """
    Test the complete flow:

    Arabic text -> language detection -> intent + sentiment ->
    SQLite saving -> retrieval -> displayed output.
    """

    import storage.db_setup as db_setup
    import storage.results_store as results_store

    test_db_path = str(
        tmp_path / "arabic_e2e.db"
    )

    monkeypatch.setattr(
        db_setup,
        "DB_PATH",
        test_db_path,
    )

    monkeypatch.setattr(
        results_store,
        "DB_PATH",
        test_db_path,
    )

    db_setup.init_db(
        test_db_path
    )

    text = "التطبيق بطيء جدًا"

    language_result = (
        detect_language_with_confidence(text)
    )

    intent_result = (
        intent_classifier.classify_intent(
            text,
            save_to_db=False,
        )
    )

    sentiment_result = (
        sentiment_analyzer.analyze_sentiment(
            text,
            save_to_db=False,
        )
    )

    combined_result = {
        "type": "nlp",
        "language": language_result,
        "result": {
            "intent": intent_result["result"],
            "sentiment": {
                "label": sentiment_result,
            },
        },
    }

    saved = results_store.save_nlp_result(
        text,
        combined_result,
    )

    assert saved is True

    retrieved_results = results_store.get_results(
        type_filter="nlp",
        limit=1,
    )

    assert len(retrieved_results) == 1

    connection = db_setup.get_connection(
        test_db_path
    )

    row = connection.execute(
        """
        SELECT input_text, output_json
        FROM results
        WHERE type = 'nlp'
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()

    connection.close()

    assert row is not None
    assert row[0] == text

    stored_result = json.loads(
        row[1]
    )

    assert stored_result == combined_result

    # Simulate the display stage without editing Yasser's main.py.
    displayed_result = {
        "input_text": text,
        **stored_result,
    }

    print(
        json.dumps(
            displayed_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    displayed_output = (
        capsys.readouterr().out
    )

    assert text in displayed_output

    assert (
        combined_result["result"]["intent"]["label"]
        in displayed_output
    )

    assert (
        combined_result["result"]["sentiment"]["label"]
        in displayed_output
    )