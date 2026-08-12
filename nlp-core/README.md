# NLP Core

Arabic intent classification and sentiment analysis built with TF-IDF and
scikit-learn. Run examples from the repository root.

## Usage

```python
import sys
sys.path.insert(0, "nlp-core")

from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment, batch_analyze

intent = classify_intent("كيف أرفع صورة؟")
sentiment = analyze_sentiment("الخدمة ممتازة جدًا")
sentiments = batch_analyze(["الخدمة ممتازة", "التجربة سيئة"])
```

`classify_intent` returns language metadata, a label, and confidence.
`analyze_sentiment` returns `إيجابي`, `سلبي`, or `محايد`. Pass
`save_to_db=True` to either single-text function to persist its result.

## Intent functions

| Function | Inputs | Output | Exceptions |
|---|---|---|---|
| `load_data()` | None | Text and label Series | Missing/invalid dataset: `FileNotFoundError`, `ValueError` |
| `build_naive_bayes(alpha=1.0)` | Smoothing value | Unfitted pipeline | Invalid estimator parameter errors |
| `build_logistic_regression(C=1.0)` | Positive regularization value | Unfitted pipeline | Invalid estimator parameter errors |
| `save_model(model, path)` | Fitted model, destination | Saved `Path` | Filesystem/joblib errors |
| `load_model(path)` | Model path | Cached model | `FileNotFoundError`, deserialization errors |
| `evaluate_model(model, X_test, y_test)` | Model, samples, labels | Accuracy, macro-F1, predictions, confidence | Incompatible data errors |
| `create_test_results(...)` | Evaluation sequences, model name, stage | Results DataFrame | Unequal-length `ValueError` |
| `save_test_results(results)` | List of DataFrames | `None`; writes CSVs | Filesystem errors |
| `get_top_features(model, n=10)` | Fitted pipeline, positive count | Class-to-features dictionary | `TypeError`, `ValueError`, `KeyError` |
| `create_features_table(features)` | Features dictionary | Markdown table | Invalid value errors |
| `update_evaluation_file(...)` | Metrics, parameters, features | `None`; updates evaluation document | Filesystem errors |
| `train_model()` | None | Best fitted pipeline | Data, training, or filesystem errors |
| `validate_text(text)` | String | Trimmed string | `TypeError`, `ValueError` |
| `validate_supported_language(text)` | String | Language dictionary | Non-Arabic `ValueError` |
| `classify_intent(text, save_to_db=False)` | Arabic text, save flag | Intent dictionary | `TypeError`, `ValueError`, `FileNotFoundError`, `RuntimeError` |

```python
result = classify_intent("السلام عليكم")
label = result["result"]["label"]
confidence = result["result"]["confidence"]
```

## Sentiment functions

| Function | Inputs | Output | Exceptions |
|---|---|---|---|
| `load_data()` | None | Text and label Series | Missing/invalid dataset: `FileNotFoundError`, `ValueError` |
| `build_model(C=1.0)` | Positive regularization value | Unfitted pipeline | Invalid estimator parameter errors |
| `save_model(model, path)` | Fitted model, destination | Saved `Path` | Filesystem/joblib errors |
| `load_model(path=MODEL_PATH)` | Model path | Cached model | `FileNotFoundError`, deserialization errors |
| `evaluate_model(model, X_test, y_test)` | Model, samples, labels | Accuracy, macro-F1, predictions, confidence | Incompatible data errors |
| `create_test_results(...)` | Evaluation sequences and metadata | Results DataFrame | Unequal-length `ValueError` |
| `save_test_results(results)` | List of DataFrames | `None`; writes CSVs | Filesystem errors |
| `get_top_features(model, n=10)` | Fitted pipeline, positive count | Class-to-features dictionary | `TypeError`, `ValueError`, `KeyError` |
| `create_features_table(features)` | Features dictionary | Markdown table | Invalid value errors |
| `update_evaluation_file(...)` | Metrics, parameters, features | `None`; updates evaluation document | Filesystem errors |
| `save_confusion_matrix(y_test, predictions)` | Expected and predicted labels | `None`; writes PNG | Plotting/filesystem errors |
| `train_model()` | None | Final fitted pipeline | Data, training, or filesystem errors |
| `validate_text(text)` | String | Trimmed string | `TypeError`, `ValueError` |
| `validate_supported_language(text)` | String | Language dictionary | Non-Arabic `ValueError` |
| `analyze_sentiment(text, save_to_db=False)` | Arabic text, save flag | Sentiment label | `TypeError`, `ValueError`, `FileNotFoundError`, `RuntimeError` |
| `batch_analyze(texts)` | List of Arabic strings | List of labels | `TypeError`, `ValueError`, `FileNotFoundError` |

## Train and test

```bash
python nlp-core/intent_classifier.py
python nlp-core/sentiment_analyzer.py
pytest tests/test_nlp_core.py
```

Training regenerates model files, test-result CSVs, evaluation sections, and
the sentiment confusion matrix. The bundled models can be used without
retraining.
