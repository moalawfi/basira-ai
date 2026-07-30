from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Paths and settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TESTS_DIR = PROJECT_ROOT / "tests"

DATA_PATH = BASE_DIR / "sentiment_dataset_ar.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.joblib"
CM_PATH = BASE_DIR / "sentiment_cm.png"
EVALUATION_PATH = BASE_DIR / "EVALUATION.md"

TEST_RESULTS_PATH = TESTS_DIR / "sentiment_test_results.csv"
MISCLASSIFIED_PATH = TESTS_DIR / "sentiment_misclassified.csv"

RANDOM_STATE = 42


# ============================================================
# 2. Load and validate dataset
# ============================================================

def load_data():
    """
    Load and validate the sentiment dataset.

    Supported CSV column names:
    - Arabic: النص, الفئة
    - English: text, label
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sentiment dataset was not found:\n{DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
    )

    data.columns = data.columns.str.strip()

    print("\nDataset columns:")
    print(data.columns.tolist())

    # Support either Arabic or English column headers.
    data = data.rename(
        columns={
            "النص": "text",
            "الفئة": "label",
        }
    )

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {data.columns.tolist()}"
        )

    data = data.dropna(
        subset=["text", "label"]
    ).copy()

    data["text"] = (
        data["text"]
        .astype(str)
        .str.strip()
    )

    data["label"] = (
        data["label"]
        .astype(str)
        .str.strip()
    )

    data = data[
        (data["text"] != "")
        & (data["label"] != "")
    ]

    # Remove exact duplicate examples.
    data = data.drop_duplicates(
        subset=["text", "label"]
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "The sentiment dataset is empty after cleaning."
        )

    class_counts = data["label"].value_counts()

    print("\nDataset size:", len(data))

    print("\nClass distribution:")
    print(class_counts)

    if len(class_counts) < 2:
        raise ValueError(
            "The dataset must contain at least two sentiment classes."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each sentiment class must contain at least two examples."
        )

    return data["text"], data["label"]


# ============================================================
# 3. Build sentiment model
# ============================================================

def build_model(C=1.0):
    """
    Create a TF-IDF + Logistic Regression pipeline.
    """

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# 4. Model persistence
# ============================================================

def save_model(model, path):
    """
    Save a fitted model using joblib.

    Args:
        model:
            Fitted sklearn model or pipeline.
        path:
            Destination path.

    Returns:
        pathlib.Path of the saved model.
    """

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        destination,
    )

    # Prevent an old model from remaining in the cache.
    load_model.cache_clear()

    return destination


@lru_cache(maxsize=4)
def load_model(path=MODEL_PATH):
    """
    Load a saved model using joblib.

    The loaded model is cached so it is not read from disk
    repeatedly during the same program execution.
    """

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"The trained sentiment model was not found:\n"
            f"{source}\n\n"
            "Run this command first:\n"
            "python nlp-core/sentiment_analyzer.py"
        )

    return joblib.load(source)


# ============================================================
# 5. Evaluate model
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Calculate accuracy, Macro-F1, predictions, and confidence.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        confidence = np.max(
            probabilities,
            axis=1,
        )
    else:
        confidence = np.full(
            len(predictions),
            np.nan,
        )

    return (
        accuracy,
        macro_f1,
        predictions,
        confidence,
    )


# ============================================================
# 6. Save held-out test predictions
# ============================================================

def create_test_results(
    X_test,
    y_test,
    predictions,
    confidence,
    model_name,
    stage,
):
    """
    Create a DataFrame containing every held-out test prediction.
    """

    results = pd.DataFrame(
        {
            "model": model_name,
            "stage": stage,
            "text": list(X_test),
            "actual": list(y_test),
            "predicted": list(predictions),
            "confidence": list(confidence),
        }
    )

    results["correct"] = (
        results["actual"]
        == results["predicted"]
    )

    return results


def save_test_results(all_test_results):
    """
    Save all test predictions and failed predictions under tests/.
    """

    TESTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if all_test_results:
        combined_results = pd.concat(
            all_test_results,
            ignore_index=True,
        )
    else:
        combined_results = pd.DataFrame(
            columns=[
                "model",
                "stage",
                "text",
                "actual",
                "predicted",
                "confidence",
                "correct",
            ]
        )

    combined_results.to_csv(
        TEST_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    misclassified = combined_results[
        combined_results["correct"] == False  # noqa: E712
    ].copy()

    misclassified.to_csv(
        MISCLASSIFIED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nAll sentiment test predictions saved to:")
    print(TEST_RESULTS_PATH)

    print("\nMisclassified sentiment predictions saved to:")
    print(MISCLASSIFIED_PATH)

    print("\nTotal saved predictions:", len(combined_results))
    print("Total failed predictions:", len(misclassified))


# ============================================================
# 7. Get top features
# ============================================================

def get_top_features(model, n=10):
    """
    Return the top n TF-IDF features for every sentiment class.
    """

    if n <= 0:
        raise ValueError(
            "n must be greater than zero."
        )

    if not isinstance(model, Pipeline):
        raise TypeError(
            "model must be a fitted sklearn Pipeline."
        )

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    feature_names = np.asarray(
        vectorizer.get_feature_names_out()
    )

    class_names = [
        str(class_name)
        for class_name in classifier.classes_
    ]

    coefficients = classifier.coef_

    top_features = {}

    # Binary Logistic Regression contains one coefficient row.
    if (
        len(class_names) == 2
        and coefficients.shape[0] == 1
    ):
        weights_by_class = {
            class_names[0]: -coefficients[0],
            class_names[1]: coefficients[0],
        }
    else:
        weights_by_class = {
            class_name: coefficients[index]
            for index, class_name in enumerate(class_names)
        }

    for class_name, weights in weights_by_class.items():
        top_indices = np.argsort(
            weights
        )[-n:][::-1]

        top_features[class_name] = (
            feature_names[top_indices].tolist()
        )

    return top_features


# ============================================================
# 8. Convert top features to Markdown
# ============================================================

def create_features_table(top_features):
    """
    Convert the top-features dictionary into a Markdown table.
    """

    rows = [
        "| Sentiment class | Top 10 features |",
        "|---|---|",
    ]

    for class_name, features in top_features.items():
        safe_class_name = class_name.replace(
            "|",
            "\\|",
        )

        features_text = ", ".join(
            feature.replace("|", "\\|")
            for feature in features
        )

        rows.append(
            f"| {safe_class_name} | {features_text} |"
        )

    return "\n".join(rows)


# ============================================================
# 9. Update EVALUATION.md
# ============================================================

def update_evaluation_file(
    baseline_accuracy,
    baseline_macro_f1,
    tuned_accuracy,
    tuned_macro_f1,
    best_parameters,
    best_cv_accuracy,
    top_features,
):
    """
    Create or replace the generated Sentiment section.
    """

    start_marker = "<!-- SENTIMENT_EVALUATION_START -->"
    end_marker = "<!-- SENTIMENT_EVALUATION_END -->"

    parameters_text = ", ".join(
        f"`{name}` = `{value}`"
        for name, value in best_parameters.items()
    )

    features_table = create_features_table(
        top_features
    )

    sentiment_section = f"""
{start_marker}

## Sentiment Analysis Evaluation

The baseline and tuned sentiment models were evaluated using
the same held-out test set. `GridSearchCV` was performed only
on the training set to prevent test-data leakage.

### Results Before and After Tuning

| Model | Stage | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Logistic Regression | Before tuning | {baseline_accuracy:.4f} | {baseline_macro_f1:.4f} |
| Logistic Regression | After tuning | {tuned_accuracy:.4f} | {tuned_macro_f1:.4f} |

### Hyperparameter Tuning

**Method:** GridSearchCV

**Scoring metric:** Accuracy

**Best cross-validation accuracy:** {best_cv_accuracy:.4f}

**Best hyperparameters:** {parameters_text}

### Saved Test Files

- All predictions: `tests/sentiment_test_results.csv`
- Failed predictions: `tests/sentiment_misclassified.csv`

### Top 10 Features per Sentiment Class

{features_table}

### Generated Files

- `sentiment_model.joblib`
- `sentiment_cm.png`
- `tests/sentiment_test_results.csv`
- `tests/sentiment_misclassified.csv`

{end_marker}
""".strip()

    if not EVALUATION_PATH.exists():
        initial_content = f"""# NLP Model Evaluation

This file is generated automatically by the training scripts.

{sentiment_section}
"""

        EVALUATION_PATH.write_text(
            initial_content,
            encoding="utf-8",
        )

        return

    current_content = EVALUATION_PATH.read_text(
        encoding="utf-8"
    )

    if (
        start_marker in current_content
        and end_marker in current_content
    ):
        before_section = current_content.split(
            start_marker,
            1,
        )[0].rstrip()

        after_section = current_content.split(
            end_marker,
            1,
        )[1].lstrip()

        updated_content = (
            f"{before_section}\n\n"
            f"{sentiment_section}"
        )

        if after_section:
            updated_content += (
                f"\n\n{after_section}"
            )
    else:
        updated_content = (
            current_content.rstrip()
            + "\n\n"
            + sentiment_section
            + "\n"
        )

    EVALUATION_PATH.write_text(
        updated_content.rstrip() + "\n",
        encoding="utf-8",
    )


# ============================================================
# 10. Save confusion matrix
# ============================================================

def save_confusion_matrix(y_test, predictions):
    """
    Create and save the tuned sentiment confusion matrix.
    """

    class_names = sorted(
        pd.Series(y_test).unique()
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=class_names,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(
        "Sentiment Analysis Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    fig.tight_layout()

    fig.savefig(
        CM_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nConfusion matrix saved to:")
    print(CM_PATH)


# ============================================================
# 11. Train and tune model
# ============================================================

def train_model():
    """
    Train the baseline model, tune it, save held-out results,
    save the confusion matrix, retrain on all data, and persist
    the final model.
    """

    print("=" * 60)
    print("SENTIMENT ANALYZER TRAINING")
    print("=" * 60)

    texts, labels = load_data()

    # Use the same held-out split before and after tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    smallest_training_class = int(
        y_train.value_counts().min()
    )

    cv_splits = min(
        5,
        smallest_training_class,
    )

    if cv_splits < 2:
        raise ValueError(
            "There are not enough training examples per class "
            "for GridSearchCV."
        )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_test_results = []

    # --------------------------------------------------------
    # Baseline model
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE MODEL — BEFORE TUNING")
    print("=" * 60)

    baseline_model = build_model(
        C=1.0
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    (
        baseline_accuracy,
        baseline_macro_f1,
        baseline_predictions,
        baseline_confidence,
    ) = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    all_test_results.append(
        create_test_results(
            X_test=X_test,
            y_test=y_test,
            predictions=baseline_predictions,
            confidence=baseline_confidence,
            model_name="Logistic Regression",
            stage="Before tuning",
        )
    )

    print(
        f"\nBaseline Accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Baseline Macro-F1: "
        f"{baseline_macro_f1:.4f}"
    )

    print("\nBaseline Classification Report:")

    print(
        classification_report(
            y_test,
            baseline_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # GridSearchCV
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRIDSEARCHCV HYPERPARAMETER TUNING")
    print("=" * 60)

    parameter_grid = {
        "classifier__C": [
            0.01,
            0.1,
            0.5,
            1.0,
            2.0,
            10.0,
            100.0,
        ]
    }

    grid_search = GridSearchCV(
        estimator=build_model(),
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    tuned_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        "\nBest cross-validation accuracy:",
        f"{grid_search.best_score_:.4f}",
    )

    # --------------------------------------------------------
    # Evaluate tuned model
    # --------------------------------------------------------

    (
        tuned_accuracy,
        tuned_macro_f1,
        tuned_predictions,
        tuned_confidence,
    ) = evaluate_model(
        tuned_model,
        X_test,
        y_test,
    )

    all_test_results.append(
        create_test_results(
            X_test=X_test,
            y_test=y_test,
            predictions=tuned_predictions,
            confidence=tuned_confidence,
            model_name="Logistic Regression",
            stage="After tuning",
        )
    )

    print("\n" + "=" * 60)
    print("TUNED MODEL — AFTER GRIDSEARCHCV")
    print("=" * 60)

    print(
        f"\nTuned Accuracy: "
        f"{tuned_accuracy:.4f}"
    )

    print(
        f"Tuned Macro-F1: "
        f"{tuned_macro_f1:.4f}"
    )

    print("\nTuned Classification Report:")

    print(
        classification_report(
            y_test,
            tuned_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BEFORE AND AFTER COMPARISON")
    print("=" * 60)

    print(
        f"\nAccuracy before tuning: "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Accuracy after tuning:  "
        f"{tuned_accuracy:.4f}"
    )

    accuracy_change = (
        tuned_accuracy
        - baseline_accuracy
    )

    print(
        f"Accuracy change:        "
        f"{accuracy_change:+.4f}"
    )

    print(
        f"\nMacro-F1 before tuning: "
        f"{baseline_macro_f1:.4f}"
    )

    print(
        f"Macro-F1 after tuning:  "
        f"{tuned_macro_f1:.4f}"
    )

    macro_f1_change = (
        tuned_macro_f1
        - baseline_macro_f1
    )

    print(
        f"Macro-F1 change:        "
        f"{macro_f1_change:+.4f}"
    )

    # Save every held-out prediction and the failed subset.
    save_test_results(
        all_test_results
    )

    # Save the tuned model's confusion matrix.
    save_confusion_matrix(
        y_test,
        tuned_predictions,
    )

    # --------------------------------------------------------
    # Train final model on all data
    # --------------------------------------------------------

    print(
        "\nTraining final model on the full dataset..."
    )

    final_model = grid_search.best_estimator_

    final_model.fit(
        texts,
        labels,
    )

    save_model(
        final_model,
        MODEL_PATH,
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Top features
    # --------------------------------------------------------

    top_features = get_top_features(
        final_model,
        n=10,
    )

    print("\n" + "=" * 60)
    print("TOP 10 FEATURES PER SENTIMENT CLASS")
    print("=" * 60)

    for class_name, features in top_features.items():
        print(f"\n{class_name}:")
        print(", ".join(features))

    # --------------------------------------------------------
    # EVALUATION.md
    # --------------------------------------------------------

    update_evaluation_file(
        baseline_accuracy=baseline_accuracy,
        baseline_macro_f1=baseline_macro_f1,
        tuned_accuracy=tuned_accuracy,
        tuned_macro_f1=tuned_macro_f1,
        best_parameters=grid_search.best_params_,
        best_cv_accuracy=grid_search.best_score_,
        top_features=top_features,
    )

    print("\nEvaluation saved to:")
    print(EVALUATION_PATH)

    print("\n" + "=" * 60)
    print("SENTIMENT TRAINING COMPLETED")
    print("=" * 60)

    return final_model


# ============================================================
# 12. Validate input text
# ============================================================

def validate_text(text):
    """
    Validate and clean one input text.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError(
            "text cannot be empty"
        )

    return cleaned_text


# ============================================================
# 13. Analyze one text
# ============================================================

def analyze_sentiment(
    text,
    save_to_db=False,
):
    """
    Analyze one text and optionally save the prediction to SQLite.

    The return value remains a string to preserve compatibility
    with the existing tests and project interface.
    """

    cleaned_text = validate_text(text)

    model = load_model(
        MODEL_PATH
    )

    prediction = str(
        model.predict(
            [cleaned_text]
        )[0]
    )

    if save_to_db:
        from storage.results_store import save_nlp_result

        result = {
            "type": "sentiment",
            "result": {
                "label": prediction,
            },
        }

        saved = save_nlp_result(
            cleaned_text,
            result,
        )

        if not saved:
            raise RuntimeError(
                "The sentiment prediction was created, "
                "but saving it to the database failed."
            )

    return prediction


# ============================================================
# 14. Analyze multiple texts
# ============================================================

def batch_analyze(texts):
    """
    Analyze multiple input texts.
    """

    if not isinstance(texts, list):
        raise TypeError(
            "texts must be a list"
        )

    if len(texts) == 0:
        return []

    cleaned_texts = [
        validate_text(text)
        for text in texts
    ]

    model = load_model(
        MODEL_PATH
    )

    predictions = model.predict(
        cleaned_texts
    )

    return [
        str(prediction)
        for prediction in predictions
    ]


# ============================================================
# 15. Main
# ============================================================

if __name__ == "__main__":

    train_model()

    print("\n" + "=" * 60)
    print("SINGLE TEXT TEST")
    print("=" * 60)

    test_text = "الخدمة ممتازة جدا"

    result = analyze_sentiment(
        test_text
    )

    print(f"\nText: {test_text}")
    print(f"Prediction: {result}")

    print("\n" + "=" * 60)
    print("BATCH TEST")
    print("=" * 60)

    test_texts = [
        "الخدمة ممتازة جدا",
        "التجربة كانت سيئة",
        "وصل الطلب اليوم",
    ]

    results = batch_analyze(
        test_texts
    )

    for text, sentiment in zip(
        test_texts,
        results,
    ):
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment}")