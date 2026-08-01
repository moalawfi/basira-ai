


from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ============================================================
# Paths and settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TESTS_DIR = PROJECT_ROOT / "tests"

DATA_PATH = BASE_DIR / "intent_data.csv"
MODEL_PATH = BASE_DIR / "intent_model.joblib"
EVALUATION_PATH = BASE_DIR / "EVALUATION.md"

TEST_RESULTS_PATH = TESTS_DIR / "intent_test_results_after_expansion.csv"
MISCLASSIFIED_PATH = TESTS_DIR / "intent_misclassified_after_expansion.csv"
TEST_SUMMARY_PATH = TESTS_DIR / "intent_test_summary_after_expansion.csv"

RANDOM_STATE = 42


# ============================================================
# Dataset
# ============================================================

def load_data():
    """Load and validate intent_data.csv."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset was not found:\n{DATA_PATH}")

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data.columns = data.columns.str.strip()

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {data.columns.tolist()}"
        )

    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()

    data = data[
        (data["text"] != "") & (data["label"] != "")
    ].drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("The intent dataset is empty after cleaning.")

    class_counts = data["label"].value_counts()

    if len(class_counts) < 2:
        raise ValueError("The dataset must contain at least two classes.")

    if class_counts.min() < 2:
        raise ValueError("Each class must contain at least two examples.")

    print("\nDataset size:", len(data))
    print("\nClass distribution:")
    print(class_counts)

    return data["text"], data["label"]


# ============================================================
# Model builders
# ============================================================

def build_naive_bayes(alpha=1.0):
    """Create a TF-IDF + Multinomial Naive Bayes pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                MultinomialNB(alpha=alpha),
            ),
        ]
    )


def build_logistic_regression(C=1.0):
    """Create a TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
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
# Model persistence
# ============================================================

def save_model(model, path):
    """Save a fitted model using joblib."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)

    # Prevent using an old cached model after retraining.
    load_model.cache_clear()

    return destination


@lru_cache(maxsize=4)
def load_model(path):
    """Load a saved model using joblib."""
    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"The trained model was not found:\n{source}\n\n"
            "Run this command first:\n"
            "python nlp-core/intent_classifier.py"
        )

    return joblib.load(source)


# ============================================================
# Evaluation and saved test results
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Return accuracy, Macro-F1, predictions, and confidence."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        confidence = np.max(probabilities, axis=1)
    else:
        confidence = np.full(len(predictions), np.nan)

    return accuracy, macro_f1, predictions, confidence


def create_test_results(
    X_test,
    y_test,
    predictions,
    confidence,
    model_name,
    stage,
):
    """Create a table containing every test-set prediction."""
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
        results["actual"] == results["predicted"]
    )

    return results


def save_test_results(all_test_results):
    """
    Save detailed predictions, misclassified samples, and a compact
    model/stage summary in tests/ without overwriting the older CSV files.
    """
    TESTS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Full held-out predictions after dataset expansion.
    combined_results.to_csv(
        TEST_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Incorrect predictions only, for future error analysis.
    misclassified = combined_results[
        combined_results["correct"] == False  # noqa: E712
    ].copy()

    misclassified.to_csv(
        MISCLASSIFIED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # One summary row per model and evaluation stage.
    if combined_results.empty:
        summary = pd.DataFrame(
            columns=[
                "model",
                "stage",
                "tested",
                "correct",
                "incorrect",
                "accuracy",
            ]
        )
    else:
        summary = (
            combined_results
            .groupby(["model", "stage"], as_index=False)
            .agg(
                tested=("correct", "size"),
                correct=("correct", "sum"),
            )
        )
        summary["correct"] = summary["correct"].astype(int)
        summary["incorrect"] = (
            summary["tested"] - summary["correct"]
        )
        summary["accuracy"] = (
            summary["correct"] / summary["tested"]
        )
        summary = summary[
            [
                "model",
                "stage",
                "tested",
                "correct",
                "incorrect",
                "accuracy",
            ]
        ]

    summary.to_csv(
        TEST_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nExpanded-data test predictions saved to:")
    print(TEST_RESULTS_PATH)

    print("\nExpanded-data misclassified predictions saved to:")
    print(MISCLASSIFIED_PATH)

    print("\nExpanded-data test summary saved to:")
    print(TEST_SUMMARY_PATH)

    print("\nTotal saved predictions:", len(combined_results))
    print("Total failed predictions:", len(misclassified))

    if not summary.empty:
        print("\nTest summary after dataset expansion:")
        print(summary.to_string(index=False))


# ============================================================
# Top features
# ============================================================

def get_top_features(model, n=10):
    """Return the top n TF-IDF features for each class."""
    if n <= 0:
        raise ValueError("n must be greater than zero.")

    if not isinstance(model, Pipeline):
        raise TypeError("model must be a fitted sklearn Pipeline.")

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    feature_names = np.asarray(
        vectorizer.get_feature_names_out()
    )
    class_names = [
        str(class_name)
        for class_name in classifier.classes_
    ]

    top_features = {}

    if isinstance(classifier, LogisticRegression):
        coefficients = classifier.coef_

        if len(class_names) == 2 and coefficients.shape[0] == 1:
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
            top_indices = np.argsort(weights)[-n:][::-1]
            top_features[class_name] = (
                feature_names[top_indices].tolist()
            )

    elif isinstance(classifier, MultinomialNB):
        for index, class_name in enumerate(class_names):
            weights = classifier.feature_log_prob_[index]
            top_indices = np.argsort(weights)[-n:][::-1]
            top_features[class_name] = (
                feature_names[top_indices].tolist()
            )
    else:
        raise TypeError(
            "Only LogisticRegression and MultinomialNB are supported."
        )

    return top_features


def create_features_table(top_features):
    """Convert top features to a Markdown table."""
    rows = [
        "| Intent class | Top 10 features |",
        "|---|---|",
    ]

    for class_name, features in top_features.items():
        features_text = ", ".join(
            feature.replace("|", "\\|")
            for feature in features
        )
        rows.append(f"| {class_name} | {features_text} |")

    return "\n".join(rows)


# ============================================================
# EVALUATION.md
# ============================================================

def update_evaluation_file(
    baseline_results,
    tuned_results,
    best_model_name,
    best_parameters,
    top_features,
):
    """Create or replace the Intent section in EVALUATION.md."""
    start_marker = "<!-- INTENT_EVALUATION_START -->"
    end_marker = "<!-- INTENT_EVALUATION_END -->"

    table_rows = [
        "| Model | Stage | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|",
    ]

    for result in baseline_results + tuned_results:
        table_rows.append(
            f"| {result['model']} "
            f"| {result['stage']} "
            f"| {result['accuracy']:.4f} "
            f"| {result['macro_f1']:.4f} |"
        )

    results_table = "\n".join(table_rows)

    parameters_text = ", ".join(
        f"`{name}` = `{value}`"
        for name, value in best_parameters.items()
    )

    features_table = create_features_table(top_features)

    intent_section = f"""
{start_marker}

## Intent Classification Evaluation

The models were evaluated using the same held-out test split.
Hyperparameter tuning was performed only on the training data
using `GridSearchCV`.

### Results

{results_table}

### Selected Model

**Best model:** {best_model_name}

**Best hyperparameters:** {parameters_text}

### Saved Test Files

- All predictions: `tests/intent_test_results_after_expansion.csv`
- Failed predictions: `tests/intent_misclassified_after_expansion.csv`
- Test summary: `tests/intent_test_summary_after_expansion.csv`

### Top 10 Features per Intent Class

{features_table}

{end_marker}
""".strip()

    if not EVALUATION_PATH.exists():
        EVALUATION_PATH.write_text(
            "# NLP Model Evaluation\n\n"
            "This file is generated automatically by the training scripts.\n\n"
            f"{intent_section}\n",
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
            f"{intent_section}"
        )

        if after_section:
            updated_content += f"\n\n{after_section}"
    else:
        updated_content = (
            current_content.rstrip()
            + "\n\n"
            + intent_section
        )

    EVALUATION_PATH.write_text(
        updated_content.rstrip() + "\n",
        encoding="utf-8",
    )


# ============================================================
# Training
# ============================================================

def train_model():
    """
    Train, tune, evaluate, save test results, and save the
    selected model.
    """
    print("=" * 60)
    print("INTENT CLASSIFIER TRAINING")
    print("=" * 60)

    texts, labels = load_data()

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
    cv_splits = min(5, smallest_training_class)

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

    baseline_models = {
        "Naive Bayes": build_naive_bayes(alpha=1.0),
        "Logistic Regression": build_logistic_regression(C=1.0),
    }

    baseline_results = []
    tuned_results = []
    tuned_candidates = []
    all_test_results = []

    print("\n" + "=" * 60)
    print("BASELINE RESULTS — BEFORE TUNING")
    print("=" * 60)

    for model_name, model in baseline_models.items():
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        (
            accuracy,
            macro_f1,
            predictions,
            confidence,
        ) = evaluate_model(
            fitted_model,
            X_test,
            y_test,
        )

        baseline_results.append(
            {
                "model": model_name,
                "stage": "Before tuning",
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

        all_test_results.append(
            create_test_results(
                X_test=X_test,
                y_test=y_test,
                predictions=predictions,
                confidence=confidence,
                model_name=model_name,
                stage="Before tuning",
            )
        )

        print(f"\n{model_name}")
        print("-" * 40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

    grid_searches = {
        "Naive Bayes": GridSearchCV(
            estimator=build_naive_bayes(),
            param_grid={
                "classifier__alpha": [
                    0.01,
                    0.1,
                    0.5,
                    1.0,
                    2.0,
                ]
            },
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "Logistic Regression": GridSearchCV(
            estimator=build_logistic_regression(),
            param_grid={
                "classifier__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                ]
            },
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
    }

    print("\n" + "=" * 60)
    print("TUNED RESULTS — AFTER GRIDSEARCHCV")
    print("=" * 60)

    for model_name, grid_search in grid_searches.items():
        grid_search.fit(X_train, y_train)
        tuned_model = grid_search.best_estimator_

        (
            accuracy,
            macro_f1,
            predictions,
            confidence,
        ) = evaluate_model(
            tuned_model,
            X_test,
            y_test,
        )

        tuned_results.append(
            {
                "model": model_name,
                "stage": "After tuning",
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

        tuned_candidates.append(
            {
                "model_name": model_name,
                "model": tuned_model,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "parameters": grid_search.best_params_,
            }
        )

        all_test_results.append(
            create_test_results(
                X_test=X_test,
                y_test=y_test,
                predictions=predictions,
                confidence=confidence,
                model_name=model_name,
                stage="After tuning",
            )
        )

        print(f"\n{model_name}")
        print("-" * 40)
        print("Best parameters:", grid_search.best_params_)
        print(
            "Best cross-validation accuracy:",
            f"{grid_search.best_score_:.4f}",
        )
        print("Test accuracy:", f"{accuracy:.4f}")
        print("Test Macro-F1:", f"{macro_f1:.4f}")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

    # Save all held-out test predictions before retraining on all data.
    save_test_results(all_test_results)

    best_candidate = max(
        tuned_candidates,
        key=lambda candidate: (
            candidate["accuracy"],
            candidate["macro_f1"],
        ),
    )

    best_model_name = best_candidate["model_name"]
    best_parameters = best_candidate["parameters"]
    best_model = best_candidate["model"]

    print("\n" + "=" * 60)
    print("SELECTED MODEL")
    print("=" * 60)
    print("Best model:", best_model_name)
    print("Best parameters:", best_parameters)
    print(
        "Test accuracy:",
        f"{best_candidate['accuracy']:.4f}",
    )
    print(
        "Test Macro-F1:",
        f"{best_candidate['macro_f1']:.4f}",
    )

    # Retrain the selected configuration on the complete dataset.
    best_model.fit(texts, labels)

    save_model(best_model, MODEL_PATH)

    top_features = get_top_features(best_model, n=10)

    print("\nTop 10 features per intent class:")
    for class_name, features in top_features.items():
        print(f"\n{class_name}:")
        print(", ".join(features))

    update_evaluation_file(
        baseline_results=baseline_results,
        tuned_results=tuned_results,
        best_model_name=best_model_name,
        best_parameters=best_parameters,
        top_features=top_features,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nEvaluation saved to:")
    print(EVALUATION_PATH)

    return best_model


# ============================================================
# Prediction
# ============================================================

def validate_text(text):
    """Validate and clean one input text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("text cannot be empty")

    return cleaned_text


def classify_intent(text, save_to_db=True):
    """
    Classify one text and optionally save the result to SQLite.
    """
    cleaned_text = validate_text(text)
    model = load_model(MODEL_PATH)

    prediction = model.predict([cleaned_text])[0]

    if hasattr(model, "predict_proba"):
        confidence = float(
            np.max(model.predict_proba([cleaned_text])[0])
        )
    else:
        confidence = None

    result = {
        "type": "intent",
        "result": {
            "label": str(prediction),
            "confidence": confidence,
        },
    }

    if save_to_db:
        from storage.results_store import save_nlp_result

        saved = save_nlp_result(
            cleaned_text,
            result,
        )

        if not saved:
            raise RuntimeError(
                "The prediction was created, but saving it "
                "to the database failed."
            )

    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_model()

    test_examples = [
        "السلام عليكم",
        "كيف أرفع صورة؟",
        "حلل هذا النص",
        "التطبيق بطيء جدًا",
    ]

    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)

    for example in test_examples:
        result = classify_intent(example)
        print(f"\nText: {example}")
        print(f"Prediction: {result}")