from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Paths and settings
# ============================================================

# intent_classifier.py is already inside nlp-core,
# so BASE_DIR should point to the same folder.
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "intent_data.csv"
MODEL_PATH = BASE_DIR / "intent_model.joblib"
EVALUATION_PATH = BASE_DIR / "EVALUATION.md"

RANDOM_STATE = 42


# ============================================================
# 2. Load and validate dataset
# ============================================================

def load_data():
    """
    Load the intent dataset.

    Required CSV columns:
    - text
    - label
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found:\n{DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
    )

    # Remove accidental spaces from column names
    data.columns = data.columns.str.strip()

    required_columns = {"text", "label"}

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {data.columns.tolist()}"
        )

    # Remove missing values
    data = data.dropna(
        subset=["text", "label"]
    ).copy()

    # Clean text and labels
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

    # Remove empty rows
    data = data[
        (data["text"] != "")
        & (data["label"] != "")
    ]

    if data.empty:
        raise ValueError(
            "The intent dataset is empty after cleaning."
        )

    class_counts = data["label"].value_counts()

    print("\nDataset size:", len(data))

    print("\nClass distribution:")
    print(class_counts)

    if len(class_counts) < 2:
        raise ValueError(
            "The dataset must contain at least two intent classes."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each class must contain at least two examples."
        )

    return data["text"], data["label"]


# ============================================================
# 3. Build Naive Bayes model
# ============================================================

def build_naive_bayes(alpha=1.0):
    """
    Create a TF-IDF + Multinomial Naive Bayes pipeline.
    """

    model = Pipeline(
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
                MultinomialNB(
                    alpha=alpha,
                ),
            ),
        ]
    )

    return model


# ============================================================
# 4. Build Logistic Regression model
# ============================================================

def build_logistic_regression(C=1.0):
    """
    Create a TF-IDF + Logistic Regression pipeline.
    """

    model = Pipeline(
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

    return model


# ============================================================
# 5. Evaluate one model
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Calculate accuracy, Macro-F1, and predictions.
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

    return accuracy, macro_f1, predictions


# ============================================================
# 6. Get top features
# ============================================================

def get_top_features(model, n=10):
    """
    Return the top n features for every intent class.

    The model must be a fitted Pipeline containing:
    - tfidf
    - classifier
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

    top_features = {}

    # Logistic Regression feature weights
    if isinstance(classifier, LogisticRegression):

        coefficients = classifier.coef_

        # Binary Logistic Regression has only one coefficient row
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

    # Naive Bayes feature weights
    elif isinstance(classifier, MultinomialNB):

        for index, class_name in enumerate(class_names):

            weights = classifier.feature_log_prob_[index]

            top_indices = np.argsort(weights)[-n:][::-1]

            top_features[class_name] = (
                feature_names[top_indices].tolist()
            )

    else:
        raise TypeError(
            "Only LogisticRegression and MultinomialNB "
            "are supported."
        )

    return top_features


# ============================================================
# 7. Create Markdown feature table
# ============================================================

def create_features_table(top_features):
    """
    Convert top features into a Markdown table.
    """

    rows = [
        "| Intent class | Top 10 features |",
        "|---|---|",
    ]

    for class_name, features in top_features.items():

        features_text = ", ".join(
            feature.replace("|", "\\|")
            for feature in features
        )

        rows.append(
            f"| {class_name} | {features_text} |"
        )

    return "\n".join(rows)


# ============================================================
# 8. Update EVALUATION.md
# ============================================================

def update_evaluation_file(
    baseline_results,
    tuned_results,
    best_model_name,
    best_parameters,
    top_features,
):
    """
    Create EVALUATION.md automatically.

    If the file already exists, only the generated Intent
    section is replaced.
    """

    start_marker = "<!-- INTENT_EVALUATION_START -->"
    end_marker = "<!-- INTENT_EVALUATION_END -->"

    table_rows = [
        "| Model | Stage | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|",
    ]

    all_results = baseline_results + tuned_results

    for result in all_results:

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

    features_table = create_features_table(
        top_features
    )

    intent_section = f"""
{start_marker}

## Intent Classification Evaluation

The baseline and tuned models were evaluated using the same held-out test set. Hyperparameter tuning was performed only on the training data using `GridSearchCV`.

### Results Before and After Tuning

{results_table}

### Selected Model

**Best model:** {best_model_name}

**Best hyperparameters:** {parameters_text}

### Top 10 Features per Intent Class

{features_table}

{end_marker}
""".strip()

    # Create the file automatically if it does not exist
    if not EVALUATION_PATH.exists():

        initial_content = f"""# NLP Model Evaluation

This file is generated automatically by the training scripts.

{intent_section}
"""

        EVALUATION_PATH.write_text(
            initial_content,
            encoding="utf-8",
        )

    else:

        current_content = EVALUATION_PATH.read_text(
            encoding="utf-8"
        )

        # Replace the previous Intent section
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
                updated_content += (
                    f"\n\n{after_section}"
                )

        # Add the Intent section if it is not present
        else:

            updated_content = (
                current_content.rstrip()
                + "\n\n"
                + intent_section
                + "\n"
            )

        EVALUATION_PATH.write_text(
            updated_content.rstrip() + "\n",
            encoding="utf-8",
        )


# ============================================================
# 9. Train, tune, and select best model
# ============================================================

def train_model():
    """
    Train baseline models, apply GridSearchCV,
    compare results, and save the best model.
    """

    print("=" * 60)
    print("INTENT CLASSIFIER TRAINING")
    print("=" * 60)

    texts, labels = load_data()

    # Same test set for all model comparisons
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Determine a safe number of CV folds
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

    # --------------------------------------------------------
    # Baseline models
    # --------------------------------------------------------

    baseline_models = {
        "Naive Bayes": build_naive_bayes(
            alpha=1.0
        ),
        "Logistic Regression": build_logistic_regression(
            C=1.0
        ),
    }

    baseline_results = []

    print("\n" + "=" * 60)
    print("BASELINE RESULTS — BEFORE TUNING")
    print("=" * 60)

    for model_name, model in baseline_models.items():

        fitted_model = clone(model)

        fitted_model.fit(
            X_train,
            y_train,
        )

        accuracy, macro_f1, predictions = evaluate_model(
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

    # --------------------------------------------------------
    # GridSearchCV for Naive Bayes
    # --------------------------------------------------------

    nb_grid_search = GridSearchCV(
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
    )

    # --------------------------------------------------------
    # GridSearchCV for Logistic Regression
    # --------------------------------------------------------

    lr_grid_search = GridSearchCV(
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
    )

    grid_searches = {
        "Naive Bayes": nb_grid_search,
        "Logistic Regression": lr_grid_search,
    }

    tuned_results = []
    tuned_candidates = []

    print("\n" + "=" * 60)
    print("TUNED RESULTS — AFTER GRIDSEARCHCV")
    print("=" * 60)

    for model_name, grid_search in grid_searches.items():

        grid_search.fit(
            X_train,
            y_train,
        )

        tuned_model = grid_search.best_estimator_

        accuracy, macro_f1, predictions = evaluate_model(
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

        print(f"\n{model_name}")
        print("-" * 40)

        print(
            "Best parameters:",
            grid_search.best_params_,
        )

        print(
            "Best cross-validation accuracy:",
            f"{grid_search.best_score_:.4f}",
        )

        print(
            "Test accuracy:",
            f"{accuracy:.4f}",
        )

        print(
            "Test Macro-F1:",
            f"{macro_f1:.4f}",
        )

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

    # --------------------------------------------------------
    # Select best tuned model
    # --------------------------------------------------------

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

    # Train the selected model on the full dataset
    best_model.fit(
        texts,
        labels,
    )

    # Save trained model
    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    # Clear cached model after retraining
    load_model.cache_clear()

    # Extract top 10 features per class
    top_features = get_top_features(
        best_model,
        n=10,
    )

    print("\nTop 10 features per intent class:")

    for class_name, features in top_features.items():
        print(f"\n{class_name}:")
        print(", ".join(features))

    # Create or update EVALUATION.md
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
# 10. Load saved model
# ============================================================

@lru_cache(maxsize=1)
def load_model():
    """
    Load the saved intent model only once.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained intent model was not found.\n"
            "Run this command first:\n\n"
            "python nlp-core/intent_classifier.py"
        )

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# 11. Validate prediction input
# ============================================================

def validate_text(text):
    """
    Validate one input text.
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
# 12. Classify one intent
# ============================================================

def classify_intent(text):
    """
    Classify one text and return a JSON-compatible dictionary.
    """

    cleaned_text = validate_text(text)

    model = load_model()

    prediction = model.predict(
        [cleaned_text]
    )[0]

    return {
        "type": "intent",
        "result": {
            "label": str(prediction),
        },
    }


# ============================================================
# 13. Main
# ============================================================

if __name__ == "__main__":

    # Train, tune, and save the model
    train_model()

    # Test predictions
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