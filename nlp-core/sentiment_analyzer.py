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

DATA_PATH = BASE_DIR / "sentiment_dataset_ar.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.joblib"
CM_PATH = BASE_DIR / "sentiment_cm.png"
EVALUATION_PATH = BASE_DIR / "EVALUATION.md"

RANDOM_STATE = 42


# ============================================================
# 2. Load and validate dataset
# ============================================================

def load_data():
    """
    Load and validate the Arabic sentiment dataset.

    Required CSV columns:
    - النص
    - الفئة
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sentiment dataset was not found:\n{DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
    )

    # Remove accidental spaces from column names
    data.columns = data.columns.str.strip()

    print("\nDataset columns:")
    print(data.columns.tolist())

    # Convert Arabic column names to internal English names
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

    # Remove rows with missing values
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

    Word unigrams and bigrams are used so that the model's
    top features are understandable.
    """

    model = Pipeline(
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

    return model


# ============================================================
# 4. Evaluate model
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
# 5. Get top features
# ============================================================

def get_top_features(model, n=10):
    """
    Return the top n TF-IDF features for every sentiment class.

    Args:
        model:
            A fitted sklearn Pipeline containing:
            - tfidf
            - classifier

        n:
            Number of top features for each class.

    Returns:
        Dictionary containing the top features for each class.
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

    # Binary Logistic Regression contains one coefficient row
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

    return top_features


# ============================================================
# 6. Convert top features to Markdown
# ============================================================

def create_features_table(top_features):
    """
    Convert the top features dictionary into a Markdown table.
    """

    rows = [
        "| Sentiment class | Top 10 features |",
        "|---|---|",
    ]

    for class_name, features in top_features.items():

        safe_class_name = class_name.replace("|", "\\|")

        features_text = ", ".join(
            feature.replace("|", "\\|")
            for feature in features
        )

        rows.append(
            f"| {safe_class_name} | {features_text} |"
        )

    return "\n".join(rows)


# ============================================================
# 7. Update EVALUATION.md
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
    Create or update EVALUATION.md automatically.

    This function only replaces the generated Sentiment section.
    The Intent section remains unchanged.
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

The baseline and tuned sentiment models were evaluated using the same held-out test set. `GridSearchCV` was performed only on the training set to prevent test-data leakage.

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

### Top 10 Features per Sentiment Class

{features_table}

### Generated Files

- `sentiment_model.joblib`
- `sentiment_cm.png`

{end_marker}
""".strip()

    # Create EVALUATION.md if it does not exist
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

    # Replace an existing Sentiment section
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

    # Add the Sentiment section if it does not exist
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
# 8. Save confusion matrix
# ============================================================

def save_confusion_matrix(y_test, predictions):
    """
    Create and save the sentiment confusion matrix.
    """

    class_names = sorted(
        y_test.unique()
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
# 9. Train and tune model
# ============================================================

def train_model():
    """
    Train the baseline model, apply GridSearchCV,
    compare the results, save the confusion matrix,
    save the final model, and update EVALUATION.md.
    """

    print("=" * 60)
    print("SENTIMENT ANALYZER TRAINING")
    print("=" * 60)

    texts, labels = load_data()

    # Use the same test set before and after tuning
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Determine a safe number of cross-validation folds
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
    # Baseline model before tuning
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
    ) = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    print(
        f"\nBaseline Accuracy: {baseline_accuracy:.4f}"
    )

    print(
        f"Baseline Macro-F1: {baseline_macro_f1:.4f}"
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
    # Evaluate tuned model on the same test set
    # --------------------------------------------------------

    (
        tuned_accuracy,
        tuned_macro_f1,
        tuned_predictions,
    ) = evaluate_model(
        tuned_model,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("TUNED MODEL — AFTER GRIDSEARCHCV")
    print("=" * 60)

    print(
        f"\nTuned Accuracy: {tuned_accuracy:.4f}"
    )

    print(
        f"Tuned Macro-F1: {tuned_macro_f1:.4f}"
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
        tuned_accuracy - baseline_accuracy
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
        tuned_macro_f1 - baseline_macro_f1
    )

    print(
        f"Macro-F1 change:        "
        f"{macro_f1_change:+.4f}"
    )

    # --------------------------------------------------------
    # Save confusion matrix using tuned predictions
    # --------------------------------------------------------

    save_confusion_matrix(
        y_test,
        tuned_predictions,
    )

    # --------------------------------------------------------
    # Train final tuned model on all available data
    # --------------------------------------------------------

    print("\nTraining final model on the full dataset...")

    final_model = grid_search.best_estimator_

    final_model.fit(
        texts,
        labels,
    )

    # Save final model
    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    # Clear cached old model
    load_model.cache_clear()

    print("\nModel saved to:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Extract top features
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

        print(
            ", ".join(features)
        )

    # --------------------------------------------------------
    # Update EVALUATION.md
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
# 10. Load saved model
# ============================================================

@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained sentiment model.

    The model is cached so it is loaded only once.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained sentiment model was not found.\n"
            "Run this command first:\n\n"
            "python nlp-core/sentiment_analyzer.py"
        )

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# 11. Validate input text
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
# 12. Analyze one text
# ============================================================

def analyze_sentiment(text):
    """
    Analyze the sentiment of one text.

    Returns:
        The predicted sentiment label as a string.
    """

    cleaned_text = validate_text(text)

    model = load_model()

    prediction = model.predict(
        [cleaned_text]
    )[0]

    return str(prediction)


# ============================================================
# 13. Analyze multiple texts
# ============================================================

def batch_analyze(texts):
    """
    Analyze multiple texts.

    Args:
        texts:
            A list of input sentences.

    Returns:
        A list of predicted sentiment labels.
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

    model = load_model()

    predictions = model.predict(
        cleaned_texts
    )

    return [
        str(prediction)
        for prediction in predictions
    ]


# ============================================================
# 14. Main
# ============================================================

if __name__ == "__main__":

    # Train, tune, and save the sentiment model
    train_model()

    # Single prediction test
    print("\n" + "=" * 60)
    print("SINGLE TEXT TEST")
    print("=" * 60)

    test_text = "الخدمة ممتازة جدا"

    result = analyze_sentiment(
        test_text
    )

    print(f"\nText: {test_text}")
    print(f"Prediction: {result}")

    # Batch prediction test
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