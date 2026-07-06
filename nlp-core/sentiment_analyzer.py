from pathlib import Path
from functools import lru_cache

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "sentiment_dataset_ar.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.joblib"
CM_PATH = BASE_DIR / "sentiment_cm.png"


# ============================================================
# Load Dataset
# ============================================================

def load_data():
    """
    Load and validate the Arabic sentiment dataset.

    Expected CSV columns:
    - النص
    - الفئة
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # utf-8-sig handles Arabic CSV files saved by Excel/VS Code
    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig"
    )

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    print("\nDataset columns:")
    print(df.columns.tolist())

    # Convert Arabic column names to internal English names
    df = df.rename(
        columns={
            "النص": "text",
            "الفئة": "label",
        }
    )

    required_columns = {"text", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Remove missing values
    df = df.dropna(
        subset=["text", "label"]
    ).copy()

    # Clean text and labels
    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
    )

    # Remove empty rows
    df = df[
        (df["text"] != "")
        & (df["label"] != "")
    ]

    if df.empty:
        raise ValueError(
            "Dataset is empty after cleaning."
        )

    # Show class distribution
    class_counts = df["label"].value_counts()

    print("\nClass distribution:")
    print(class_counts)

    # Required for 5-fold cross-validation
    if class_counts.min() < 5:
        raise ValueError(
            "Each sentiment class must contain at least "
            "5 samples for 5-fold cross-validation."
        )

    return df["text"], df["label"]


# ============================================================
# Build Model
# ============================================================

def build_model():
    """
    Create a machine learning pipeline containing:

    1. TF-IDF vectorization
    2. Logistic Regression classifier
    """

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


# ============================================================
# Train and Evaluate Model
# ============================================================

def train_model():
    """
    Train the sentiment model.

    The function:
    1. Loads the dataset
    2. Performs 5-fold cross-validation
    3. Prints accuracy and Macro-F1
    4. Creates a confusion matrix
    5. Saves the trained model
    """

    print("=" * 60)
    print("SENTIMENT ANALYZER TRAINING")
    print("=" * 60)

    # Load data
    texts, labels = load_data()

    print(f"\nDataset size: {len(texts)}")

    print("\nClass distribution:")
    print(labels.value_counts())

    # Build model
    model = build_model()

    # 5-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # --------------------------------------------------------
    # Accuracy Evaluation
    # --------------------------------------------------------

    accuracy_scores = cross_val_score(
        model,
        texts,
        labels,
        cv=cv,
        scoring="accuracy",
    )

    # --------------------------------------------------------
    # Macro-F1 Evaluation
    # --------------------------------------------------------

    macro_f1_scores = cross_val_score(
        model,
        texts,
        labels,
        cv=cv,
        scoring="f1_macro",
    )

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print("\nAccuracy scores:")

    for fold, score in enumerate(accuracy_scores, start=1):
        print(f"Fold {fold}: {score:.4f}")

    print(f"\nMean Accuracy: {accuracy_scores.mean():.4f}")
    print(f"Accuracy Standard Deviation: {accuracy_scores.std():.4f}")

    print("\nMacro-F1 scores:")

    for fold, score in enumerate(macro_f1_scores, start=1):
        print(f"Fold {fold}: {score:.4f}")

    print(f"\nMean Macro-F1: {macro_f1_scores.mean():.4f}")
    print(f"Macro-F1 Standard Deviation: {macro_f1_scores.std():.4f}")

    # --------------------------------------------------------
    # Cross-validated Predictions
    # --------------------------------------------------------

    predictions = cross_val_predict(
        model,
        texts,
        labels,
        cv=cv,
        method="predict",
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            labels,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    class_names = sorted(labels.unique())

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=class_names,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(
        "Sentiment Analysis Confusion Matrix\n"
        "5-Fold Cross-Validation"
    )

    fig.tight_layout()

    fig.savefig(
        CM_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nConfusion matrix saved successfully:")
    print(CM_PATH)

    # --------------------------------------------------------
    # Train Final Model on All Data
    # --------------------------------------------------------

    print("\nTraining final model on the full dataset...")

    model.fit(texts, labels)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully:")
    print(MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    return model


# ============================================================
# Load Saved Model
# ============================================================

@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained sentiment model.

    The model is cached so it is loaded only once.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Sentiment model not found.\n"
            "Run this file first to train the model:\n\n"
            "python nlp-core/sentiment_analyzer.py"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# Validate Input
# ============================================================

def validate_text(text):
    """
    Validate a single input text.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()

    if not text:
        raise ValueError("text cannot be empty")

    return text


# ============================================================
# Analyze One Text
# ============================================================

def analyze_sentiment(text):
    """
    Analyze the sentiment of one text.

    Args:
        text: Input sentence.

    Returns:
        Predicted sentiment label.
    """

    text = validate_text(text)

    model = load_model()

    prediction = model.predict([text])[0]

    return str(prediction)


# ============================================================
# Analyze Multiple Texts
# ============================================================

def batch_analyze(texts: list):
    """
    Analyze multiple texts at once.

    Args:
        texts: List of input sentences.

    Returns:
        List of sentiment predictions.
    """

    if not isinstance(texts, list):
        raise TypeError("texts must be a list")

    if len(texts) == 0:
        return []

    cleaned_texts = []

    for text in texts:
        cleaned_text = validate_text(text)
        cleaned_texts.append(cleaned_text)

    model = load_model()

    predictions = model.predict(cleaned_texts)

    return predictions.tolist()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Train model
    train_model()

    # Test single prediction
    print("\n" + "=" * 60)
    print("SINGLE TEXT TEST")
    print("=" * 60)

    test_text = "الخدمة ممتازة جدا"

    result = analyze_sentiment(test_text)

    print(f"\nText: {test_text}")
    print(f"Prediction: {result}")

    # Test batch prediction
    print("\n" + "=" * 60)
    print("BATCH TEST")
    print("=" * 60)

    test_texts = [
        "الخدمة ممتازة جدا",
        "التجربة كانت سيئة",
        "وصل الطلب اليوم",
    ]

    results = batch_analyze(test_texts)

    for text, sentiment in zip(test_texts, results):
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment}")