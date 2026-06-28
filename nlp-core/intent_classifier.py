import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import json
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR /'nlp-core'/ "intent_data.csv"

data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(data.head())
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer( ngram_range=(1,2) )

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_model = MultinomialNB()
lr_model = LogisticRegression()

nb_model.fit(X_train_tfidf, y_train)
lr_model.fit(X_train_tfidf, y_train)

nb_pred = nb_model.predict(X_test_tfidf)
lr_pred = lr_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(y_test, nb_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Naive Bayes Accuracy:", nb_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)

print("\n==============================")
print("Naive Bayes Classification Report")
print("==============================")
print(classification_report(y_test, nb_pred, zero_division=0))
print("\n===================================")
print("Logistic Regression Classification Report")
print("===================================")
print(classification_report(y_test, lr_pred, zero_division=0))



if lr_accuracy >= nb_accuracy:
    best_model = lr_model
    best_model_name = "Logistic Regression"
else:
    best_model = nb_model
    best_model_name = "Naive Bayes"

print("\nBest Model:", best_model_name)



def classify_intent(text):
    text_tfidf = vectorizer.transform([text])
    prediction = best_model.predict(text_tfidf)
    return {
        "type": "intent",
        "result": {
            "label": prediction
        }
    }



print("\nTest Predictions:")
print("السلام عليكم =>", classify_intent("السلام عليكم"))
print("كيف أرفع صورة؟ =>", classify_intent("كيف أرفع صورة؟"))
print("حلل هذا النص =>", classify_intent("حلل هذا النص"))
print("التطبيق بطيء جدًا =>", classify_intent("التطبيق بطيء جدًا"))
