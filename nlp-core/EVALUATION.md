# NLP Model Evaluation

This file documents the evaluation of the intent-classification and
sentiment-analysis models before and after the Week 6 dataset expansion.

# Week 4: Results Before Dataset Expansion

## Intent Classification

| Model | Configuration | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Naive Bayes | Before tuning | 0.4000 | 0.3667 |
| Logistic Regression | Before tuning | 0.4000 | 0.2917 |
| Naive Bayes | After tuning | 0.3000 | 0.2833 |
| Logistic Regression | After tuning | **0.6000** | **0.5095** |

The selected intent model was Logistic Regression with:

```text
classifier__C = 0.01
```

## Sentiment Analysis

| Model | Configuration | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Logistic Regression | Before tuning | 0.2500 | 0.2857 |
| Logistic Regression | After tuning | 0.2500 | 0.2857 |

The selected sentiment-model parameter was:

```text
classifier__C = 0.01
```

The best cross-validation accuracy was `0.4619`.

---

# Week 6: Results After Dataset Expansion

Twenty additional examples were added to each dataset. The new examples
focused on short, ambiguous, and difficult sentences that were likely to
be confused by the previous models.

The models were retrained and evaluated using a stratified train-test
split with `random_state=42`.

## Intent Classification

| Model | Configuration | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Naive Bayes | Before tuning | 0.3333 | 0.3548 |
| Logistic Regression | Before tuning | 0.3333 | **0.3639** |
| Naive Bayes | After tuning | 0.3333 | 0.3548 |
| Logistic Regression | After tuning | 0.3333 | **0.3639** |

The selected intent model was Logistic Regression with:

```text
classifier__C = 1.0
```

### Saved Intent Test Files

- All test predictions: `tests/intent_test_results.csv`
- Incorrect predictions: `tests/intent_misclassified.csv`

### Top Intent Features

| Intent class | Top features |
|---|---|
| أمر | اعرض, احذف, في, احفظ, اعرض لي, النتائج, التقرير, ارفع الصورة, ارفع, الصورة للتحليل |
| تحية | الخير, هلا, مرحب, أهلين, صباح, مساء الخير, مساء, كيف الحال, الحال, السلام |
| سؤال | هل, وين ألقى, ألقى, وين, النص, أقدر, ممكن, ممكن تساعدني, تساعدني, البيانات |
| شكوى | ما, معي, ما قدرت, قدرت, قدرت أرفع, النتيجة, كل مرة, يعلق, كل, التطبيق |

## Sentiment Analysis

| Model | Configuration | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Logistic Regression | Before tuning | 0.2353 | 0.2556 |
| Logistic Regression | After tuning | **0.2941** | **0.2707** |

The selected sentiment-model parameter was:

```text
classifier__C = 0.01
```

The best cross-validation accuracy was `0.3933`.

### Saved Sentiment Test Files

- All test predictions: `tests/sentiment_test_results.csv`
- Incorrect predictions: `tests/sentiment_misclassified.csv`

### Top Sentiment Features

| Sentiment class | Top features |
|---|---|
| إيجابي | ممتازة, الخدمة ممتازة, أعجبني, جد, رائع, ممتازة وسريعة, وسريعة, ترتيب التقرير, أعجبني ترتيب, ترتيب |
| سلبي | غير, ولا, سيئة, مملة, ولم, سيئ, التقرير غير, غير واضح, النتيجة غير, صحيحة |
| محايد | في, تم, على, الملف, يحتوي, خمس, بدأ, يوم, صباح, الفصل |

---

# Before-and-After Dataset Expansion Comparison

## Intent Model Comparison

The best tuned Logistic Regression result from each week was used for
the comparison.

| Metric | Before expansion | After expansion | Change |
|---|---:|---:|---:|
| Accuracy | 0.6000 | 0.3333 | -0.2667 |
| Macro-F1 | 0.5095 | 0.3639 | -0.1456 |
| Selected `C` | 0.01 | 1.0 | Changed |

The measured intent accuracy decreased after dataset expansion. However,
this does not necessarily mean that the additional examples made the
model worse.

Although the same `random_state` was used, adding new examples changed
the complete dataset. Therefore, `train_test_split()` selected a
different group of sentences for the Week 6 test set.

The Week 6 test set may also contain more difficult and ambiguous
examples than the previous test set. This makes the two scores useful
for documentation, but they are not a perfectly controlled comparison.

The Week 6 model learned useful new features such as:

- `ممكن تساعدني`
- `وين ألقى`
- `ما قدرت`
- `كل مرة يعلق`

These features indicate that the additional examples expanded the
model's vocabulary and improved its coverage of short and difficult
inputs.

## Sentiment Model Comparison

| Metric | Before expansion | After expansion | Change |
|---|---:|---:|---:|
| Accuracy | 0.2500 | 0.2941 | +0.0441 |
| Macro-F1 | 0.2857 | 0.2707 | -0.0150 |
| Best CV accuracy | 0.4619 | 0.3933 | -0.0686 |
| Selected `C` | 0.01 | 0.01 | No change |

The sentiment model achieved a small improvement in test accuracy,
increasing from `0.2500` to `0.2941`.

However, Macro-F1 decreased slightly from `0.2857` to `0.2707`. This
suggests that the accuracy improvement was not distributed equally
across the positive, negative, and neutral classes.

The newly added examples introduced clearer sentiment features such as:

- `الخدمة ممتازة`
- `ممتازة وسريعة`
- `التقرير غير واضح`
- `النتيجة غير صحيحة`
- `تم حفظ التقرير`
- `يحتوي الملف`

These expressions expanded the model's coverage of positive, negative,
and neutral sentences.

---

# Summary of Changes

| Model | Accuracy before expansion | Accuracy after expansion | Result |
|---|---:|---:|---|
| Intent classification | 0.6000 | 0.3333 | Decreased on the new test split |
| Sentiment analysis | 0.2500 | 0.2941 | Improved by 0.0441 |

The sentiment model showed a small improvement in accuracy after adding
the new examples.

The intent model produced a lower score after expansion. However, the
test set changed when the dataset size changed, so the reduction cannot
be attributed only to the added examples.

---

# Overall Interpretation

Dataset expansion improved the linguistic coverage of both NLP models
by introducing more difficult, short, and ambiguous sentences.

The sentiment model showed a small increase in test accuracy, while the
intent model achieved lower accuracy on the new held-out split.

The results should be interpreted carefully because the Week 4 and Week
6 models were not evaluated using exactly the same test sentences. A
fixed `random_state` reproduces a split only when the dataset itself
remains unchanged.

Because twenty new examples were added to each dataset, the total number
and ordering of samples changed. This caused a new test set to be
selected during Week 6.

For a strictly controlled before-and-after comparison, a fixed test
dataset should be created and saved before expanding the training data.
Both the old and new models should then be evaluated using that same
unchanged test dataset.

Despite the unstable scores caused by the small dataset, the new
examples expanded the models' vocabulary and provided better coverage
of difficult cases. The saved test-result files can now be used to
inspect errors and guide future dataset improvements.

---

# Model Persistence

The trained models are saved using `joblib` so that the system does not
need to retrain them every time it starts.

The model-persistence functions are:

```python
save_model(model, path)
load_model(path)
```

The generated model files are:

- `nlp-core/intent_model.joblib`
- `nlp-core/sentiment_model.joblib`

The prediction functions load these saved models during execution.

---

# Generated Files

## Saved Models

- `nlp-core/intent_model.joblib`
- `nlp-core/sentiment_model.joblib`

## Evaluation Files

- `nlp-core/EVALUATION.md`
- `nlp-core/sentiment_cm.png`

## Intent Test Results

- `tests/intent_test_results.csv`
- `tests/intent_misclassified.csv`

## Sentiment Test Results

- `tests/sentiment_test_results.csv`
- `tests/sentiment_misclassified.csv`

The complete test-result CSV files contain:

- model name
- evaluation stage
- input sentence
- actual label
- predicted label
- prediction confidence
- whether the prediction was correct

The misclassified CSV files contain only the incorrectly classified
sentences, making them useful for future error analysis and dataset
expansion.

---

# Conclusion

In Week 6, twenty new examples were added to each NLP dataset, and both
models were retrained.

The sentiment model's accuracy increased from `0.2500` to `0.2941`,
while its Macro-F1 decreased slightly.

The intent model's measured accuracy decreased from `0.6000` to
`0.3333`, but this comparison was affected by the change in the held-out
test set after dataset expansion.

Both models now support persistent saving and loading through `joblib`.
Their complete test predictions and classification errors are also
saved as CSV files for future analysis.

Future evaluation should use a fixed test dataset to provide a more
reliable comparison between model versions.

<!-- INTENT_EVALUATION_START -->

## Intent Classification Evaluation

The models were evaluated using the same held-out test split.
Hyperparameter tuning was performed only on the training data
using `GridSearchCV`.

### Results

| Model | Stage | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Naive Bayes | Before tuning | 0.9600 | 0.9596 |
| Logistic Regression | Before tuning | 0.9880 | 0.9880 |
| Naive Bayes | After tuning | 0.9640 | 0.9638 |
| Logistic Regression | After tuning | 0.9960 | 0.9960 |

### Selected Model

**Best model:** Logistic Regression

**Best hyperparameters:** `classifier__C` = `10.0`

### Saved Test Files

- All predictions: `tests/intent_test_results_after_expansion.csv`
- Failed predictions: `tests/intent_misclassified_after_expansion.csv`
- Test summary: `tests/intent_test_summary_after_expansion.csv`

### Top 10 Features per Intent Class

| Intent class | Top 10 features |
|---|---|
| أمر | اعرض, احفظ, الآن, احذف, افتح, استرجع, جديد, حد, ارفع, الملف |
| تحية | يا, الله, مرحب, جميع, أهلين, مساء, أهل, صباح, الخير, النور |
| سؤال | هل, ليش, وش, أقدر, كيف أقدر, كم, متى, وين, ألقى, وين ألقى |
| شكوى | لا, غير, ما, جد, عند, رغم, بعد, النظام, التحديث, ناقص |

<!-- INTENT_EVALUATION_END -->

<!-- SENTIMENT_EVALUATION_START -->

## Sentiment Analysis Evaluation

The baseline and tuned sentiment models were evaluated using
the same held-out test set. `GridSearchCV` was performed only
on the training set to prevent test-data leakage.

### Results Before and After Tuning

| Model | Stage | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Logistic Regression | Before tuning | 0.9693 | 0.9694 |
| Logistic Regression | After tuning | 0.9693 | 0.9695 |

### Hyperparameter Tuning

**Method:** GridSearchCV

**Scoring metric:** Accuracy

**Best cross-validation accuracy:** 0.9514

**Best hyperparameters:** `classifier__C` = `100.0`

### Saved Test Files

- All predictions: `tests/sentiment_test_results.csv`
- Failed predictions: `tests/sentiment_misclassified.csv`

### Top 10 Features per Sentiment Class

| Sentiment class | Top 10 features |
|---|---|
| إيجابي | وأعجبتني, جد, وتستحق التجربة, وتستحق, كانت, بسرعة, ممتازة, فعالة, مقنعة, التجربة |
| سلبي | غير, ولم, مخيبة, سيئة, متعبة, مملة, ولم تعجبني, تعجبني, فشل, لم |
| محايد | تم, في, يحتوي, على, يبدأ, موعد, داخل, يوم, الساعة, في الساعة |

### Generated Files

- `sentiment_model.joblib`
- `sentiment_cm.png`
- `tests/sentiment_test_results.csv`
- `tests/sentiment_misclassified.csv`

<!-- SENTIMENT_EVALUATION_END -->

<!-- WEEK7_ARABIC_EVALUATION_START -->

# Week 7: Arabic Language Support and E2E Testing

Ten fixed Arabic sentences were processed using `language_detector`, the intent classifier, and the sentiment analyzer.

## Accuracy Summary

| Metric | Accuracy | Correct | Total |
|---|---:|---:|---:|
| Language detection | 1.0000 | 10 | 10 |
| Intent classification | 0.9000 | 9 | 10 |
| Sentiment analysis | 0.8000 | 8 | 10 |
| Complete E2E result | 0.7000 | 7 | 10 |

## Detailed Results

| Text | Language | Expected Intent | Predicted Intent | Expected Sentiment | Predicted Sentiment |
|---|---|---|---|---|---|
| هلا والله، حياكم الله | ar | تحية | تحية | محايد | محايد |
| أهلين، كيف الحال؟ | ar | تحية | تحية | محايد | محايد |
| هل أقدر أطلع نتائج التحليل السابقة؟ | ar | سؤال | سؤال | محايد | محايد |
| متى يكتمل رفع المستند؟ | ar | سؤال | سؤال | محايد | سلبي |
| ورني التقارير الموجودة | ar | أمر | سؤال | محايد | محايد |
| سجل النتيجة في قاعدة البيانات | ar | أمر | أمر | محايد | محايد |
| نفذ تحليل المحتوى لهذا الملف | ar | أمر | أمر | محايد | سلبي |
| النظام يتأخر كثير في إظهار النتيجة | ar | شكوى | شكوى | سلبي | سلبي |
| كل ما أحاول أرسل الملف يفشل | ar | شكوى | شكوى | سلبي | سلبي |
| التصنيف الذي ظهر لي خاطئ تمامًا | ar | شكوى | شكوى | سلبي | سلبي |

A model score below `0.80` should be reviewed before deciding whether more Arabic training examples and retraining are required.

<!-- WEEK7_ARABIC_EVALUATION_END -->
