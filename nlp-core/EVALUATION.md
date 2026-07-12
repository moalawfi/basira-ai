# NLP Model Evaluation

This file is generated automatically by the training scripts.

<!-- INTENT_EVALUATION_START -->

## Intent Classification Evaluation

The baseline and tuned models were evaluated using the same held-out test set. Hyperparameter tuning was performed only on the training data using `GridSearchCV`.

### Results Before and After Tuning

| Model | Stage | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Naive Bayes | Before tuning | 0.4000 | 0.3667 |
| Logistic Regression | Before tuning | 0.4000 | 0.2917 |
| Naive Bayes | After tuning | 0.3000 | 0.2833 |
| Logistic Regression | After tuning | 0.6000 | 0.5095 |

### Selected Model

**Best model:** Logistic Regression

**Best hyperparameters:** `classifier__C` = `0.01`

### Top 10 Features per Intent Class

| Intent class | Top 10 features |
|---|---|
| أمر | اعرض, اعرض الكلمات, ارفع, ارفع الصورة, الصورة للتحليل, عملية التحليل, عملية, ابدأ, ابدأ عملية, افتح صفحة |
| تحية | هلا, الخير, أهلين, مرحب, الحال, كيف الحال, مساء الخير, هلا والله, يا, يا هلا |
| سؤال | هل, النص, أقدر, يتم حفظها, حفظها, البيانات يتم, يتم, هل البيانات, دخول, أسجل دخول |
| شكوى | ما, النتيجة, التطبيق, بطيء, غير, جد, بطيء جد, النتيجة غير, غير دقيقة, دقيقة |

<!-- INTENT_EVALUATION_END -->

<!-- SENTIMENT_EVALUATION_START -->

## Sentiment Analysis Evaluation

The baseline and tuned sentiment models were evaluated using the same held-out test set. `GridSearchCV` was performed only on the training set to prevent test-data leakage.

### Results Before and After Tuning

| Model | Stage | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Logistic Regression | Before tuning | 0.2500 | 0.2857 |
| Logistic Regression | After tuning | 0.2500 | 0.2857 |

### Hyperparameter Tuning

**Method:** GridSearchCV

**Scoring metric:** Accuracy

**Best cross-validation accuracy:** 0.4619

**Best hyperparameters:** `classifier__C` = `0.01`

### Top 10 Features per Sentiment Class

| Sentiment class | Top 10 features |
|---|---|
| إيجابي | جد, رائع, حصلت, الفريق, متفائل بشأن, أنا متفائل, متفائل, متعاونون, والموظفون متعاونون, ممتازة |
| سلبي | غير, ولا, مملة, ولم, سيئ, كثير, أنا قلق, قلق بشأن, قلق, بطيئة والموظفون |
| محايد | في, تم, يوم, يحتوي, الفصل, صباح, على, ساعتين, الرحلة, تستغرق الرحلة |

### Generated Files

- `sentiment_model.joblib`
- `sentiment_cm.png`

<!-- SENTIMENT_EVALUATION_END -->
### Interpretation

Hyperparameter tuning did not improve the held-out test performance. This is likely due to the limited dataset size, where each sentiment class contains only a small number of examples. As a result, the train-test split is sensitive to individual samples, and the selected hyperparameter may not generalize reliably. Expanding the dataset and using repeated stratified cross-validation may provide a more stable evaluation.