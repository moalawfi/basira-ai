# NLP Extensions Documentation

هذا المجلد يحتوي على وحدات معالجة اللغات الطبيعية (NLP) المطورة داخل المشروع.

## الوحدات المتاحة (Available Modules)

### 1. كشف اللغة (`language_detector.py`)
- `detect_language_with_confidence(text)`: تقوم بتحديد لغة النص (عربي/إنجليزي) وتُرجع النتيجة مع نسبة الثقة (Confidence).

### 2. استخراج الكلمات المفتاحية (`keyword_extractor.py`)
- `extract_by_frequency()`: استخراج الكلمات المفتاحية بناءً على معدل التكرار.
- `extract_by_tfidf()`: استخراج الكلمات المفتاحية باستخدام خوارزمية TF-IDF.
- `extract_by_textrank()`: استخراج الكلمات المفتاحية باستخدام خوارزمية TextRank.
- `batch_extract()`: معالجة مجموعة من النصوص واستخراج الكلمات منها دفعة واحدة.

### 3. التلخيص (`summarizer.py`)
- `summarize(text, n_sentences)`: تلخيص استخراجي للنصوص بناءً على أهم الجمل.
- `summarize_with_stats(text)`: تقوم بالتلخيص مع حساب نسبة ضغط النص (Compression Ratio).

### 4. استخراج المواضيع - Topic Modeling (`topic_modeling.py`)
- `discover_topics(n_topics, n_words)`: استخراج المواضيع الرئيسية من النصوص المخزنة مسبقاً في قاعدة البيانات باستخدام خوارزمية LDA.