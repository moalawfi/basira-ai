# Storage Layer Documentation

هذا المجلد يحتوي على الوحدة الخاصة بإدارة قاعدة البيانات (SQLite) وحفظ نتائج التحليل في المشروع.

## تصميم الجداول (Database Schema)
يتم إنشاء الجداول التالية تلقائياً عند تشغيل `db_setup.py`:
1. **results**: يحفظ نتائج التحليل (الخاصة بالنصوص NLP والصور CV).
2. **logs**: يسجل حركات النظام والأخطاء للرجوع إليها.
3. **analytics_cache**: يخزن الإحصائيات المجمعة لتسريع الاستعلامات.

## الدوال المتاحة (Available Functions)
- `init_db()`: تهيئة وإنشاء الجداول في قاعدة البيانات.
- `drop_all_tables()`: حذف جميع الجداول (تُستخدم لأغراض الاختبار والتنظيف).
- `save_nlp_result(text, result_dict)`: حفظ نتيجة تحليل نص معين.
- `save_cv_result(image_path, result_dict)`: حفظ نتيجة تحليل صورة معينة.
- `get_results(type_filter, limit)`: جلب النتائج السابقة مع إمكانية الفلترة حسب النوع.
- `log_action(action, status, error_msg)`: توثيق وتسجيل الأخطاء أو العمليات.

## مثال على الاستخدام (Usage Example)
```python
from storage.results_store import save_nlp_result

# حفظ نتيجة تحليل نصية
save_nlp_result("نص تجريبي للذكاء الاصطناعي", {"language": "ar", "confidence": 0.99})

---

### 📄 2. توثيق مجلد معالجة النصوص (NLP Extensions)

1. في الـ VS Code، اذهب إلى مجلد **`nlp_extensions/`**.
2. أنشئ داخله ملفاً جديداً باسم: **`README.md`**
3. انسخ هذا المحتوى بالكامل وضعه داخله، ثم احفظ الملف ($Ctrl + S$):

```markdown
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