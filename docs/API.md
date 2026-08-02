# Content Analyzer API Documentation

هذا الملف يوضح المسارات (Endpoints) الخاصة بخدمات تحليل المحتوى (NLP & Analytics).

## NLP Endpoints

### 1. Language Detection
- **Endpoint:** `/analyze/language`
- **Method:** `POST`
- **Description:** يحدد لغة النص المدخل (عربي/إنجليزي) مع إرجاع نسبة الثقة.
- **Payload:** `{"text": "نص عربي تجريبي"}`
- **Response:** `{"language": "ar", "confidence": 1.0}`

### 2. Keyword Extraction
- **Endpoint:** `/analyze/keywords`
- **Method:** `POST`
- **Description:** يستخرج أهم الكلمات المفتاحية من النص.
- **Payload:** `{"text": "...", "method": "tfidf", "top_n": 5}`
- **Response:** `{"keywords": ["كلمة1", "كلمة2"]}`

### 3. Summarization
- **Endpoint:** `/analyze/summary`
- **Method:** `POST`
- **Description:** يلخص النص المدخل ويرجع نسبة الضغط.
- **Payload:** `{"text": "...", "n_sentences": 2}`
- **Response:** `{"summary": "...", "compression_ratio": "50%"}`

## Analytics Endpoints

### 4. General Statistics
- **Endpoint:** `/stats`
- **Method:** `GET`
- **Description:** يجلب الإحصائيات المجمعة من قاعدة البيانات (مثل توزيع اللغات، والمشاعر، والمواضيع).
- **Response:** `{"total_analyses": {...}, "language_distribution": {...}}`