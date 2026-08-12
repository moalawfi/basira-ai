```markdown
# Content Analyzer API Documentation



## 1. تحليل الصور (Computer Vision)



### تحليل صورة وتخزين النتيجة



Method `POST`



Endpoint `/analyze/image`



Request Form-Data تحتوي على ملف صورة بمفتاح `image`، أو JSON يحتوي على `"image_path"`.



Response

```json
{
  "message": "تم تحليل الصورة بنجاح",
  "analysis": {
    "image_path": "uploads\\test_image.jpg",
    "faces": "Not available (face_detector.py not found)",
    "shapes": [],
    "colors": [
      {"color": [255, 0, 0], "percentage": 0.45}
    ],
    "summary": {
      "num_faces": 0,
      "num_shapes": 0,
      "num_colors": 1
    },
    "db_id": true
  }
}
```



### استرجاع آخر النتائج من قاعدة البيانات



Method `GET`



Endpoint `/results`



Request لا يوجد.



Response

```json
[
  [1, "cv", "uploads\\test.jpg", "{\"shapes\": [], \"colors\": []}", "2023-10-25 14:30:00"]
]
```



---



## 2. تحليل النصوص (NLP - Mocks)



### تحليل النية (Intent Analysis)



Method `POST`



Endpoint `/analyze/intent`



Request

```json
{
  "text": "أريد إلغاء طلبي"
}
```



Response

```json
{
  "intent": "mock_intent",
  "input": "أريد إلغاء طلبي"
}
```



### تحليل المشاعر (Sentiment Analysis)



Method `POST`



Endpoint `/analyze/sentiment`



Request

```json
{
  "text": "الخدمة كانت ممتازة جداً"
}
```



Response

```json
{
  "sentiment": "positive",
  "confidence": 0.95
}
```



### استخراج الكلمات المفتاحية (Keywords Extraction)



Method `POST`



Endpoint `/analyze/keywords`



Request

```json
{
  "text": "الذكاء الاصطناعي يغير مستقبل التكنولوجيا"
}
```



Response

```json
{
  "keywords": [
    "mock",
    "keyword",
    "test"
  ]
}
```



### تحديد اللغة (Language Detection)



Method `POST`



Endpoint `/analyze/language`



Request

```json
{
  "text": "Hello world"
}
```



Response

```json
{
  "language": "ar",
  "confidence": 0.99
}
```



### تلخيص النص (Text Summarization)



Method `POST`



Endpoint `/analyze/summary`



Request

```json
{
  "text": "نص طويل جداً يحتوي على العديد من الجمل والتفاصيل..."
}
```



Response

```json
{
  "summary": "This is a mocked summary of the text."
}
```



---



## 3. الإحصائيات (Analytics)



### عرض إحصائيات النظام



Method `GET`



Endpoint `/stats`



Request لا يوجد.



Response

```json
{
  "total_analyses": 100,
  "cv_analyses": 40,
  "text_analyses": 60
}
```