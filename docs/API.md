1. تحليل الصور (Computer Vision)

تحليل صورة وتخزين النتيجة

الطريقة (Method): POST

المسار (Endpoint): /analyze/image

المدخلات (Request): Form-Data تحتوي على ملف صورة بمفتاح image.

نموذج الاستجابة (Response Example):
{  "message": "تم تحليل الصورة وحفظ النتائج بنجاح",  "analysis": {    "faces": [      {        "count": 1,        "bounding_boxes": [[10, 20, 150, 150]]      }    ],    "shapes": [      {        "type": "circle",        "area": 314.15,        "location": [100, 100]      }    ],    "dominant_colors": [      {"color": [255, 0, 0], "percentage": 0.45}    ],    "db_id": 1  }}


2. تحليل النصوص (NLP - Mocks)

تحليل النية (Intent Analysis)

الطريقة (Method): POST

المسار (Endpoint): /analyze/intent

المدخلات (Request Body - JSON):
{
  "text": "أريد إلغاء طلبي"
}

نموذج الاستجابة (Response Example):
{
  "intent": "mock_intent",
  "input": "أريد إلغاء طلبي"
}

## تحليل المشاعر (Sentiment Analysis)

الطريقة (Method): POST

المسار (Endpoint): /analyze/sentiment

المدخلات (Request Body - JSON):
{
  "text": "الخدمة كانت ممتازة جداً"
}

نموذج الاستجابة (Response Example):
{
  "sentiment": "positive",
  "confidence": 0.95
}

## استخراج الكلمات المفتاحية (Keywords Extraction)

الطريقة (Method): POST

المسار (Endpoint): /analyze/keywords

المدخلات (Request Body - JSON):
{
  "text": "الذكاء الاصطناعي يغير مستقبل التكنولوجيا"
}

نموذج الاستجابة (Response Example):
{
  "keywords": [
    "mock",
    "keyword",
    "test"
  ]
}

## تحديد اللغة (Language Detection)

الطريقة (Method): POST

المسار (Endpoint): /analyze/language

المدخلات (Request Body - JSON):
{
  "text": "Hello world"
}

نموذج الاستجابة (Response Example):
{
  "language": "ar",
  "confidence": 0.99
}

## تلخيص النص (Text Summarization)

الطريقة (Method): POST

المسار (Endpoint): /analyze/summary

المدخلات (Request Body - JSON):
{
  "text": "نص طويل جداً يحتوي على العديد من الجمل والتفاصيل..."
}

نموذج الاستجابة (Response Example):
{
  "summary": "This is a mocked summary of the text."
}


3. الإحصائيات (Analytics)

عرض إحصائيات النظام

الطريقة (Method): GET

المسار (Endpoint): /stats

المدخلات: لا يوجد.

نموذج الاستجابة (Response Example):
{
  "total_analyses": 100,
  "cv_analyses": 40,
  "text_analyses": 60,
  "most_common_shapes": [
    "circle",
    "rectangle"
  ]
}

