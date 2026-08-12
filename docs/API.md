# Content Analyzer API Documentation

## تحليل الصور (Computer Vision)

### `POST /analyze/image`

Send form-data with an image under `image`, or JSON containing `image_path`.

```json
{
  "image_path": "uploads/test_image.jpg"
}
```

```json
{
  "message": "تم تحليل الصورة بنجاح",
  "analysis": {
    "image_path": "uploads/test_image.jpg",
    "faces": 0,
    "shapes": [],
    "colors": [{"color": [255, 0, 0], "percentage": 0.45}],
    "summary": {"num_faces": 0, "num_shapes": 0, "num_colors": 1},
    "db_id": true
  }
}
```

### `GET /results`

Returns recent stored results. No request body is required.

```json
[
  [1, "cv", "uploads/test.jpg", "{\"shapes\": [], \"colors\": []}", "2023-10-25 14:30:00"]
]
```

## تحليل النصوص (NLP)

Text-analysis requests use UTF-8 JSON and `Content-Type: application/json`.
`text` must be a non-empty Arabic string.

### `POST /analyze/intent`

```json
{
  "text": "كيف أرفع صورة؟",
  "save_to_db": false
}
```

Successful response (`200 OK`):

```json
{
  "type": "intent",
  "language": {"language": "ar", "confidence": 1.0},
  "result": {"label": "سؤال", "confidence": 0.9417}
}
```

### `POST /analyze/sentiment`

```json
{
  "text": "الخدمة ممتازة والموظفون متعاونون جدًا",
  "save_to_db": false
}
```

Successful response (`200 OK`):

```json
{
  "type": "sentiment",
  "language": {"language": "ar", "confidence": 1.0},
  "result": {"label": "إيجابي"}
}
```

`save_to_db` is optional and defaults to `false`. When true, the complete
result is stored in SQLite before the response is returned.

Invalid input returns `400 Bad Request`:

```json
{
  "error": {
    "code": "invalid_text",
    "message": "text must be a non-empty Arabic string"
  }
}
```

Model-loading or persistence failures return `500 Internal Server Error`:

```json
{
  "error": {
    "code": "analysis_failed",
    "message": "The text could not be analyzed"
  }
}
```

### Additional NLP endpoints

- `POST /analyze/keywords` — extracts keywords from a JSON `text` value.
- `POST /analyze/language` — returns the detected language and confidence.
- `POST /analyze/summary` — summarizes the supplied JSON `text` value.

## الإحصائيات (Analytics)

### `GET /stats`

```json
{
  "total_analyses": 100,
  "cv_analyses": 40,
  "text_analyses": 60
}
```
