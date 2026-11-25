<div align="center">

# 👁️ Basira · بصيرة
### Smart Content Analyzer — محلل المحتوى الذكي

**An Arabic-first AI toolkit that reads text *and* sees images — intent, sentiment, keywords, faces, shapes, colors and OCR, all behind one CLI, one REST API and a modern web dashboard.**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-NLP-F7931E?logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Storage-003B57?logo=sqlite&logoColor=white)
![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)
![Location](https://img.shields.io/badge/📍-Medina%2C%20Saudi%20Arabia-2E7D32)
![Summer Training](https://img.shields.io/badge/Summer%20Training-Program-F9A825)

👤 **Author:** [Mohammed Alawfi](https://github.com/moalawfi) &nbsp;·&nbsp; 📍 **Medina, Saudi Arabia** &nbsp;·&nbsp; 🎓 **Summer Training Program**

[Features](#-features) · [Quick start](#-quick-start) · [API](#-rest-api) · [Architecture](#-architecture) · [Team](#-team)

</div>

---

## ✨ Why this project?

Most off-the-shelf analyzers treat Arabic as an afterthought. **Basira** (Arabic for *insight*) was built the other way around: Arabic text is a first-class citizen, and the same pipeline also handles English text and images. Drop in a sentence or a photo and get structured, stored, queryable results.

```text
"كيف أرفع صورة؟"       ──►  intent: سؤال (question)   ·  language: ar
"الخدمة ممتازة جدًا"   ──►  sentiment + keywords + language
group_photo.png        ──►  faces + shapes + dominant colors + OCR text
```

## 🚀 Features

| | Capability | Details |
|---|---|---|
| 💬 | **Intent classification** | Arabic intent model — question, command, greeting, complaint (TF-IDF + scikit-learn, tuned with GridSearchCV) |
| ❤️ | **Sentiment analysis** | Trained on an Arabic sentiment dataset, evaluated with cross-validation & confusion matrix |
| 🔑 | **Keyword extraction** | TF-IDF-based keyword ranking |
| 🌐 | **Language detection** | Arabic / English detection |
| 📝 | **Summarization** | Extractive text summarizer |
| 🗂️ | **Topic modeling** | Discover themes across documents |
| 🙂 | **Face detection** | OpenCV face detection with bounding boxes |
| 🔺 | **Shapes & colors** | Shape recognition + dominant colors via KMeans |
| 🔤 | **OCR** | Extract text from images (Tesseract) |
| 🗄️ | **Storage & analytics** | Every result saved to SQLite with logs and an analytics cache |
| 🖥️ | **Three interfaces** | Interactive CLI, Flask REST API, and a web dashboard |

## ⚡ Quick start

```bash
git clone https://github.com/moalawfi/basira-ai.git
cd basira-ai

python -m venv venv
# Windows: venv\Scripts\activate   ·   macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

> **OCR (optional):** install [Tesseract](https://github.com/tesseract-ocr/tesseract) and `pip install pytesseract`.

### Run it

| Mode | Command | What you get |
|---|---|---|
| 🖥️ Web dashboard + API | `python app.py` | Open **http://localhost:5000** |
| ⌨️ Interactive CLI | `python main.py` | Menu-driven text & image analysis |
| ✅ Tests | `pytest tests/` | Unit and API tests |

### Use it from Python

```python
import sys
sys.path.insert(0, "nlp-core")

from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment

print(classify_intent("كيف أرفع صورة؟"))
print(analyze_sentiment("الخدمة ممتازة جدًا"))
```

## 🔌 REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/analyze/intent` | Classify the intent of a text |
| `POST` | `/analyze/sentiment` | Sentiment of a text |
| `POST` | `/analyze/keywords` | Extract keywords |
| `POST` | `/analyze/language` | Detect language |
| `POST` | `/analyze/summary` | Summarize a text |
| `POST` | `/analyze/image` | Full image pipeline (faces, shapes, colors, OCR) |
| `POST` | `/analyze/faces` · `/shapes` · `/colors` | Individual vision tasks |
| `GET` | `/results` | Latest stored results |
| `GET` | `/stats` | Usage analytics |

```bash
curl -X POST http://localhost:5000/analyze/sentiment \
     -H "Content-Type: application/json" \
     -d '{"text": "الخدمة ممتازة جدًا"}'

curl -X POST http://localhost:5000/analyze/image -F "image=@tests/test_images/images/faces/two_faces.jpg"
```

Full request/response contracts: [`docs/API.md`](docs/API.md).

## 🏗️ Architecture

```text
            ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
            │  CLI main.py │   │ Flask app.py │   │  Web dashboard   │
            └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘
                   └──────────────────┼────────────────────┘
                                      ▼
                         services/  (shared service layer)
               ┌──────────────────────┼───────────────────────┐
               ▼                      ▼                       ▼
         nlp-core/             nlp_extensions/            cv_module/
   intent · sentiment     keywords · language ·     faces · shapes · colors
                          summary · topics                 · OCR
               └──────────────────────┼───────────────────────┘
                                      ▼
                       storage/  →  SQLite (results · logs · analytics_cache)
```

```text
basira-ai/
├── app.py                 # Flask API + web dashboard
├── main.py                # Interactive CLI
├── nlp-core/              # Intent & sentiment models, datasets, evaluation
├── nlp_extensions/        # Keywords, language, summarizer, topic modeling
├── cv_module/             # Face, shape/color detection, OCR
├── services/              # Shared NLP & image pipelines
├── storage/               # SQLite setup, result store, analytics
├── templates/ · static/   # Dashboard UI
├── tests/                 # pytest suites + test images
└── docs/                  # API, DB schema, limitations
```

## 📊 Evaluation

Model evaluation reports live next to the code:

- [`nlp-core/EVALUATION.md`](nlp-core/EVALUATION.md) — intent & sentiment metrics, confusion matrix
- [`nlp_extensions/EVALUATION.md`](nlp_extensions/EVALUATION.md) — NLP extensions
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — known limitations
- [`docs/db_schema.md`](docs/db_schema.md) — database schema

## 👥 Team

Developed in **Medina, Saudi Arabia** 🇸🇦 as part of our **Summer Training Program**, built as a team project following a real-world Git workflow — feature branches, reviewed pull requests, GitHub Issues and a Kanban board.

| Member | Area |
|---|---|
| **Mohammed Alawfi** | NLP Core — intent classification, sentiment analysis, model evaluation |
| **Yasser Althibyani** | Computer Vision engine, CLI, Flask API, tests |
| **Abdulrahman Almutairi** | NLP Extensions, database & analytics |

## 🤝 Contributing

Issues and pull requests are welcome. Please branch from `main`, keep changes focused, and make sure `pytest tests/` passes before opening a PR.

---

<div align="center">

If you find this project useful, consider giving it a ⭐

</div>
