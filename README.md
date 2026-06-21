# content-analyzer-team
# Smart Content Analyzer - محلل المحتوى الذكي

## Project Overview

Smart Content Analyzer is an integrated application that accepts **text or image input** from the user and analyzes it using multiple techniques. The project includes a simple REST API, organized SQLite database storage, logging, and automated testing.

The goal of this project is to simulate a real-world software development workflow using GitHub branches, Pull Requests, Issues, and a Kanban board.

---

## Team Members

| Name | Student ID | Main Role |
|---|---:|---|
| Mohammed Abdulrahman Alawfi | 4406223 | NLP Core |
| Yasser Misfer Althibyani | 4403545 | CV Engine + Interface |
| Abdulrahman Majed Almutairi | 4401251 | NLP Extensions + Database |

---

## Main Features

### Text Analysis

#### NLP Core - Mohammed
- Multi-class intent classification.
- Sentiment analysis.
- Model evaluation and comparison.

#### NLP Extensions - Abdulrahman
- Keyword extraction.
- Language detection for Arabic and English.
- Simple extractive summarization.
- Topic modeling as a stretch goal.

### Image Analysis - Yasser
- Face detection.
- Shape detection.
- Dominant color detection.
- OCR as a stretch goal.

### Storage - Abdulrahman
- SQLite database.
- Organized tables for results, logs, and analytics.
- Result saving and retrieval functions.

### Interface - Yasser
- Advanced CLI.
- Simple REST API using Flask.

### Testing - All Members
- Unit tests for the main functions using `pytest`.

---

## Git Workflow

```bash
git clone <repository-url>
git checkout -b <name>-module
git status
git add .
git commit -m "Describe the change"
git push origin <name>-module
git pull origin main
```

### Advanced Git Commands

```bash
git log --oneline --graph --all     # Show commit tree and branches
git diff main                       # Compare current branch with main
git rebase main                     # Update current branch with latest main changes
git stash                           # Temporarily save unfinished changes
git tag v0.1                        # Create a version tag
```

### Main Git Rule

No one should edit the `main` branch directly.

Each member should:
1. Work on their own branch.
2. Push changes to GitHub.
3. Open a Pull Request.
4. Get at least one review from another team member.
5. Merge only after review.

---

## Week 1 Repository Setup

### Task 0 - Day 1

1. Mohammed creates the GitHub repository named `content-analyzer-team`.
2. Mohammed adds Yasser and Abdulrahman as collaborators.
3. The team creates `.gitignore` to exclude:

```txt
__pycache__/
*.db
venv/
.env
```

4. The team creates `requirements.txt` with the required libraries:

```txt
scikit-learn
nltk
opencv-python
flask
pytest
```

5. Each member clones the repository and creates a personal branch.
6. The team creates GitHub Issues for weekly tasks.
7. The team uses GitHub Projects Kanban board to track progress.

---

## Team Responsibilities

| Member | Student ID | Responsibilities |
|---|---:|---|
| Mohammed Abdulrahman Alawfi | 4406223 | Intent classification, sentiment analysis, and model evaluation |
| Abdulrahman Majed Almutairi | 4401251 | Keyword extraction, language detection, summarization, database, and analytics |
| Yasser Misfer Althibyani | 4403545 | Face detection, shape detection, color detection, CLI, Flask API, and tests |

---

## Project Structure

```text
content-analyzer-team/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── app.py
├── nlp-core/
│   ├── intent_classifier.py
│   └── sentiment_analyzer.py
├── nlp-extensions/
│   ├── keyword_extractor.py
│   ├── language_detector.py
│   ├── summarizer.py
│   └── topic_modeling.py
├── cv-module/
│   ├── face_detector.py
│   ├── shape_color_detector.py
│   └── ocr_extractor.py
├── storage/
│   ├── db_setup.py
│   ├── results_store.py
│   └── analytics.py
├── interface/
│   └── cli_interface.py
├── tests/
│   ├── test_nlp_core.py
│   ├── test_nlp_extensions.py
│   ├── test_cv.py
│   └── test_storage.py
└── docs/
    └── API.md
```

---

## 8-Week Development Plan

| Week | Mohammed - NLP Core | Abdulrahman - NLP Extensions + DB | Yasser - CV + Interface |
|---|---|---|---|
| Week 1 | Setup and NLP Core skeleton | Setup, DB table design, and skeleton | Setup, project structure, and basic CLI |
| Week 2 | Intent classifier with 4 classes and dataset | Arabic/English language detection | Face detector with bounding boxes |
| Week 3 | Sentiment analyzer with cross-validation | Keyword extractor using TF-IDF | Shape detector and dominant colors using KMeans |
| Week 4 | Advanced evaluation using GridSearchCV | Build `db_setup.py` and `results_store.py` | Start Flask API and support database setup |
| Week 5 | Connect NLP Core to database and add tests | Extractive summarizer and DB integration | Main Flask endpoints and CV tests |
| Week 6 | Improve dataset and retrain models | `analytics.py` and NLP Extensions tests | Connect CV module to API |
| Week 7 | Arabic support and comprehensive testing | Topic modeling stretch goal and DB documentation | Full CLI/API integration and tests |
| Week 8 | Final documentation and demo preparation | Final documentation and presentation | Final testing, documentation, and presentation leadership |

---

## Final Success Criteria

By Week 8, the project should meet the following requirements:

1. Running `python main.py` opens a complete CLI that supports text and image analysis.
2. Running `python app.py` starts the Flask API on `localhost:5000`.
3. The API includes endpoints for each analysis type and analytics.
4. All results are saved in a SQLite database.
5. The database contains at least three tables:
   - `results`
   - `logs`
   - `analytics_cache`
6. Running `pytest tests/` passes without errors.
7. The project includes at least 15 test cases across 4 test files.
8. `docs/API.md` documents all endpoints with request and response examples.
9. The main `README.md` explains the project, setup, usage, and team roles.
10. The Git history includes 35+ commits distributed across team members.
11. GitHub Issues and Kanban board are used to track progress.

---

## General Notes

- Use GitHub Projects Kanban board to track tasks weekly.
- Each core function should include a clear docstring explaining inputs, outputs, and exceptions.
- Hold a weekly 20-30 minute meeting to review the Kanban board.
- If time becomes limited, prioritize:
  1. NLP Core
  2. Basic CV features
  3. Database
  4. Basic API
  5. Stretch features such as OCR, Topic Modeling, and Summarizer

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run CLI

```bash
python main.py
```

### Run Flask API

```bash
python app.py
```

---

## Running Tests

```bash
pytest tests/
```

---

## Current Status

Week 1 setup is in progress/completed depending on the current branch status.

Completed initial tasks include:
- Repository setup.
- NLP Core folder creation.
- Initial mock functions for intent classification and sentiment analysis.
- GitHub Issues.
- Kanban board setup.
