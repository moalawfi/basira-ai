# Known Limitations

## OCR (Optical Character Recognition)

The `cv_module/ocr_extractor.py` module uses `pytesseract` for text extraction
from images. This requires **two separate components**:

1. **pytesseract** - the Python wrapper package (`pip install pytesseract`)
2. **Tesseract binary** - the OCR engine itself, installed separately

### Current Status

- The Tesseract binary is installed at
  `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- It may not be available in `PATH` on every system or shell. In that case,
  configure `pytesseract.pytesseract.tesseract_cmd` with the path above, or add
  the Tesseract installation directory to `PATH`.
- When Tesseract is missing, `extract_text()` returns:
  ```json
  {"success": false, "text": "", "error": "Tesseract binary is not installed or not in PATH."}
  ```

### How to Install Tesseract

**Windows:**
1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and note the installation path.
3. Add the Tesseract install directory to your system PATH, or set it in Python:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt install tesseract-ocr
```

---

## NLP Intent and Sentiment Analysis

- `classify_intent()` and `analyze_sentiment()` are trained on **Arabic text
  only**. Passing English or other languages will raise a `ValueError`.
- The models are loaded from pre-trained `.joblib` files in the `nlp-core/`
  directory. If these files are missing, the functions will fail.

---

## NLP Module Import (nlp-core)

- The `nlp-core/` directory uses a **hyphen** in its name, which prevents
  normal Python `import` statements.
- The `services/nlp_service.py` layer uses `importlib.util.spec_from_file_location`
  to dynamically load these modules.
- Do **not** rename `nlp-core/` without coordinating with the team.

---

## Database

- Results are stored in SQLite (`content_analyzer.db`).
- The database file is created automatically on first run via `storage/db_setup.py`.
- Concurrent writes from multiple processes are not supported.
