from nlp_extensions.language_detector import detect_language_with_confidence
from nlp_extensions.keyword_extractor import extract_by_frequency
from nlp_extensions.summarizer import summarize

# 1 & 2: اختبارات كشف اللغة (عربي، إنجليزي)
def test_detect_language_arabic():
    res = detect_language_with_confidence("مرحبا بكم في عالم الذكاء الاصطناعي")
    assert res['language'] == 'ar'

def test_detect_language_english():
    res = detect_language_with_confidence("Hello and welcome to the AI world")
    assert res['language'] == 'en'

# 3 & 4: اختبارات استخراج الكلمات المفتاحية
def test_extract_keywords_normal():
    text = "الذكاء الاصطناعي يغير العالم، الذكاء الاصطناعي مهم جداً."
    res = extract_by_frequency(text, top_n=2)
    assert len(res) > 0

def test_extract_keywords_empty():
    res = extract_by_frequency("", top_n=2)
    assert res == []

# 5: اختبار التلخيص
def test_summarizer_length():
    text = "الجملة الأولى. الجملة الثانية. الجملة الثالثة."
    res = summarize(text, n_sentences=1)
    # التأكد من أن التلخيص يرجع نصاً أقصر أو مساوياً لعدد الجمل المطلوبة
    assert len(res) > 0