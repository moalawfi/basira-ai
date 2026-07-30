import re
import math
from collections import Counter

ARABIC_STOPWORDS = set([
    "في", "من", "على", "إلى", "عن", "أو", "ثم", "أن", "إنا", "هذا", "هذه", "التي", 
    "الذي", "مع", "كل", "هو", "هي", "كان", "كانت", "تم", "لقد", "بين", "بعد", "قبل"
])

ENGLISH_STOPWORDS = set([
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "she", "her", "it", 
    "they", "them", "what", "which", "who", "this", "that", "am", "is", "are", "was", 
    "were", "be", "been", "a", "an", "the", "and", "but", "if", "or", "because", "as", 
    "of", "at", "by", "for", "with", "about", "to", "from", "in", "out", "on", "so"
])

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def clean_words(sentence):
    words = re.findall(r'[\u0600-\u06FF\w]+', sentence.lower())
    return [w for w in words if w not in ARABIC_STOPWORDS and w not in ENGLISH_STOPWORDS and not w.isdigit()]

def summarize(text, n_sentences=2):
    sentences = split_into_sentences(text)
    if len(sentences) <= n_sentences:
        return text

    all_words = []
    sentence_words = []
    for sent in sentences:
        words = clean_words(sent)
        sentence_words.append(words)
        all_words.extend(words)

    if not all_words:
        return " ".join(sentences[:n_sentences])

    total_docs = len(sentences)
    word_counts = Counter(all_words)
    doc_freq = Counter()
    for words in sentence_words:
        for w in set(words):
            doc_freq[w] += 1

    scores = []
    for i, words in enumerate(sentence_words):
        score = 0.0
        if words:
            for w in words:
                tf = words.count(w) / len(words)
                idf = math.log(total_docs / (doc_freq[w] if doc_freq[w] > 0 else 1)) + 1
                score += tf * idf
            score /= len(words)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([item[0] for item in scores[:n_sentences]])

    selected_sentences = [sentences[idx] for idx in top_indices]
    return " ".join(selected_sentences)

def summarize_with_stats(text, n_sentences=2):
    summary = summarize(text, n_sentences=n_sentences)
    
    words_before = len(re.findall(r'\w+', text))
    words_after = len(re.findall(r'\w+', summary))
    
    compression_ratio = round((1 - (words_after / words_before)) * 100, 2) if words_before > 0 else 0.0

    return {
        "summary": summary,
        "words_before": words_before,
        "words_after": words_after,
        "compression_ratio": f"{compression_ratio}%"
    }


if __name__ == "__main__":
    import sys
    import os
    
    # التأكد من قدرة بايثون على قراءة المجلدات الأخرى
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from storage.results_store import save_nlp_result
    from language_detector import detect_language_with_confidence
    from keyword_extractor import extract_by_frequency

    # 1. تجهيز 3 نصوص طويلة (أكثر من 200 كلمة باستخدام التكرار)
    text1 = "يعد الذكاء الاصطناعي من أهم التقنيات الحديثة التي تساهم في تغيير شكل الحياة اليومية للبشر في مختلف القطاعات. " * 30
    text2 = "Natural language processing is a crucial field in modern computer science bridging the gap between humans and machines. " * 25
    text3 = "تقنيات تعلم الآلة تعتمد على البيانات الضخمة لاستخراج الأنماط والتنبؤ بالمستقبل في مختلف المجالات الطبية والهندسية. " * 25

    texts = [text1, text2, text3]

    print("--- Starting NLP Pipeline & DB Integration ---")
    
    # 2. معالجة النصوص وربطها بقاعدة البيانات
    for i, text in enumerate(texts, 1):
        print(f"\nProcessing Text {i}...")
        
        # استدعاء أدوات التحليل
        lang_info = detect_language_with_confidence(text)
        keywords = extract_by_frequency(text, top_n=5)
        summary_info = summarize_with_stats(text, n_sentences=2)
        
        # تجميع النتائج في قاموس واحد
        final_results_dict = {
            "language": lang_info,
            "keywords": keywords,
            "summary_stats": summary_info
        }
        
        # حفظ النتيجة في قاعدة البيانات
        is_saved = save_nlp_result(text, final_results_dict)
        
        print(f"Summary Compression: {summary_info['compression_ratio']}")
        print(f"Saved to Database: {'Success' if is_saved else 'Failed'}")