import sys
import os

# التأكد من قراءة المجلدات بشكل صحيح
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from nlp_extensions.language_detector import detect_language_with_confidence
from nlp_extensions.keyword_extractor import extract_by_frequency
from nlp_extensions.summarizer import summarize_with_stats
from storage.results_store import save_nlp_result
from storage.analytics import total_analyses_by_type

def run_pipeline():
    print("--- Starting End-to-End NLP Pipeline ---")
    
    # 1. إدخال النص
    text = "الذكاء الاصطناعي ومعالجة اللغات الطبيعية من أهم التقنيات الحديثة التي تساهم في تطور البشرية. " * 15
    
    # 2. كشف اللغة
    print("\n[1] Detecting Language...")
    lang_info = detect_language_with_confidence(text)
    print(lang_info)
    
    # 3. استخراج الكلمات المفتاحية
    print("\n[2] Extracting Keywords...")
    keywords = extract_by_frequency(text, top_n=3)
    print(keywords)
    
    # 4. التلخيص
    print("\n[3] Summarizing...")
    summary_info = summarize_with_stats(text, n_sentences=1)
    print(f"Compression Ratio: {summary_info['compression_ratio']}")
    
    # 5. الحفظ في قاعدة البيانات
    print("\n[4] Saving to Database...")
    final_data = {
        "language": lang_info,
        "keywords": keywords,
        "summary_stats": summary_info
    }
    is_saved = save_nlp_result(text, final_data)
    print(f"Database Save: {'Success' if is_saved else 'Failed'}")
    
    # 6. جلب التحليلات المجمعة
    print("\n[5] Fetching Analytics...")
    stats = total_analyses_by_type()
    print(f"Total Analyses by type: {stats}")
    
    print("\n--- Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    run_pipeline()