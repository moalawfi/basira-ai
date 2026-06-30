import re
from langdetect import detect_langs

def detect_language_regex(text):
    if not text.strip():
        return "unknown", 0.0
        
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = arabic_chars + english_chars
    
    if total_chars == 0:
        return "unknown", 0.0
        
    if arabic_chars >= english_chars:
        confidence = arabic_chars / total_chars
        return "ar", round(confidence, 2)
    else:
        confidence = english_chars / total_chars
        return "en", round(confidence, 2)
    

def detect_language_library(text):
    if not text.strip():
        return "unknown", 0.0
    try:
        predictions = detect_langs(text)
        best_match = predictions[0]
        lang = best_match.lang
        confidence = best_match.prob
        return lang, round(confidence, 2)
    except Exception:
        return "unknown", 0.0

def detect_language_with_confidence(text):
    lang_reg, conf_reg = detect_language_regex(text)
    lang_lib, conf_lib = detect_language_library(text)
    
    if lang_reg == "ar" and conf_reg > 0.5:
        final_lang = lang_reg
        final_conf = conf_reg
    else:
        final_lang = lang_lib if conf_lib >= conf_reg else lang_reg
        final_conf = max(conf_lib, conf_reg)
        
    return {
        "language": final_lang,
        "confidence": final_conf
    }


