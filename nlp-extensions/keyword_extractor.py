import re
from collections import Counter
import math

ARABIC_STOPWORDS = set([
    "في", "من", "على", "إلى", "عن", "أو", "ثم", "أن", "إنا", "هذا", "هذه", "التي", 
    "الذي", "مع", "كل", "هو", "هي", "كان", "كانت", "تم", "لقد", "بين", "بعد", "قبل"
])

ENGLISH_STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", 
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", 
    "this", "that", "am", "is", "are", "was", "were", "be", "been", "being", "have", 
    "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", 
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", 
    "with", "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", 
    "over", "under", "again", "further", "then", "once", "here", "there", "when", 
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", 
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", 
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

def clean_and_tokenize(text):
    text = text.lower()
    words = re.findall(r'[\u0600-\u06FF\w]+', text)
    
    
    cleaned_words = [
        w for w in words 
        if w not in ARABIC_STOPWORDS and w not in ENGLISH_STOPWORDS and not w.isdigit()
    ]
    return cleaned_words


def extract_by_frequency(text, top_n=5):
    words = clean_and_tokenize(text)
    
    if not words:
        return []
        
    word_counts = Counter(words)
    top_words = word_counts.most_common(top_n)
    
    return [word for word, count in top_words]



def extract_by_tfidf(text, corpus, top_n=5):

    current_words = clean_and_tokenize(text)
    if not current_words:
        return []
        
    cleaned_corpus = [clean_and_tokenize(doc) for doc in corpus]
    total_docs = len(cleaned_corpus) + 1  
    
    word_counts = Counter(current_words)
    total_words = len(current_words)
    
    tfidf_scores = {}
    for word, count in word_counts.items():
        tf = count / total_words
        
        doc_count = 1 if any(word in doc for doc in cleaned_corpus) else 0
        if word in current_words:
            doc_count += 1
            
        idf = math.log(total_docs / doc_count) + 1
        
        tfidf_scores[word] = tf * idf
        
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]


def extract_by_textrank(text, top_n=5):
    words = clean_and_tokenize(text)
    if not words:
        return []
        
    word_set = list(set(words))
    word_indices = {word: i for i, word in enumerate(word_set)}
    n = len(word_set)
    
    graph = [[0] * n for _ in range(n)]
    window_size = 4
    
    for i in range(len(words)):
        for j in range(i + 1, min(i + window_size, len(words))):
            w1, w2 = words[i], words[j]
            if w1 != w2:
                idx1, idx2 = word_indices[w1], word_indices[w2]
                graph[idx1][idx2] += 1
                graph[idx2][idx1] += 1
                
    scores = [1.0] * n
    d = 0.85
    
    for _ in range(10):
        new_scores = [1.0 - d] * n
        for i in range(n):
            for j in range(n):
                if graph[j][i] > 0:
                    total_out = sum(graph[j])
                    if total_out > 0:
                        new_scores[i] += d * (scores[j] * graph[j][i] / total_out)
        scores = new_scores
        
    word_scores = {word_set[i]: scores[i] for i in range(n)}
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]


def batch_extract(texts, method="tfidf", corpus=None, top_n=5):
    results = []
    if corpus is None:
        corpus = texts
        
    for text in texts:
        if method == "frequency":
            keywords = extract_by_frequency(text, top_n=top_n)
        elif method == "textrank":
            keywords = extract_by_textrank(text, top_n=top_n)
        else:
            keywords = extract_by_tfidf(text, corpus=corpus, top_n=top_n)
        results.append(keywords)
        
    return results



if __name__ == "__main__":
    test_texts = [
        "الذكاء الاصطناعي يغير العالم بشكل سريع جداً وتطوير الأنظمة الذكية ممتع",
        "تقنيات معالجة اللغات الطبيعية تساعد في تحليل النصوص بدقة عالية وسهولة",
        "Python is an amazing programming language for machine learning and data science",
        "Natural language processing provides powerful tools for automated text analysis",
        "النصوص المختلطة تحتوي على كلمات مثل AI و الذكاء الاصطناعي في نفس الوقت"
    ]
    
    print("--- Testing Frequency Method ---")
    for text in test_texts[:2]:
        print(extract_by_frequency(text, top_n=3))
        
    print("\n--- Testing TF-IDF Method ---")
    print(extract_by_tfidf(test_texts[0], test_texts, top_n=3))
    
    print("\n--- Testing TextRank Method ---")
    print(extract_by_textrank(test_texts[2], top_n=3))
    
    print("\n--- Testing Batch Extraction ---")
    batch_results = batch_extract(test_texts, method="tfidf", top_n=3)
    for res in batch_results:
        print(res)