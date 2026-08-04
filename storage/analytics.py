import sqlite3
import json
from storage.db_setup import DB_PATH, get_connection

def cache_metric(metric_name, metric_value):
    """حفظ النتيجة في جدول التخزين المؤقت"""
    try:
        conn = get_connection(DB_PATH)
        cursor = conn.cursor()
        # تحويل القيمة لنص JSON لتناسب قاعدة البيانات
        cursor.execute('''
            INSERT INTO analytics_cache (metric_name, metric_value)
            VALUES (?, ?)
        ''', (metric_name, json.dumps(metric_value, ensure_ascii=False)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Cache error: {e}")

def total_analyses_by_type():
    """حساب عدد التحليلات لكل نوع"""
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT type, COUNT(*) FROM results GROUP BY type')
    rows = cursor.fetchall()
    conn.close()
    
    result = {row[0]: row[1] for row in rows}
    cache_metric('total_analyses_by_type', result)
    return result

def sentiment_distribution():
    """توزيع المشاعر عبر التحليلات (إن وجدت)"""
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT output_json FROM results WHERE type='nlp'")
    rows = cursor.fetchall()
    conn.close()
    
    distribution = {"positive": 0, "negative": 0, "neutral": 0}
    for row in rows:
        try:
            data = json.loads(row[0])
            if "sentiment" in data:
                sent = data["sentiment"].lower()
                if sent in distribution:
                    distribution[sent] += 1
        except:
            pass
            
    cache_metric('sentiment_distribution', distribution)
    return distribution

def top_keywords_overall(n=10):
    """أكثر الكلمات المفتاحية تكراراً"""
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT output_json FROM results WHERE type='nlp'")
    rows = cursor.fetchall()
    conn.close()
    
    keyword_counts = {}
    for row in rows:
        try:
            data = json.loads(row[0])
            if "keywords" in data:
                for kw in data["keywords"]:
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        except:
            pass
            
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    result = {k: v for k, v in sorted_keywords}
    cache_metric('top_keywords_overall', result)
    return result

def language_distribution():
    """توزيع اللغات (عربي مقابل إنجليزي)"""
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT output_json FROM results WHERE type='nlp'")
    rows = cursor.fetchall()
    conn.close()
    
    distribution = {"ar": 0, "en": 0, "other": 0}
    for row in rows:
        try:
            data = json.loads(row[0])
            if "language" in data and "language" in data["language"]:
                lang = data["language"]["language"]
                if lang in distribution:
                    distribution[lang] += 1
                else:
                    distribution["other"] += 1
        except:
            pass
            
    cache_metric('language_distribution', distribution)
    return distribution

def daily_activity(days=7):
    """عدد التحليلات يومياً في آخر 7 أيام"""
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DATE(timestamp) as day, COUNT(*) 
        FROM results 
        WHERE timestamp >= date('now', ?) 
        GROUP BY day
    ''', (f'-{days} days',))
    rows = cursor.fetchall()
    conn.close()
    
    result = {row[0]: row[1] for row in rows}
    cache_metric('daily_activity', result)
    return result

if __name__ == "__main__":
    print("Total Analyses:", total_analyses_by_type())
    print("Language Dist:", language_distribution())
    print("Analytics cached successfully!")