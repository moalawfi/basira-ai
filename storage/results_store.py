import sqlite3
import json
from storage.db_setup import DB_PATH, get_connection

def log_action(action, status, error_msg=None):
    """تسجيل العمليات وحالة نجاحها أو الأخطاء في جدول السجلات"""
    try:
        conn = get_connection(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO logs (action, status, error_msg)
            VALUES (?, ?, ?)
        ''', (action, status, error_msg))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to log action: {e}")

def save_nlp_result(input_text, results_dict):
    """حفظ نتائج تحليل النصوص في قاعدة البيانات"""
    try:
        conn = get_connection(DB_PATH)
        cursor = conn.cursor()
        # تحويل القاموس إلى نص JSON ليتم حفظه
        output_json = json.dumps(results_dict, ensure_ascii=False)
        
        cursor.execute('''
            INSERT INTO results (type, input_text, output_json)
            VALUES (?, ?, ?)
        ''', ('nlp', input_text, output_json))
        
        conn.commit()
        conn.close()
        log_action("save_nlp_result", "success")
        return True
    except Exception as e:
        log_action("save_nlp_result", "failed", str(e))
        return False

def save_cv_result(image_path, results_dict):
    """حفظ نتائج تحليل الصور (خاص بياسر)"""
    try:
        conn = get_connection(DB_PATH)
        cursor = conn.cursor()
        output_json = json.dumps(results_dict, ensure_ascii=False)
        
        cursor.execute('''
            INSERT INTO results (type, input_text, output_json)
            VALUES (?, ?, ?)
        ''', ('cv', image_path, output_json))
        
        conn.commit()
        conn.close()
        log_action("save_cv_result", "success")
        return True
    except Exception as e:
        log_action("save_cv_result", "failed", str(e))
        return False

def get_results(type_filter=None, limit=10):
    """استرجاع آخر النتائج مع إمكانية الفلترة بنوع معين"""
    try:
        conn = get_connection(DB_PATH)
        cursor = conn.cursor()
        
        if type_filter:
            cursor.execute('''
                SELECT * FROM results 
                WHERE type = ? 
                ORDER BY timestamp DESC LIMIT ?
            ''', (type_filter, limit))
        else:
            cursor.execute('''
                SELECT * FROM results 
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            
        rows = cursor.fetchall()
        conn.close()
        log_action("get_results", "success")
        return rows
    except Exception as e:
        log_action("get_results", "failed", str(e))
        return []