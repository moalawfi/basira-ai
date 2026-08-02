import os
import sys
import json

# التأكد من قدرة بايثون على قراءة المجلدات الأخرى
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from storage.db_setup import get_connection, DB_PATH

def discover_topics(n_topics=3, n_words_per_topic=5):
    """
    استخراج المواضيع (Topics) من النصوص المخزنة في قاعدة البيانات
    باستخدام خوارزمية LDA
    """
    try:
        # 1. الاتصال بقاعدة البيانات وجلب النصوص
        conn = get_connection(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT input_text FROM results WHERE type='nlp'")
        rows = cursor.fetchall()
        conn.close()

        # تصفية النصوص الفارغة أو القصيرة جداً
        texts = [row[0] for row in rows if row[0] and len(row[0].strip()) > 10]
        
        # التأكد من وجود نصوص كافية
        if len(texts) == 0:
            return {"error": "لا يوجد نصوص كافية في قاعدة البيانات"}
            
        # تعديل عدد المواضيع إذا كانت النصوص قليلة جداً
        actual_topics = min(n_topics, len(texts))

        # 2. تحويل النصوص إلى مصفوفة أرقام وتكرار
        # min_df=1 لضمان عملها حتى لو كانت النصوص التجريبية في جهازك قليلة
        vectorizer = CountVectorizer(max_df=1.0, min_df=1) 
        tf = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # 3. تطبيق خوارزمية LDA
        lda = LatentDirichletAllocation(n_components=actual_topics, random_state=42)
        lda.fit(tf)
        
        # 4. تجميع النتائج
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            # جلب أعلى الكلمات لكل موضوع
            top_features_ind = topic.argsort()[:-n_words_per_topic - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics[f"Topic_{topic_idx + 1}"] = top_features
            
        return topics

    except Exception as e:
        return {"error": f"حدث خطأ أثناء استخراج المواضيع: {str(e)}"}

# كود تجريبي لاختبار الدالة عند التشغيل
if __name__ == "__main__":
    print("--- Starting Topic Modeling from Database ---")
    extracted_topics = discover_topics(n_topics=3, n_words_per_topic=5)
    print(json.dumps(extracted_topics, indent=4, ensure_ascii=False))