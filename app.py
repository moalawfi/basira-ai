# app.py - Week 4 Version
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# التأكد من وجود مجلد للصور المرفوعة
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# استيراد آمن لقاعدة البيانات (يجب أن يكون هنا قبل استخدامه)
# ==========================================
save_cv_result = None
get_results = None

try:
    from storage.results_store import save_cv_result, get_results
except Exception as e:
    print(f"Warning: Database modules not fully ready. Using mocks. Error: {e}")
    
    def save_cv_result(img_path, data):
        print(f"[DB MOCK] Saved result for: {img_path}")
        return 1  # يرجع رقم وهمي
    
    def get_results(limit=10):
        print(f"[DB MOCK] Fetching {limit} results.")
        return []  # يرجع قائمة فارغة


# استيراد دالة تحليل الصور من مجلد cv_module
from cv_module.shape_color_detector import analyze_image


# ==========================================
# 1. تحليل صورة وربطها بقاعدة البيانات
# ==========================================
@app.route('/analyze/image', methods=['POST'])
def analyze_image_endpoint():
    """يستقبل مسار صورة أو ملف مرفوع، يستدعي analyze_image، ويرجع JSON"""
    image_path = None

    # الحالة الأولى: إرسال الصورة كملف مرفوع (Form-Data)
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "اسم الملف فارغ"}), 400
        
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        
    # الحالة الثانية: إرسال مسار الصورة كنص (JSON)
    elif request.is_json:
        data = request.get_json()
        image_path = data.get("image_path")
        
    else:
        return jsonify({"error": "يجب إرسال صورة مرفوعة أو مسار الصورة بصيغة JSON"}), 400

    # التأكد من أن الصورة موجودة فعلاً
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "الصورة غير موجودة في المسار المحدد"}), 404

    try:
        # 1. استدعاء دالة تحليل الصور (الأسبوع 3)
        analysis_result = analyze_image(image_path)

        # 2. الربط مع قاعدة البيانات (المهمة 1 و 2 من الأسبوع 4)
        if save_cv_result:
            db_id = save_cv_result(image_path, analysis_result)
            analysis_result["db_id"] = db_id
        else:
            analysis_result["db_id"] = None
            analysis_result["db_warning"] = "Database module not connected"

        return jsonify({
            "message": "تم تحليل الصورة بنجاح",
            "analysis": analysis_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# 2. استرجاع آخر 10 نتائج من قاعدة البيانات
# ==========================================
@app.route('/results', methods=['GET'])
def get_results_endpoint():
    """يرجع آخر 10 نتائج من قاعدة البيانات"""
    if not get_results:
        return jsonify({"error": "Database module is not available yet"}), 503
        
    try:
        results = get_results(limit=10)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)