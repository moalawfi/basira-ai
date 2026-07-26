import os
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from storage.results_store import save_cv_result

try:
    from face_detector import detect_faces
    _FACE_DETECTOR_AVAILABLE = True
except Exception:
    _FACE_DETECTOR_AVAILABLE = False
    print("Warning: face_detector.py not found - Face detection disabled in analyze_image.")

def detect_shapes(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_AREA = 500
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        sides = len(approx)

        if sides == 3:
            shape_type = "Triangle"
        elif sides == 4:
            x_, y_, w_, h_ = cv2.boundingRect(approx)
            aspect = float(w_) / h_ if h_ != 0 else 0
            shape_type = "Square" if 0.9 <= aspect <= 1.1 else "Rectangle"
        elif sides == 5:
            shape_type = "Pentagon"
        elif sides == 6:
            shape_type = "Hexagon"
        else:
            circularity = (4 * np.pi * area) / (peri ** 2) if peri > 0 else 0
            if circularity > 0.75:
                shape_type = "Circle"
                sides = 0
            else:
                shape_type = f"Polygon ({sides})"

        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        results.append({
            "type": shape_type,
            "sides": int(sides),
            "area": float(area),
            "location": {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "center": (cx, cy)}
        })

    return results

def detect_dominant_colors(image_path, n_colors=3):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    if max(h, w) > 200:
        scale = 200.0 / max(h, w)
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    counts = Counter(kmeans.labels_)
    total = sum(counts.values())
    colors = []
    
    for i in range(n_colors):
        count = counts.get(i, 0)
        center = kmeans.cluster_centers_[i]
        colors.append({
            "color": (int(center[0]), int(center[1]), int(center[2])),
            "percentage": round(count / total * 100, 2)
        })

    colors.sort(key=lambda c: c["percentage"], reverse=True)
    return colors

def analyze_image(image_path, save_to_db=False):
    result = {"image_path": image_path, "faces": [], "shapes": [], "colors": []}

    if _FACE_DETECTOR_AVAILABLE:
        try:
            result["faces"] = detect_faces(image_path)
        except Exception as e:
            print(f"Error in face detection: {e}")
            result["faces"] = []
    else:
        result["faces"] = "Not available (face_detector.py not found)"

    try:
        result["shapes"] = detect_shapes(image_path)
    except Exception as e:
        print(f"Error in shape detection: {e}")
        result["shapes"] = []

    try:
        result["colors"] = detect_dominant_colors(image_path, n_colors=3)
    except Exception as e:
        print(f"Error in color detection: {e}")
        result["colors"] = []

    result["summary"] = {
        "num_faces": (len(result["faces"]) if isinstance(result["faces"], list) else 0),
        "num_shapes": len(result["shapes"]),
        "num_colors": len(result["colors"]),
    }

    # ==========================================
    # إضافة المهمة 2: الحفظ الاختياري في قاعدة البيانات
    # ==========================================
    if save_to_db:
        try:
            # نستدعي دالة عبدالرحمن لحفظ النتيجة
            db_id = save_cv_result(image_path, result)
            
            # نضيف رقم التعريف الخاص بقاعدة البيانات للنتيجة كإثبات نجاح الحفظ
            result["db_id"] = db_id
            print(f"Successfully saved analysis to DB with ID: {db_id}")
            
        except Exception as e:
            # إذا حدث خطأ في قاعدة البيانات، لا نوقف البرنامج، بل نرسل تحذير مع النتيجة
            result["db_error"] = f"Failed to save to DB: {str(e)}"
            print(f"Database Error: {e}")
    # ==========================================

    return result

def draw_and_save_analysis(image_path, output_path, analysis):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to load image"}

    for shape in analysis.get("shapes", []):
        loc = shape["location"]
        x, y, w, h = loc["x"], loc["y"], loc["w"], loc["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, shape["type"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    return {"message": f"Saved to {output_path}"}


if __name__ == "__main__":
    import os
    
    os.makedirs("test_outputs", exist_ok=True)
    
    test_images = {
        "Blue Circle": "test_images/shapes/blue_circle.png",
        "Circle": "test_images/shapes/circle.png",
        "Heptagon": "test_images/shapes/heptagon.png",
        "Hexagon": "test_images/shapes/hexagon.png",
        "Red Triangle": "test_images/shapes/red_triangle.png",
        "Square": "test_images/shapes/square.png",
        "Triangel": "test_images/shapes/triangel.png"
    }
    
    print("--- Starting Week 3 Tests ---")
    
    for desc, path in test_images.items():
        print(f"\n[Testing: {desc}] -> Path: {path}")
        
        analysis = analyze_image(path)
        print(f"Analysis: Found {analysis['summary']['num_shapes']} shapes.")
        
        for s in analysis["shapes"]:
            print(f"  - {s['type']} (Sides: {s['sides']}, Area: {s['area']})")
        
        if analysis["shapes"]:
            out_path = f"test_outputs/{desc.replace(' ', '_').lower()}_output.jpg"
            draw_result = draw_and_save_analysis(path, out_path, analysis)
            print(f"Drawing: {draw_result.get('message', draw_result.get('error'))}")
        else:
            print("Skipping drawing: No shapes detected.")
            
    print("\n--- Tests Completed ---")