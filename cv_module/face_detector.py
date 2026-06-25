import cv2
import os

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_faces(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    faces_list = [{"box": [int(x), int(y), int(w), int(h)]} for (x, y, w, h) in faces]
    
    return {
        "count": len(faces_list),
        "boxes": faces_list
    }

def draw_faces(image_path, output_path):
    
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, img)
    
    return {"message": f"Successfully saved drawn faces to {output_path}", "faces_drawn": len(faces)}

def detect_faces_with_eyes(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    valid_faces = []
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        if len(eyes) >= 2:
            valid_faces.append({"box": [int(x), int(y), int(w), int(h)], "eyes_found": len(eyes)})
            
    return {
        "count": len(valid_faces),
        "boxes": valid_faces
    }

if __name__ == "__main__":
    os.makedirs("test_outputs", exist_ok=True)
    
    test_images = {
        "No Faces": "test_images/no_faces.jpg",
        "One Face": "test_images/one_face.jpg",
        "Two Faces": "test_images/two_faces.jpg",
        "Group (5+)": "test_images/group_photo.png"
    }
    
    print("--- Starting Week 2 Tests ---")
    
    for desc, path in test_images.items():
        print(f"\n[Testing: {desc}] -> Path: {path}")
        
        result_normal = detect_faces(path)
        print(f"Normal Detection: Found {result_normal.get('count', 0)} faces.")
        
        result_eyes = detect_faces_with_eyes(path)
        print(f"Eye-Validated Detection: Found {result_eyes.get('count', 0)} valid faces.")
        
        if "error" not in result_normal:
            out_path = f"test_outputs/{desc.replace(' ', '_').lower()}_output.jpg"
            draw_result = draw_faces(path, out_path)
            print(f"Drawing: {draw_result.get('message', draw_result.get('error'))}")
        else:
            print(f"Skipping drawing: {result_normal['error']}")
            
    print("\n--- Tests Completed ---")