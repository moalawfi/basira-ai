from cv_module.face_detector import detect_faces
from cv_module.shape_color_detector import analyze_shapes_colors
from cv_module.ocr_extractor import extract_text

def show_menu():
    print("\n=== Smart Content Analyzer ===")
    print("1. Text Analysis (intent + sentiment)")
    print("2. Extended Text Analysis (keywords + language + summary)")
    print("3. Image Analysis")
    print("4. View All Stored Results")
    print("5. View Analytics")
    print("6. Exit")


while True:
    show_menu()
    choice = input("choose a number : ")

    if choice == "1":
        print("Mock: Text Analysis")

    elif choice == "2":
        print("Mock: Extended Text Analysis")

    elif choice == "3":
        print(detect_faces("test.jpg"))
        print(analyze_shapes_colors("test.jpg"))
        print(extract_text("test.jpg"))

    elif choice == "4":
        print("Mock: Display Results")

    elif choice == "5":
        print("Mock: Statistics")

    elif choice == "6":
        print("Goodbye!")
        break

    else:
        print("wrong choice")
