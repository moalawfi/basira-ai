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
        print("Mock: Image Analysis")

    elif choice == "4":
        print("Mock: Display Results")

    elif choice == "5":
        print("Mock: Statistics")

    elif choice == "6":
        print("Goodbye!")
        break

    else:
        print("wrong choice")
