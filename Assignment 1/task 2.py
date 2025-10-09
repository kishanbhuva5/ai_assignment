shopping_list = []
while True:
    print("\nWould you like to")
    print("(1) Add or")
    print("(2) Remove or")
    print("(3) Quit?")
    user_choice = input("Your choice: ")
    if user_choice == "1":
        item = input("What will be added?: ")
        shopping_list.append(item)
    elif user_choice == "2":
        if len(shopping_list) == 0:
            print("The list is empty, nothing to remove.")
        else:
            print(f"There are {len(shopping_list)} items in the list.")
            try:
                index = int(input("Which item is deleted?: "))
                if 0 <= index < len(shopping_list):
                    removed = shopping_list.pop(index)
                    print(f"Removed: {removed}")
                else:
                    print("Incorrect selection.")
            except ValueError:
                print("Incorrect selection.")
    elif user_choice == "3":
        print("\nThe following items remain in the list:")
        for item in shopping_list:
            print(item)
        break
    else:
        print("Incorrect selection.")
