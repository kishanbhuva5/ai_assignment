product_price=[10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
total_sum=0
print("Supermarket\n===========")

while True:
    user_choice = int(input("Please select product (1-10) 0 to Quit): "))
    if user_choice == 0:
        break
    elif 1 <= user_choice <= 10:
        price = product_price[user_choice - 1]
        total_sum += price
        print(f"Product: {user_choice} Price: {price}")
    else:
        print("Incorrect selection.")

print(f"Total: {total_sum}")

payment = int(input("Payment: "))
change = payment - total_sum
print(f"Change: {change}")