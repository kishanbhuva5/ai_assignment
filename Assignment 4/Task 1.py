import numpy as np
import matplotlib.pyplot as plt

value_of_n=[500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in value_of_n:
    dice1 = np.random.randint(1, 7, size=n)
    dice2 = np.random.randint(1, 7, size=n)
    s = dice1 + dice2
    h, h2 = np.histogram(s, bins=np.arange(2, 14))
    plt.bar(h2[:-1], h / n)
    plt.title(f"Histogram of sums of two dice (n = {n})")
    plt.xlabel("Sum of dice")
    plt.ylabel("Relative frequency")
    plt.show()
