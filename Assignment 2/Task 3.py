import pandas as pd
import matplotlib.pyplot as mpl
import numpy as np

file_path = "weight-height.csv"
df = pd.read_csv(file_path)

length_in = df["Height"].values   # Heights in inches
weight_lb = df["Weight"].values   # Weights in pounds

length_cm = length_in * 2.54
weight_kg = weight_lb * 0.453592
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean length: {mean_length:.2f} cm")
print(f"Mean weight: {mean_weight:.2f} kg")

mpl.figure(figsize=(7,5))
mpl.hist(length_cm, bins=20, color='skyblue', edgecolor='black')
mpl.axvline(mean_length, color='red', linestyle='dashed', label=f"Mean = {mean_length:.2f} cm")
mpl.title("Histogram of Lengths (cm)")
mpl.xlabel("Length (cm)")
mpl.ylabel("Frequency")
mpl.legend()
mpl.grid(True, linestyle='solid', alpha=0.5)

mpl.show()