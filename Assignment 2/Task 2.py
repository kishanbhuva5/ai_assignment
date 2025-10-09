import numpy as np
import matplotlib.pyplot as mpl

x=np.array([1,2,3,4,5,6,7,8,9])
print(x)
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])
print(y)
mpl.figure(figsize=(7,5))
mpl.scatter(x, y, marker='+', color='r')
mpl.title("Scattering Pattern of Points (x, y)")
mpl.xlabel("x-axis")
mpl.ylabel("y-axis")
mpl.grid(True)
mpl.show()