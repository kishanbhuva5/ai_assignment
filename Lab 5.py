import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# x=np.linspace(0,10,10)  #"b" gives the slope of line while "a"  gives the y-intercept
# y=1 + 2*x #y=a+bx
# plt.figure(figsize=(10,6))
# plt.plot(x,y,color="red",marker="*",linestyle="solid",linewidth=2,markersize=10,label='y=1+2x')
# plt.title("Graph of y=1+2x")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.grid(True)
# plt.legend()
# plt.show()

# file_path = "linreg_data.csv"
# df = pd.read_csv(file_path)
# df=np.array(df)
# xd=df[:,0]
# yd=df[:,1]
#
# pdxy=xd*yd
# print(pdxy)
#
#
# plt.figure(figsize=(7,5))
# plt.scatter(xd, yd, marker='o', color='orange')
# plt.title("Scattering Pattern of Points (x, y)")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.show()

import numpy as np
import pandas as pd

df = pd.read_csv('linreg_data.csv',names=['x','y'], skiprows=0)
givenX = df['x']
givenY = df['y']

XY = givenX * givenY

meanOfXY = np.mean(XY)

meanOfX = np.mean(givenX)
meanOfY = np.mean(givenY)

squareOfX = np.square(givenX)

n = len(givenX)

b = (np.sum(XY) - n * meanOfX * meanOfY) / (np.sum(squareOfX) - n * (meanOfX ** 2))
a = meanOfY - b * meanOfX

print("Slope (b):", b)
print("Intercept (a):", a)



yhat = a+b*givenX

RSS = np.sum((givenY-yhat)**2)

print("RSS:", RSS)

RMSE = np.sqrt((1/n)*np.sum((givenY-yhat)**2))

print("Root Mean Squared Error:", RMSE)

MAE = (1/n)*np.sum(np.abs(givenY-yhat))

print("Mean Absolute Error:", MAE)

MSE = np.sum((givenY-yhat)**2)/n

print("Mean Squared Error:", MSE)

R2 = 1 - np.sum((givenY-yhat)**2)/np.sum((givenY-meanOfY)**2)

print("R2:", R2)


