import matplotlib.pyplot as mpl
import numpy as np

x = np.linspace(0, 10, 10)

y1=2*x+1
y2=2*x+2
y3=2*x+3

mpl.figure(figsize=(10,6))
mpl.plot(x,y1,color="red",marker="*",linestyle="solid",linewidth=2,markersize=10,label='y=2x+1')
mpl.plot(x,y2,color="green",marker="o",linestyle="solid",linewidth=2,markersize=10,label='y=2x+2')
mpl.plot(x,y3,color="blue",marker="s",linestyle="solid",linewidth=2,markersize=10,label='y=2x+3')

mpl.title("Graphs of y=2x+1, y=2x+2, y=2x+3")
mpl.xlabel("x-axis")
mpl.ylabel("y-axis")

mpl.grid(True)
mpl.legend()
mpl.show()