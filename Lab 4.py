import matplotlib.pyplot as mpl

x=[1,2,3,4,5]
y=[2,3,5,7,11]
mpl.plot(x,y,color="blue",marker="*",linestyle="dashed",linewidth=2,markersize=10)
mpl.xlabel('Test X axis')
mpl.ylabel('Test Y axis')
mpl.show()