import numpy as np
from numpy.matrixlib.defmatrix import matrix

arr=np.zeros(10)
print(arr)
arr1=np.ones(10)
print(arr1)
arr2=np.full((10,10),11)
print(arr2)
a=np.linspace(0,5,4)
print(a)
b=np.arange(0,5,0.2)
print(b)
c=np.random.random(10)
print(c)
d=np.random.randint(1,10,10)
print(d)
e=np.random.randn(10)#range between 0-3 both signed and unsigned
print(e)
f=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print(f)
resh=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print(resh.reshape(3,4))
print(resh.reshape(2,2,3))
x=resh.reshape(2,2,3)
print(x.shape)
g=np.repeat([[1,1,1]],4,axis=1) #0 means row and 1 means column
print(g)
h=np.array([[1,2,3],[4,5,6]])
print(h)

for i in range(2):
    for j in range(3):
        print(f"Matrix [{i}]{j}]:{h[i][j]}")





