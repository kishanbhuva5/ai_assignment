import numpy as np
a=np.array([1,2,3])
b=np.array([4,5,6])
c=np.array([7,8,9])

x=np.array([[[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]],[[1,2,3],[2,3,11]]]])
print(x.ndim)
print(x)
print(x.shape)
print(x[0][1][0][1])
# (1,3,2,3)==(1-->list/block, 3-->matrix, 2-->rows, 3--->column)



