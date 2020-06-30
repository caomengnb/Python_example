import numpy as np
from numpy import *

r0=5
c0=5
state = ones((r0, c0))
b=[1,2,3,4,5]
print(state*b)
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a
s=rand_arr(-0.1,0.1,1,5)
ssss=[[1,2,3],
      [4,5,6],
      [7,8,9]]
a=2
b=str(a)+'正确'
print(s[:,2])

