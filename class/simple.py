"""
import np
# np.pi
# 3.141592653589793

import np as np
# np.pi
# 3.141592653589793

from np import *
# pi
# 3.141592653589793
"""
import numpy as np

# a = [[1,2], [3,4]]
# b = [[0,4], [1,1]]
# c = [[0,0], [0,0]]
# for x in range(2):
#     for y in range(2):
#         for z in range(2):
#             c[x][y] = c[x][y] + a[x][z]*b[z][y]
# print(c)


# x = np.floor(4.5)
# y = np.floor(-4.5)
# sum1 = x + y
# print(x, y, sum1)

# a = np.ceil(4.5)
# b = np.ceil(-4.5)
# sum2 = a + b
# print(a, b, sum2)

# z = np.array([[1,2], [3,4],[5,6]])
# print(z.ndim, z.shape, z.dtype)

# a = np.array([[1,2], [3,4]])
# b = np.array([[0,4], [1,1]])
# c = np.matmul(a, b)
# print(c)

a = np.random.randint(1, 100, 20)
print(a)
a.sort()
print(a)
print(np.unique)
