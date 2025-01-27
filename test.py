import numpy as np

from simplex.simplex import simplex

# A = np.array([
#         [1,-2,1,1,0],
#         [1,-3,2,0,1]
#     ])

# b = np.array([[2], [3]])

# c = np.array([0,0,0,-1,-1])

# simplex(A, b, c, [4,5])

A = np.array([
    [1,-2,1,0,0],
    [0,5,-3,1,0],
    [0,4,-2,0,1]
])

b = np.array([1,1,2]).reshape(3,1)

c = np.array([0,-4,3,0,0])

simplex(A,b,c,[1,4,5])