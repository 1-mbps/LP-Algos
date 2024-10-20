import numpy as np

from simplex.canonical_form import canonical_form
from simplex import simplex

A = np.array([
        [1,-2,1,1,0],
        [1,-3,2,0,1]
    ])

b = np.array([[2], [3]])

c = np.array([0,0,0,-1,-1])

simplex(A, b, c, [4,5])