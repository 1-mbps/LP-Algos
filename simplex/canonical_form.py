import numpy as np

def canonical_form(A: np.ndarray, b: np.ndarray, c: np.ndarray, basis: list[int]):

    # get basis columns of constraint matrix
    ab = A[:, basis]
    abinv = np.linalg.inv(ab)
    abinv_t = np.transpose(abinv)

    # get basis columns of c
    cb = c[basis]
    y = np.matmul(abinv_t, cb)
    y_t = np.transpose(y)

    ytA = np.matmul(y_t, A)

    new_c = np.transpose(np.transpose(c) - ytA)

    ytb = np.matmul(y_t, b)

    new_A = np.matmul(abinv, A)
    new_b = np.matmul(abinv, b)

    return new_A, new_b, new_c, ytb