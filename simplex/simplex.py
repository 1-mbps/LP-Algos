import numpy as np

from simplex.canonical_form import canonical_form

def has_solution(A, b):
    # Ensure A is a 2D array and b is a 1D array
    A = np.atleast_2d(A)
    b = np.atleast_1d(b)
    
    # Check if dimensions are compatible
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimensions of A and b are not compatible")
    
    # Create the augmented matrix [A|b]
    augmented = np.column_stack((A, b))
    
    # Calculate the ranks
    rank_A = np.linalg.matrix_rank(A)
    rank_augmented = np.linalg.matrix_rank(augmented)
    
    # Check if the ranks are equal
    return rank_A == rank_augmented


def simplex_solver(A, b, c, basis: list[int], z: int, iterations=1):
    """
    Perform the simplex algorithm on the linear program max {c^T x + z : Ax = b, x ≥ 0}.
    """

    new_A, new_b, new_c, new_z = canonical_form(A, b, c, basis)
    new_z = new_z[0]
    new_z += z

    if all(new_c <= 0):
        print(f"Optimal value found: {new_z}\n")
        print(f"A:\n{new_A}\n\n b:\n{new_b} \n\n c:\n{new_c} \n\n z:\n{new_z}")

        ab = A[:, basis]
        abinv = np.linalg.inv(ab)

        # get basis columns of c
        cb = c[basis]

        cert = np.matmul(np.transpose(cb), abinv)

        print(f"Certificate of optimality: {cert}")
    else:

        print(f"\n================SIMPLEX ITERATION {iterations}================\n")

        print(f"Canonical form:\nA:\n{new_A}\n\n b:\n{new_b} \n\n c:\n{new_c} \n\n z:\n{new_z}\n")

        # ----- Choose new basis using Bland's rule, which guarantees termination -----

        # Choose entering variable
        k = -1
        for i in range(len(new_c)):
            if new_c[i] > 0:
                k = int(i)
                break

        if all(new_A[:, k]) <= 0:
            print("LP is unbounded.")

            x = np.array([0]*len(new_c))
            d = np.array([0]*len(new_c))

            for index, value in enumerate(basis):
                x[value] = new_b[index][0]
                d[value] = -new_A[:, k][index]

            d[k] = 1
            print(f"Certificate of unboundedness: x = {x}, d = {d}. All x(t) = x + td (t ≥ 0) are feasible.")
                
        else:
            print(f"Performing Bland's rule...\n")
            print(f"Entering variable: k = {k+1}.\nFinding the leaving variable...\n")

            # Choose i (1 ≤ i ≤ |B|) that minimizes the value of t = b_i / (A_B)_{ik}
            # in case of ties, choose smaller i
            # then leaving variable is i-th element of B

            l = -1
            t = 1000000000
            for i in range(len(basis)):
                if new_A[i, k] <= 0:
                    print(f"i = {i+1} - A_ik = {new_A[i, k]} ≤ 0 - unusable")
                else:
                    ratio = b[i][0]/new_A[i, k]
                    print(f"i = {i+1} - A_ik = {new_A[i, k]} - Ratio: {b[i][0]}/{new_A[i, k]} = {ratio}")
                    if ratio < t:
                        l = basis[i]
                        t = ratio

            print(f"\nMinimum ratio was {t}.")
            print(f"Leaving variable: element {l+1} of original basis {[b+1 for b in basis]} -> {l+1}")

            # Remove leaving variable
            basis.remove(l)

            # Add entering variable
            basis.append(k)
            
            basis.sort()
            print(f"New basis: {[b+1 for b in basis]}\n")

            simplex_solver(new_A, new_b, new_c, basis, new_z, iterations=iterations+1)

def simplex(A, b, c, basis: list[int], z=0):
    """
    Perform the simplex algorithm on the linear program max {c^T x + z : Ax = b, x ≥ 0}.

    Args:
        A: m x n constraint matrix
        b: m x 1 vector
        c: 1 x n vector
        basis: the initial basis to use. Your basis must be 1-indexed.
        z: constant (real number)
    """

    if not has_solution(A, b):
        print("LP is infeasible.")

    basis = [b-1 for b in basis]

    simplex_solver(A, b, c, basis, z)
