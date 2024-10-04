#----------------------------------------------------------------
# File:     gauss_solve.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Thu Sep 26 10:38:32 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# A Python wrapper module around the C library libgauss.so

import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libgauss.so')

 
def lu(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Modify the array in C (e.g., add 10 to each element)
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    L = [
        [[modified_array_2d[i][j] for j in range(i)] + [1] + [0 for j in range(i+1,n)]]
        for i in range(n)
    ]

    U = [
        [ 0 for j in range(i) ] + [ modified_array_2d[i][j] for j in range(i, n) ]
        for i in range(n)
    ]

    return L, U

def plu(A, use_c=False):
    """
    Accepts list of lists A of floats and returns

    P: list of integers (permutation)
    L: lower triangular list of lists
    U: upper triangular list of lists
    """
    n = len(A)

    if use_c:
        flat_array_2d = [item for row in A for item in row]
        A_c_version = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)
        P_c_version = (ctypes.c_int * n)()

        # Define function signature
        lib.plu.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))

        # Do PA=LU in C to c_array_2d
        lib.plu(n, A_c_version, P_c_version)

        # Convert A back to a 2D Python list of lists
        modified_array_2d = [
            [A_c_version[i * n + j] for j in range(n)]
            for i in range(n)
        ]

        # Extract L and U parts from A, fill with 0's and 1's
        L = [
            [[modified_array_2d[i][j] for j in range(i)] + [1] + [0 for j in range(i+1,n)]]
            for i in range(n)
        ]

        U = [
            [ 0 for j in range(i) ] + [ modified_array_2d[i][j] for j in range(i, n) ]
            for i in range(n)
        ]
        
        P = list(P_c_version)
    # Use Python version
    else:
        # Initialize matrices
        P_np = np.linspace(0, n-1, n)
        L = np.zeros(shape=(n,n))
        U = np.array(A)

        for k in range(n-1):
            # Find the row index r of the pivot element
            r = np.argmax(np.abs(U[k:, k])) + k
            
            # Make row/element swaps
            U[[k, r]] = U[[r, k]]
            P_np[r], P_np[k] = P_np[k], P_np[r]
            L[[k, r], 0:k] = L[[r, k], 0:k] 

            for i in range(k+1, n):
                L[i, k] = (U[i, k] / U[k, k])
                U[i] = U[i] - L[i, k] * U[k]
        
        # Add ones to diagonal of L and convert matrices to correct return form
        L = (L + np.eye(n)).tolist()
        P = P_np.astype(int).tolist()
        U = U.tolist()
        
    return P, L, U

if __name__ == "__main__":

    A = [[4.0, 9.0, 10.0],
         [14.0, 30.0, 34.0],
         [2.0, 3.0, 3.0]];

    P, L, U = plu(A, use_c=True)
