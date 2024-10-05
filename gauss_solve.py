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

 
def lu(A, use_c=False):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    n = len(A)
    if use_c:
        # Create a 2D array in Python and flatten it
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
            [modified_array_2d[i][j] for j in range(i)] + [1] + [0 for j in range(i+1,n)]
            for i in range(n)
        ]

        U = [
            [ 0 for j in range(i) ] + [ modified_array_2d[i][j] for j in range(i, n) ]
            for i in range(n)
        ]
    else:
        for k in range(n):
            if k == 0:
                U[k, k:] = A[k, k:]
            else:
                U[k, k:] = A[k, k:] - np.dot(L[k, :k], U[:k, k:])
            L[k, k] = 1.0
            if k < n - 1:
                if k == 0:
                    L[k+1:, k] = A[k+1:, k] / U[k, k]
                else:
                    L[k+1:, k] = (A[k+1:, k] - np.dot(L[k+1:, :k], U[:k, k])) / U[k, k]
        L = L.tolist()
        U = U.tolist()
        # # Initialize matrices
        # L = np.zeros((n,n))
        # U = np.zeros((n,n))

        # for k in range(n):
        #     for i in range(k, n):
        #         U[k, i] = A[k][i] - np.sum(L[k,j]*U[j,i] for j in range(k))
        #     for i in range(k+1, n):
        #         L[i, k] = (A[i][k] - np.sum(L[i,j]*U[j,k] for j in range(k)))/U[k, k]
        #     L[k, k] = 1
        # L = L.tolist()
        # U = U.tolist()
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
            [modified_array_2d[i][j] for j in range(i)] + [1] + [0 for j in range(i+1,n)]
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
    
    L1, U1 = lu(A, use_c=False)

    P, L, U = plu(A, use_c=True)
    