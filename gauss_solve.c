/*----------------------------------------------------------------
* File:     gauss_solve.c
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 15:40:29 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/

#include <math.h>
#include "helpers.h"
void gauss_solve_in_place(const int n, double A[n][n], double b[n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k+1; i < n; ++i) {
      /* Store the multiplier into A[i][k] as it would become 0 and be
	 useless */
      A[i][k] /= A[k][k];
      for(int j = k+1; j < n; ++j) {
	      A[i][j] -= A[i][k] * A[k][j];
      }
      b[i] -= A[i][k] * b[k];
    }
  } /* End of Gaussian elimination, start back-substitution. */
  for(int i = n-1; i >= 0; --i) {
    for(int j = i+1; j<n; ++j) {
      b[i] -= A[i][j] * b[j];
    }
    b[i] /= A[i][i];
  } /* End of back-substitution. */
}

void lu_in_place(const int n, double A[n][n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	      /* U[k][i] -= L[k][j] * U[j][i] */
	      A[k][i] -=  A[k][j] * A[j][i]; 
      }
    }
    for(int i = k+1; i<n; ++i) {
      for(int j=0; j<k; ++j) {
        /* L[i][k] -= A[i][k] * U[j][k] */
        A[i][k] -= A[i][j]*A[j][k]; 
      }
      /* L[k][k] /= U[k][k] */
      A[i][k] /= A[k][k];	
    }
  }
}

void lu_in_place_reconstruct(int n, double A[n][n])
{
  for(int k = n-1; k >= 0; --k) {
    for(int i = k+1; i<n; ++i) {
      A[i][k] *= A[k][k];
      for(int j=0; j<k; ++j) {
	      A[i][k] += A[i][j]*A[j][k];
      }
    }
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	      A[k][i] +=  A[k][j] * A[j][i];
      }
    }
  }
}

void plu(int n, double A[n][n], int P[n]) {
    // Initialize permutation array P to identity
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }
    
    // Main loop over each column
    for (int k = 0; k < n - 1; k++) {
        // Find the pivot element
        double max = fabs(A[k][k]);
        int r = k;
        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i][k]) > max) {
                max = fabs(A[i][k]);
                r = i;
            }
        }

        // Swap rows in A and update permutation vector P
        if (r != k) {
            // Swap rows k and r in A
            for (int j = 0; j < n; j++) {
                double temp = A[k][j];
                A[k][j] = A[r][j];
                A[r][j] = temp;
            }

            // Swap entries in permutation vector P
            int tempP = P[k];
            P[k] = P[r];
            P[r] = tempP;

            // Swap elements in L (columns 0 to k-1)
            for (int j = 0; j < k; j++) {
                double temp = A[k][j];
                A[k][j] = A[r][j];
                A[r][j] = temp;
            }
        }

        // Compute the multipliers and update the matrix
        for (int i = k + 1; i < n; i++) {
            A[i][k] = A[i][k] / A[k][k]; // Multiplier for L
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j]; // Update U
            }
        }
    }
}