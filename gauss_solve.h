/*----------------------------------------------------------------
* File:     gauss_solve.h
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 15:40:51 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/

#ifndef GAUSS_SOLVE_H
#define GAUSS_SOLVE_H

void gauss_solve_in_place(const int n, double A[n][n], double b[n]);
void lu_in_place(const int n, double A[n][n]);
void lu_in_place_reconstruct(int n, double A[n][n]);
void plu(int n, double A[n][n], int P[n]);

#endif
