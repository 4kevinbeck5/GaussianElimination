/*----------------------------------------------------------------
* File:     helpers.h
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 17:52:09 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/

#ifndef PRINT_HELPERS_H
#define PRINT_HELPERS_H

#define FLAG_WHOLE 0
#define FLAG_LOWER_PART 1
#define FLAG_UPPER_PART 2

void   matrix_times_vector(int n, const double A[n][n], const double x[n], double y[n]);
double norm(const int n, const double x[n]);
double norm_dist(const int n, const double x[n], const double y[n]);
double frobenius_norm(const int n, const double X[n][n]);
double frobenius_norm_dist(const int n, const double X[n][n], const double Y[n][n]);

void   print_vector(int n, double x[n]);
void   print_matrix(int n, double A[n][n], int flag);
void   generate_random_matrix(int n, double matrix[n][n]);
void   create_matrix(int n, double (**matrix)[n]);
void   destroy_matrix(int n, double (*matrix)[n]);
void   copy_matrix(int n, const double A[n][n], double A_copy[n][n]);

#endif
