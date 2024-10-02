/*----------------------------------------------------------------
 * File:     helpers.c
 *----------------------------------------------------------------
 *
 * Author:   Marek Rychlik (rychlik@arizona.edu)
 * Date:     Sun Sep 22 15:46:13 2024
 * Copying:  (C) Marek Rychlik, 2020. All rights reserved.
 *
 *----------------------------------------------------------------*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


#include "helpers.h"

void matrix_times_vector(int n, const double A[n][n], const double x[n],
			 double y[n])
{
  for(int i = 0; i < n; ++i) {
    y[i] = 0;
    for(int j = 0; j < n; ++j)
      y[i] += A[i][j] * x[j];
  }
}

double norm(const int n, const double x[n])
{
  double S = 0;
  for(int j = 0; j < n; ++j) {
    S += x[j] * x[j];
  }
  return sqrt(S);
}

double norm_dist(const int n, const double x[n], const double y[n])
{
  double S = 0;
  for(int j = 0; j < n; ++j) {
    S += (x[j]-y[j]) * (x[j]-y[j]);
  }
  return sqrt(S);
}

double frobenius_norm(const int n, const double X[n][n])
{
  double S = 0;
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      S += X[i][j] * X[i][j];
    }
  }
  return sqrt(S);
}

double frobenius_norm_dist(const int n, const double X[n][n], const double Y[n][n])
{
  double S = 0;
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      S += (X[i][j]-Y[i][j]) * (X[i][j]-Y[i][j]);
    }
  }
  return sqrt(S);
}

void print_vector(int n, double x[n])
{
  for(int i=0; i<n; ++i) {
    printf("%g\n", x[i]);
  }
  puts("\n");
}

void print_matrix(int n, double A[n][n], int flag)
{
  switch(flag) {
  case FLAG_WHOLE:
    for(int i = 0; i<n; ++i) {
      for(int j = 0; j<n; ++j) {
	printf("%12.6g\t", A[i][j]);
      }
      puts("\n");
    }
    break;

  case FLAG_LOWER_PART:
    for(int i = 0; i<n; ++i) {
      for(int j = 0; j<i; ++j) {
	printf("%12.6g\t", A[i][j]);
      }
      printf("%12.6g\t", 1.0);
      for(int j = i+1; j<n; ++j) {
	printf("%12.6g\t", 0.0);
      }
      puts("\n");
    }
    break;

  case FLAG_UPPER_PART:
    for(int i = 0; i<n; ++i) {
      for(int j = 0; j<i; ++j) {
	printf("%12.6g\t", 0.0);
      }
      for(int j = i; j<n; ++j) {
	printf("%12.6g\t", A[i][j]);
      }
      puts("\n");
    }
    break;
  }

  puts("\n");
}


// Function to generate a random square matrix of size 'n'
void generate_random_matrix(int n, double matrix[n][n])
{
  // Seed the random number generator with the current time
  srand(time(0));
    
  // Fill the matrix with random integers between 0 and 99
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i][j] = rand() % 100;
    }
  }
}

void create_matrix(int n, double (**matrix)[n])
{
  double *store = (double *)malloc(n * n * sizeof(double));
  double (*A)[n] = (double (*)[n])store;
  *matrix = A;
}

void destroy_matrix(int n, double (*matrix)[n])
{
  double *store = (double *)matrix;
  free(store);
}
     
void copy_matrix(int n, const double A[n][n], double A_copy[n][n])
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A_copy[i][j] = A[i][j];
    }
  }
}
