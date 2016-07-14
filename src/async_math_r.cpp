#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <stdlib.h>
#include "async_math.h"

// Include the stuff for R
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// [[Rcpp::export]]
arma::mat mat_mult(const arma::mat a, const arma::mat b) {
  arma::mat c(a.n_rows, b.n_cols);

  // Get the dimensions
  const int a_cols = a.n_cols;
  const int a_rows = a.n_rows;
  const int b_cols = b.n_cols;
  const int b_rows = b.n_rows;

  // Create arrays for the data
  real_t *a_rl = new real_t[a_rows * a_cols];
  real_t *b_rl = new real_t[b_rows * b_cols];

  // Copy the data to arrays
  for (int i = 0; i < fmax(a_rows, b_rows); i++) {
    for (int j = 0; j < fmax(a_cols, b_cols); j++) {
      if (i < a_rows && j < a_cols)
        a_rl[(i * a_cols) + j] = (real_t)a(i, j);
      if (i < b_rows && j < b_cols)
        b_rl[(i * b_cols) + j] = (real_t)b(i, j);
    } // end for (i)
  } // end for (i)

  // Initialize dimensions
  int a_dim[] = {a.n_rows, a.n_cols};
  int b_dim[] = {b.n_rows, b.n_cols};

  // Initialize return pointers
  real_t *c_db = NULL;
  int *c_dim = NULL;

  clock_t begin = clock();

  // Call multiplication function
  AsyncMath::MatrixMult(a_rl, a_dim, b_rl, b_dim, c_db, c_dim);

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  Rcout << "time: " << time_spent << std::endl;

  // Copy data into output matrix
  for (int i = 0; i < c.n_rows; i++)
    for (int j = 0; j < c.n_cols; j++)
      c(i, j) = c_db[(i * c.n_cols) + j];

  // Clean up
  delete [] a_rl;
  delete [] b_rl;

  return c;
} // end mat.mult
