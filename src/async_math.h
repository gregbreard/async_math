// Check if we will be using double or floats
#define USE_DOUBLE false
#if USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

#ifndef ASYNC_MATH_H_
#define ASYNC_MATH_H_

class AsyncMath {
public:
  static void MatrixMult (real_t* a, const int* a_dim, real_t* b, const int* b_dim,
                   real_t *& c, int *& c_dim);

private:
  static void load_kernel();
  static void release_kernel();

}; // end AsyncMath

#endif  // ASYNC_MATH_H_
