/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   FUNC_MACRO: A General header for all classes
------------------------------------------------------------------------- */

#ifndef LMP_FUNC_MACRO_H
#define LMP_FUNC_MACRO_H

#ifndef CONP_NO_DEBUG
#include <cstdio>
#include <time.h>
#define FUNC_MACRO(X)                                                                                                  \
  static int FUNCTION_COUNT = 1;                                                                                       \
  {                                                                                                                    \
    if (X < DEBUG_LOG_LEVEL) {                                                                                         \
      time_t ltime;                                                                                                    \
      ltime = time(nullptr);                                                                                           \
      if (me == 0)                                                                                                     \
        printf("LOG: %s() [%d] times from %s:%d at %s", __FUNCTION__, FUNCTION_COUNT, __FILE__, __LINE__,              \
               asctime(localtime(&ltime)));                                                                            \
      FUNCTION_COUNT++;                                                                                                \
      fflush(stdout);                                                                                                  \
    };                                                                                                                 \
  };
#else
#define FUNC_MACRO(X) ;
#endif
#include <cstring>
template <class T>
inline void set_zeros(T* ptr, int value, size_t n)
{
  memset(ptr, value, n * sizeof(T));
}

template <class T>
inline void mul_mat(T* mat, T* b, T* c, size_t n)
{
  for (size_t i = 0; i < n, i++) {
    double value = 0.0;
    for (size_t j = 0; j < n, j++) {
      value += mat[0] * b[j];
      mat++;
    }
    c[i] = value;
  }
}

#if __cplusplus < 201103L // For C++11
#include <map>
using std::map;
#define MAP_RESERVE(X, Y) ;
#else
#include <unordered_map>

template <class T0, class T1>
using map = std::unordered_map<T0, T1>;
#define MAP_RESERVE(X, Y) X.reserve(Y);
// using std::unordered_map;
// typedef unordered_map map
#endif

// Comparing floating point numbers by Bruce Dawson is a good place to start
// when looking at floating point comparison.
#include <cmath>
inline bool isclose(double a, double b, double epsilon)
{
  return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

inline bool essentiallyEqual(double a, double b, double epsilon)
{
  return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

inline bool definitelyGreaterThan(double a, double b, double epsilon)
{
  return (a - b) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

inline bool definitelyLessThan(double a, double b, double epsilon)
{
  return (b - a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}
#endif
