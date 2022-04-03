/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   DEBUG functions
------------------------------------------------------------------------- */

#ifndef CONP_NO_DEBUG

#include "debug_func.h"

#include <cmath>
#include <cstdio>
#include <unistd.h>

namespace LAMMPS_NS
{

void print_vec(double* arr, int N, const char* name)
{
  printf("  %s = ", name);
  for (int i = 0; i < N; ++i) {
    printf("%12.10g ", arr[i]);
  }
  printf("\n");
}

void print_vec(int* arr, int N, const char* name)
{
  printf("  %s = ", name);
  for (int i = 0; i < N; ++i) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

double norm_mat(double* mat, int N)
{
  double res = 0;
  for (int i = 0; i < N; ++i) {
    res += mat[i] * mat[i];
  }
  if (res < 0)
    return -sqrt(fabs(res));
  else
    return sqrt(res);
}

double norm_mat_p(double* mat, int N, const char* info)
{
  double value = norm_mat(mat, N);
  printf("%3s %.16g with size = %d\n", info, value, N);
  return value;
}

void write_mat(const double* const ptr, int N, const char* name)
{
  FILE* outa = fopen(name, "wb");
  printf("write as %s with size = %d\n", name, N);
  if (ptr == NULL) printf("the input ptr is NULL.\n");
  fwrite(ptr, sizeof(double), N, outa);
  fclose(outa);
}

double hash_vec(double* mat, int N)
{
  double res = 0, inv_N = 1 / double(N);
  for (int i = 0; i < N; ++i) {
    res += (mat[i] + i * inv_N) * (mat[i] + i * inv_N);
  }
  return sqrt(res);
}

double hash_vec(int* mat, int N)
{
  double res = 0, inv_N = 1 / double(N);
  for (int i = 0; i < N; ++i) {
    res += (mat[i] + i * inv_N) * (mat[i] + i * inv_N);
  }
  return sqrt(res);
}

double* load_mat(int N, const char* name)
{
  double* ptr  = (double*)malloc(sizeof(double) * N);
  FILE* infile = fopen(name, "r");
  int result   = fread(ptr, sizeof(double), N, infile);
  if (result != N) {
    printf("cannot read %d double from %s\n", N, name);
  }
  return ptr;
}

void print_hash(double* arr, int N, const char* name)
{
  printf("  %s = ", name);
  double hash   = hash_vec(arr, N);
  double normal = norm_mat(arr, N);
  printf("hash = %12.10g, Norm. = %12.10g Size = %d\n", hash, normal, N);
}

void eext(int n)
{
  printf("LOG: exit with %d\n", n);
  fflush(stdout);
  sleep(n);
  exit(0);
}

} // end namespace LAMMPS_NS
#endif