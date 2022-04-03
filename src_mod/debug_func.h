/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   DEBUG functions
------------------------------------------------------------------------- */

#ifndef CONP_GA_DEBUG_FUNC_H
#define CONP_GA_DEBUG_FUNC_H

#ifndef CONP_NO_DEBUG
namespace LAMMPS_NS
{

void print_vec(double* arr, int N, const char* name);
void print_vec(int* arr, int N, const char* name);
double norm_mat(double* mat, int N);
double norm_mat_p(double* mat, int N, const char* info);
void write_mat(const double* const ptr, int N, const char* name);
double hash_vec(double* mat, int N);
double hash_vec(int* mat, int N);
double* load_mat(int N, const char* name);
void print_hash(double* arr, int N, const char* name);
void eext(int n);

} // namespace LAMMPS_NS
#endif

#endif