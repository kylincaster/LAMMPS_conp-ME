/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   pppm_la_conp/GA: PPPM class to perform pppm calculation for Conp/GA
   with the intermediate matrix implementation
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

// clang-format off
KSpaceStyle(pppm_la_conp/ME, PPPM_LA_conp_GA)
// clang-format on

#else

#ifndef LMP_PPPM_LA_CONP_GA_H
#define LMP_PPPM_LA_CONP_GA_H

#include "kspace.h"
#include "kspace_conp_GA.h"
#include "pppm_conp_GA.h"

#if defined(FFT_FFTW3)
#define LMP_FFT_LIB "FFTW3"
#elif defined(FFT_MKL)
#define LMP_FFT_LIB "MKL FFT"
#elif defined(FFT_CUFFT)
#define LMP_FFT_LIB "cuFFT"
#else
#define LMP_FFT_LIB "KISS FFT"
#endif

#ifdef FFT_SINGLE
// typedef float FFT_SCALAR;
#define LMP_FFT_PREC "single"
// #define MPI_SINGLE MPI_FLOAT
// #define MPI_FFT_SCALAR MPI_FLOAT
#else

// typedef double FFT_SCALAR;
#define LMP_FFT_PREC "double"
// #define MPI_FFT_SCALAR MPI_DOUBLE
#endif

namespace LAMMPS_NS
{

class PPPM_LA_conp_GA : public PPPM_conp_GA {
public:
  PPPM_LA_conp_GA(class LAMMPS*);
  virtual ~PPPM_LA_conp_GA();
  virtual void setup_FIX(int);
  virtual const char* type_name() { return "pppm_la_conp/GA"; };
  virtual void settings(int narg, char** arg);
  virtual bool is_compute_AAA() { return true; }
  virtual double** compute_AAA(double** mat);
  void folder(FFT_SCALAR* dst, FFT_SCALAR* src, char* occ);

protected:
  FFT_SCALAR** gw;
  FFT_SCALAR** compute_intermediate_matrix(bool);
  virtual void calc_GA_POT(double, bool);
  FFT_SCALAR* b_part;
  bool matrix_flag, matrix_saved;
  // void setup_nxyz();
  /*
  int nxhi_conp, nxlo_conp;
  int nyhi_conp, nylo_conp;
  int nzhi_conp, nzlo_conp;
  int nx_conp, ny_conp, nz_conp, nxyz_conp;
  */
};

} // namespace LAMMPS_NS

#endif
#endif