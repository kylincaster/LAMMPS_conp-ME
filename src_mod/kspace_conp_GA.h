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

   KSpaceConpGA: A General Kspace style for Conp/GA
------------------------------------------------------------------------- */

#ifndef LMP_KSPACE_CONP_GA_H
#define LMP_KSPACE_CONP_GA_H

#include "lmptype.h"
#include <cstddef>

namespace LAMMPS_NS
{
class Electro_Control_GA;
class KSpaceConpGA {
public:
  virtual const char* type_name() { return ""; };
  virtual double get_qsum() { return 0; };
  virtual double get_qsqsum() { return 0; };
  virtual void set_qsqsum(double, double){};
  virtual void compute_GAtoGA(){}; // ＝0;
  virtual void compute_SOLtoGA(bool){};
  virtual void setup_FIX(int) { printf("wrong _FIX"); }; // = 0;
  virtual bool is_compute_AAA() { return false; }
  virtual double** compute_AAA(double** mat) { return nullptr; }
  void set_electro(Electro_Control_GA* electro0) { electro = electro0; }
  void set_debug_level(int level) { DEBUG_LOG_LEVEL = level; }
  virtual void clear_saved() {}
  /*
 virtual const char* type_name() = 0;
 virtual double get_qsum() = 0;
 virtual double get_qsqsum() = 0;
 virtual void set_qsqsum(double, double) = 0;
 virtual void compute_GAtoGA() = 0; // ＝0;
 virtual void setup_FIX() = 0;// = 0;
 virtual void compute_GAtoSOL()=0;
  */
  int DEBUG_LOG_LEVEL;
  Electro_Control_GA* electro;
  int FIX_groupbit0;

  KSpaceConpGA() : electro(nullptr) { DEBUG_LOG_LEVEL = 0; };
  virtual ~KSpaceConpGA(){}; // printf("~KSpaceConpGA()\n"); exit(0);};
  /*
  virtual void compute_vector(bigint *, double *) = 0;
  virtual void compute_vector_corr(bigint *, double *) = 0;
  virtual void compute_matrix(bigint *, double **) = 0;
  virtual void compute_matrix_corr(bigint *, double **) = 0;
  */
};
} // namespace LAMMPS_NS

#endif