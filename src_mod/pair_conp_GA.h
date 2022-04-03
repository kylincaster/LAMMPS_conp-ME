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

   Pair_CONP_GA: The General Class for pair_style used in Conp/GA
------------------------------------------------------------------------- */

#ifndef LMP_PAIR_CONP_GA_H
#define LMP_PAIR_CONP_GA_H
#include "lmptype.h"

namespace LAMMPS_NS
{
class Electro_Control_GA;
class Pair_CONP_GA {
protected:
public:
  // virtual const char* type_name() { return ""; };
  virtual double pair_SOLtoGA_energy(int, bool, bool) { return 0.0; };
  virtual double pair_GAtoGA_energy(int, bool) { return 0.0; };
  virtual double compute_single(int, int, double) { return 0.0; };
  virtual void compute_Group(int, int, int, bool){};
  virtual void clear_saved(){};
  /*
  virtual const char* type_name() = 0;
  virtual double get_qsum() = 0;
  virtual double get_qsqsum() = 0;
  virtual void set_qsqsum(double, double) = 0;
  virtual void compute_GAtoGA() = 0; // ＝0;
  virtual void setup_FIX() = 0;// = 0;
  virtual void compute_GAtoSOL()=0;
   */
  static double* cl_atom;
  Electro_Control_GA* electro;
  int DEBUG_LOG_LEVEL;
  int me;
  int compute_id;
  bigint sv_SOL_timestep;
  bool sv_GA_flag, sv_SOL_flag, sv_SOL_saved;
  Pair_CONP_GA();
  virtual ~Pair_CONP_GA(){};
}; // class Electro_Control_GA {int* FIX;};

enum class CHARGE : int { POINT, GAUSSIAN };
enum class STAT : int { E, POT };
// if SUM_OP == OFF, just convert no further operation
// if SUM_OP == ON,  copy the converted value to ek_atom
// if SUM_OP == BOTH, save both energy(cl_atom) and potential (ek_atom)
enum class SUM_OP : int { OFF, ON, BOTH };
// enum { POT_TO_E = -1, NIL, E_TO_POT };

// enum for fix conp/GA
enum class CELL : int { NONE, POT, Q };

enum class PCG_MAT : int { FULL, NORMAL, DC };
enum class SYM : int { NONE, MEAN, SQRT, TRAN };

// enum for the control of individual electrode
enum class CONTROL : int { NONE, METAL, CHARGED };

enum class TOL : int { REL_B, MAX_Q, RES, PE, STD_Q, STD_RQ, ABS_Q, NONE };
enum class varTYPE : int { NONE, CONST, EQUAL, ATOM };

enum class MIN : int {
  NONE,
  INV,
  CG,
  PCG,
  PCG_EX,
  NEAR,
  FAR,
};
} // namespace LAMMPS_NS
#endif
