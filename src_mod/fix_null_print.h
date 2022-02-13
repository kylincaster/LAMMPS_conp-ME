/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.0.0 06/11/2019
   by Gengping JIANG @ WUST

   FixStyle(null_print,FixConpGA)  Fix_Name only avail for lowercase

   A NULL print FIX to as an alternative to other thermo output
------------------------------------------------------------------------- */

/* to print zero as thermo output */

#ifdef FIX_CLASS

// clang-format off
FixStyle(null_print, FixNullPrint)
// clang-format on

#else

#ifndef LMP_FIX_NULL_PRINT_H

#define LMP_FIX_NULL_PRINT_H

#include "fix.h"
#include <ctime>
#include <map>
#include <vector>

namespace LAMMPS_NS
{

class FixNullPrint : public Fix {
public:
  int me;
  int setmask();
  double compute_scalar();
  double compute_vector(int);
  FixNullPrint(LAMMPS* lmp, int narg, char** arg) : Fix(lmp, narg, arg)
  {
    scalar_flag = 1;
    vector_flag = 1;
    size_vector = 10000; // Number of thermo output vector
  };
  ~FixNullPrint(){};
  // double compute_scalar();
  // double compute_vector(int n);
};

}; // namespace LAMMPS_NS

#endif
#endif
