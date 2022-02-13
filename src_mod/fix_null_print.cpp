/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.0.0 06/11/2019
   by Gengping JIANG @ WUST

   FixStyle(null_print,FixConpGA)  Fix_Name only avail for lowercase

   A NULL print FIX to as an alternative to other thermo output
------------------------------------------------------------------------- */

#include "fix_null_print.h"
#include "fix.h"

using namespace LAMMPS_NS;
using namespace FixConst;

int FixNullPrint::setmask()
{
  // printf("[FixNullPrint] setmask()\n");
  int mask = 0;
  // mask |= THERMO_ENERGY;
  return mask;
}

double FixNullPrint::compute_scalar() { return 0; }

double FixNullPrint::compute_vector(int n) { return double(n); }
