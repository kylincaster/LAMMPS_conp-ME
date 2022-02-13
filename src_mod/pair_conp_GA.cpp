/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   Pair_CONP_GA: The General Class for pair_style used in Conp/GA
------------------------------------------------------------------------- */
#include "pair_conp_GA.h"
#include <cstddef>

using namespace LAMMPS_NS;

double* Pair_CONP_GA::cl_atom = NULL;
Pair_CONP_GA::Pair_CONP_GA() : electro(NULL)
{
  DEBUG_LOG_LEVEL = 0;
  compute_id      = -1;
  sv_GA_flag      = false;
  sv_SOL_flag     = false;
  sv_SOL_saved    = false;
  sv_SOL_timestep = -1;
  electro         = NULL;
  me              = -1;
}