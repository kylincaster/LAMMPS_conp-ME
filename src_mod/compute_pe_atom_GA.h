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

   ComputeStyle("pe/atom/GA", ComputePEAtomGA)
   A Potential energy computer with the support of output
   short range Coulomb energy.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

// clang-format off
ComputeStyle(pe/atom_GA, ComputePEAtomGA)
// clang-format on

#else

#ifndef LMP_COMPUTE_PE_ATOM_GA_H
#define LMP_COMPUTE_PE_ATOM_GA_H

#include "compute.h"
#include "pair_conp_GA.h"

namespace LAMMPS_NS
{

class ComputePEAtomGA : public Compute {
public:
  ComputePEAtomGA(class LAMMPS*, int, char**);
  ~ComputePEAtomGA();
  void init() {}
  void compute_peratom();
  int pack_reverse_comm(int, int, double*);
  void unpack_reverse_comm(int, int*, double*);
  double memory_usage();

private:
  Pair_CONP_GA* pair;
  int pairflag, bondflag, angleflag, dihedralflag, improperflag;
  int kspaceflag, cl_pairflag, fixflag;
  int nilflag;
  int nmax;
  double* energy;
};

} // namespace LAMMPS_NS

#endif
#endif

    /* ERROR/WARNING messages:

    E: Illegal ... command

    Self-explanatory.  Check the input script syntax and compare to the
    documentation for the command.  You can use -echo screen as a
    command-line option when running LAMMPS to see the offending line.

    E: Per-atom energy was not tallied on needed timestep

    You are using a thermo keyword that requires potentials to
    have tallied energy, but they didn't on this timestep.  See the
    variable doc page for ideas on how to make this work.

    */
