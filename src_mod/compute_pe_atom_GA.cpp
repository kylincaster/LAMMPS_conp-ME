/* -*- c++ -*- -------------------------------------------------------------
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

#include "compute_pe_atom_GA.h"
#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "update.h"
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePEAtomGA::ComputePEAtomGA(LAMMPS* lmp, int narg, char** arg)
    : Compute(lmp, narg, arg), energy(nullptr), pair(nullptr)
{
  if (narg < 3) error->all(FLERR, "Illegal compute pe/atom command");

  peratom_flag      = 1;
  size_peratom_cols = 0;
  peatomflag        = 1;
  timeflag          = 1;
  comm_reverse      = 1;

  if (narg == 3) {
    pairflag = 1;
    bondflag = angleflag = dihedralflag = improperflag = 1;
    kspaceflag                                         = 1;
    cl_pairflag                                        = 0;
    fixflag                                            = 1;
  } else {
    pairflag = 0;
    bondflag = angleflag = dihedralflag = improperflag = 0;
    cl_pairflag = kspaceflag = 0;
    fixflag                  = 0;
    nilflag                  = 0;
    int iarg                 = 3;
    while (iarg < narg) {
      if (strcmp(arg[iarg], "pair") == 0)
        pairflag = 1;
      else if (strcmp(arg[iarg], "bond") == 0)
        bondflag = 1;
      else if (strcmp(arg[iarg], "angle") == 0)
        angleflag = 1;
      else if (strcmp(arg[iarg], "dihedral") == 0)
        dihedralflag = 1;
      else if (strcmp(arg[iarg], "improper") == 0)
        improperflag = 1;
      else if (strcmp(arg[iarg], "coul") == 0)
        cl_pairflag = 1;
      else if (strcmp(arg[iarg], "kspace") == 0)
        kspaceflag = 1;
      else if (strcmp(arg[iarg], "fix") == 0)
        fixflag = 1;
      else if (strcmp(arg[iarg], "__NIL_COMPUTER__") == 0)
        nilflag = 1;
      else
        error->all(FLERR, "Illegal compute pe/atom command");
      iarg++;
    }
  }

  pair = dynamic_cast<Pair_CONP_GA*>(force->pair);
  if (cl_pairflag) {
    if (pair == nullptr) {
      error->all(FLERR, "The calculation of Coulomb energy in real space is "
                        "only support for pair_style in Conp/GA package");
    }
  }
  if (nilflag) {
    comm_reverse = 0;
  }
  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputePEAtomGA::~ComputePEAtomGA() { memory->destroy(energy); }

/* ---------------------------------------------------------------------- */

void ComputePEAtomGA::compute_peratom()
{
  if (nilflag) return;
  int i;
  // printf("invoke compute_peratom\n");
  invoked_peratom = update->ntimestep;
  if (update->eflag_atom != invoked_peratom) error->all(FLERR, "Per-atom energy was not tallied on needed timestep");

  // grow local energy array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(energy);
    nmax = atom->nmax;
    memory->create(energy, nmax, "pe/atom:energy");
    vector_atom = energy;
  }

  // npair includes ghosts if either newton flag is set
  //   b/c some bonds/dihedrals call pair::ev_tally with pairwise info
  // nbond includes ghosts if newton_bond is set
  // ntotal includes ghosts if either newton flag is set
  // KSpace includes ghosts if tip4pflag is set

  int nlocal  = atom->nlocal;
  int npair   = nlocal;
  int nbond   = nlocal;
  int ntotal  = nlocal;
  int nkspace = nlocal;
  if (force->newton) npair += atom->nghost;
  if (force->newton_bond) nbond += atom->nghost;
  if (force->newton) ntotal += atom->nghost;
  if (force->kspace && force->kspace->tip4pflag) nkspace += atom->nghost;

  // clear local energy array

  for (i = 0; i < ntotal; i++)
    energy[i] = 0.0;

  // add in per-atom contributions from each force

  if (pairflag && force->pair && force->pair->compute_flag) {
    double* eatom = force->pair->eatom;
    for (i = 0; i < npair; i++)
      energy[i] += eatom[i];
  }

  if (bondflag && force->bond) {
    double* eatom = force->bond->eatom;
    for (i = 0; i < nbond; i++)
      energy[i] += eatom[i];
  }

  if (angleflag && force->angle) {
    double* eatom = force->angle->eatom;
    for (i = 0; i < nbond; i++)
      energy[i] += eatom[i];
  }

  if (dihedralflag && force->dihedral) {
    double* eatom = force->dihedral->eatom;
    for (i = 0; i < nbond; i++)
      energy[i] += eatom[i];
  }

  if (improperflag && force->improper) {
    double* eatom = force->improper->eatom;
    for (i = 0; i < nbond; i++)
      energy[i] += eatom[i];
  }

  if (cl_pairflag && pair) {
    double* eatom = pair->cl_atom;
    for (i = 0; i < npair; i++)
      energy[i] += eatom[i];
  }

  if (kspaceflag && force->kspace && force->kspace->compute_flag) {
    double* eatom = force->kspace->eatom;
    for (i = 0; i < nkspace; i++)
      energy[i] += eatom[i];
  }

  // add in per-atom contributions from relevant fixes
  // always only for owned atoms, not ghost

  if (fixflag && modify->n_energy_atom) modify->energy_atom(nlocal, energy);

  // communicate ghost energy between neighbor procs

  if (force->newton || (force->kspace && force->kspace->tip4pflag)) comm->reverse_comm_compute(this);

  // zero energy of atoms not in group
  // only do this after comm since ghost contributions must be included

  int* mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) energy[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

int ComputePEAtomGA::pack_reverse_comm(int n, int first, double* buf)
{
  int i, m, last;

  m    = 0;
  last = first + n;
  for (i = first; i < last; i++)
    buf[m++] = energy[i];
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputePEAtomGA::unpack_reverse_comm(int n, int* list, double* buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    energy[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputePEAtomGA::memory_usage()
{
  double bytes = (double)nmax * sizeof(double);
  return bytes;
}
