/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Version: 1.00 Data: Jan/18/2019
   Kylin (WUST)
   A mixture Guassian/Point charge interaction within PBC.
   The reciprocal interaction was consistent with the Point charge version of
   Ewald summation or PPPM in Lammps
   Version: 1.01  Data: Nov/19/2020
   Kylin (WUST)
   A coulombic interaction for point charge with a virtual gaussian-like self
energy. The reciprocal interaction was consistent with the Point charge version
of Ewald summation or PPPM in Lammps
------------------------------------------------------------------------- */

#include "func_macro.h"

#include "electro_control_GA.h"
#include "fix_conp_GA.h"
#include "pair_lj_cut_point_long.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "integrate.h"
#include "kspace.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "erfcx_y100.h"

#include <algorithm>
#include <cfloat>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

enum { P_CHARGE, G_CHARGE };
enum { NO_SELF, ONLY_SELF, MIXTURE };
/* ---------------------------------------------------------------------- */

PairLJCutPointLong::PairLJCutPointLong(LAMMPS* lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  respa_enable         = 1;
  writedata            = 1;
  ftable               = nullptr;
  qdist                = 0.0;
  cut_respa            = nullptr;

  self_flag         = true; // self-energy for point/Gaussian charges
  gauss_flag        = false;
  pair_num          = 0;
  pair_num_allocate = 0;
  pair_ij           = nullptr;
  pair_ecoul        = nullptr;
  pair_evdwl        = nullptr;
  pair_fvdwl        = nullptr;
  pair_fcoul        = nullptr;
  pair_dxyz         = nullptr;
  me                = comm->me;

  char* compute_str[4] = {"__Electro_Control_GA_Compute__", "all", "pe/atom_GA", "__NIL_COMPUTER__"};
  modify->add_compute(4, compute_str, 0);
  compute_id = modify->ncompute - 1;
}

/* ---------------------------------------------------------------------- */

PairLJCutPointLong::~PairLJCutPointLong()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

    memory->destroy(self_Greenf);
  }

  if (ftable) {
    free_tables();
    if (gauss_flag) {
      memory->destroy(ftable_list);
      memory->destroy(dftable_list);
      memory->destroy(etable_list);
      memory->destroy(detable_list);
      memory->destroy(ctable_list);
      memory->destroy(dctable_list);
      free(ftable_vec);
      free(dftable_vec);
      free(etable_vec);
      free(detable_vec);
      free(ctable_vec);
      free(dctable_vec);
    }
  };

  memory->destroy(eta);
  memory->destroy(sigma_width);
  memory->destroy(eta_mix_table);
  memory->destroy(isGCharge);
  if (sv_SOL_flag) {
    memory->destroy(pair_ij);
    memory->destroy(pair_ecoul);
    memory->destroy(pair_evdwl);
    memory->destroy(pair_fcoul);
    memory->destroy(pair_fvdwl);
    memory->destroy(pair_dxyz);
  }
}

void PairLJCutPointLong::clear_saved()
{
  sv_SOL_saved    = false;
  sv_SOL_timestep = -1;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutPointLong::settings(int narg, char** arg)
{
  if (narg < 1) error->all(FLERR, "Illegal pair_style command");

  cut_lj_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (narg == 1)
    cut_coul = cut_lj_global;
  else
    cut_coul = utils::numeric(FLERR, arg[1], false, lmp);
  // Add atomtype dependent Gaussian distribution Width:
  // eta = 1/(2*sigma_i^2)^0.5
  // For point charge sigma_i = 0; eta -> \Infty
  int n = atom->ntypes, j, me = comm->me;
  bool self_energy_flag = false;
  n_plus                = n + 1;
  memory->create(eta, n + 1, "pair:eta");
  memory->create(isGCharge, n + 1, "pair:isGCharge");
  memory->create(sigma_width, n + 1, "pair:sigma_width");
  memory->create(eta_mix_table, n + 1, n + 1, "pair:eta_mix_table");

  memory->create(self_Greenf, n + 1, "pair:self_Greenf");

  // The default
  // Initialize eta, sigma_width and eta_mix_table with zero.
  // in eta_mix_table, Zero mean P <-> P interaction.
  for (int i = 0; i <= n; ++i) {
    eta[i] = sigma_width[i] = self_Greenf[i] = 0;
    isGCharge[i]                             = 0;
    for (int j = 0; j <= n; ++j) {
      eta_mix_table[i][j] = 0;
    }
  }

  // Set eta_mix_table for Gaussian Charge interaction
  int iarg = 2, atype, ilo, ihi;
  double value;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "gauss") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal pair_lj_cut_gauss_long argument for gauss option");
      // utils::bounds(FLERR, arg[iarg + 1], 1, atom->ntypes, ilo, ihi, error);
      atype = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);

      value = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      if (value <= 0.0)
        error->all(FLERR, "Illegal inverse Guassian distribution width (only "
                          "value for eta > 0)!");
      if (me == 0) utils::logmesg(lmp, "Set type[{}] as Gauss charge;\n", atype);
      isGCharge[atype]   = G_CHARGE;
      eta[atype]         = value;
      sigma_width[atype] = 1 / sqrt(value * value * 2);
      self_Greenf[atype] = eta[atype] / MY_SQRT2;
      iarg += 3;
      gauss_flag = self_energy_flag = true;
    } else if (strcmp(arg[iarg], "point") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal pair_lj_cut_point_long argument for self option");
      // utils::bounds(FLERR, arg[iarg + 1], 1, atom->ntypes, ilo, ihi, error);
      atype = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);

      value = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      if (atype < 1 || atype > n) error->all(FLERR, "Illegal atom type!");
      if (value < 0.0)
        error->all(FLERR, "Illegal inverse Guassian distribution width (only "
                          "value for eta_self >= 0)!");

      if (me == 0) utils::logmesg(lmp, "Set type[{}] as Point charge;\n", atype);

      // value = 0;
      isGCharge[atype]   = P_CHARGE;
      eta[atype]         = 0;
      sigma_width[atype] = 0;
      self_Greenf[atype] = value / MY_SQRT2;
      if (value != 0) self_energy_flag = true;
      iarg += 3;

    } else if (strcmp(arg[iarg], "eta") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal pair_lj_cut_gauss_long argument for self option");
      // utils::bounds(FLERR, arg[iarg + 1], 1, atom->ntypes, ilo, ihi, error);
      atype = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);

      value = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      if (atype < 1 || atype > n) error->all(FLERR, "Illegal atom type!");
      if (value <= 0.0)
        error->all(FLERR, "Illegal inverse Guassian distribution width (only "
                          "value for eta > 0)!");
      if (value > 1e30) {
        isGCharge[atype] = P_CHARGE;
        value            = 0;
        if (me == 0) utils::logmesg(lmp, "Set type[{}] as Point charge, ", atype);
      } else {
        isGCharge[atype] = G_CHARGE;
        if (me == 0) utils::logmesg(lmp, "Set type[{}] as Gauss charge, ", atype);
      }
      eta[atype]         = value;
      sigma_width[atype] = value == 0 ? 0 : 1 / sqrt(value * value * 2);
      if (sigma_width[atype] != 0) gauss_flag = true;
      value = atof(arg[iarg + 3]);
      if (value < 0.0) error->all(FLERR, "Illegal extra eta value (only valid for eta > 0)!");
      self_Greenf[atype] = value / MY_SQRT2;
      if (value != 0) self_energy_flag = true;
      if (me == 0) utils::logmesg(lmp, "self_Coeff[atype] = {}\n",
                                  self_Greenf[atype]); // self_Greenf -> self_Coeff
      iarg += 4;

    } else if (strcmp(arg[iarg], "self") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_lj_cut_point_long argument for self option");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        self_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        self_flag = false;
      } else
        error->all(FLERR, "Illegal pair_lj_cut_point_long argument for self option");
      iarg += 2;
    } else if (strcmp(arg[iarg], "sv_SOL") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_lj_cut_point_long argument for sv_SOL option");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        sv_SOL_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        sv_SOL_flag = false;
      } else
        error->all(FLERR, "Illegal pair_lj_cut_point_long argument for self option");
      iarg += 2;
    } else if (strcmp(arg[iarg], "sv_GA") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_lj_cut_point_long argument for sv_GA option");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        sv_GA_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        sv_GA_flag = false;
      } else
        error->all(FLERR, "Illegal pair_lj_cut_point_long argument for self option");
      iarg += 2;
    } else {
      if (me == 0) utils::logmesg(lmp, "{} {} \n", iarg, arg[iarg]);
      error->all(FLERR, "Illegal pair_lj_cut_point_long arguments");
    }
  }

  for (atype = 1; atype <= n; ++atype) {
    if (eta[atype] == 0) {
      if (me == 0) utils::logmesg(lmp, "Atom type: [{}] Eta: Inf, Sigma_width 0 \n", atype);
      continue;
    } else {
      if (me == 0) utils::logmesg(lmp, "Atom type: [{}] Eta: {}, Sigma_width {} \n", atype, eta[atype], sigma_width[atype]);
    }
    for (j = 1; j <= n; ++j) {
      value = sigma_width[atype] * sigma_width[atype] + sigma_width[j] * sigma_width[j];
      // 1/eta_mix_table[i][j]^2 = 1/eta[i]^2 + 1/eta[j]^2;
      eta_mix_table[atype][j] = 1 / sqrt(2 * value);
      eta_mix_table[j][atype] = eta_mix_table[atype][j];
    }
  }
  if (me == 0) {
    utils::logmesg(lmp, "The default charge type are point charge. \n");
    utils::logmesg(lmp, "eta_table = \n");
    // value =
    for (atype = 1; atype <= n; ++atype) {
      for (j = 1; j <= n; ++j) {
        utils::logmesg(lmp, "{:12.10f} ", eta_mix_table[atype][j]);
      }
      utils::logmesg(lmp, "\n");
    }
  }

  // reset cutoffs that have been explicitly set
  if (allocated) {
    int i, j;
    for (i = 1; i <= n; i++)
      for (j = i; j <= n; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }

  if (self_energy_flag == false && self_flag == true) {
    self_flag = false;
  }
  if (self_flag == false) {
    for (int i = 1; i < n_plus; i++) {
      self_Greenf[i] = 0.0;
    }
  }

  // gauss_flag = true;
  if (me == 0) {
    if (gauss_flag == true)
      utils::logmesg(lmp, "[Pair] Calculate with the Gaussian Charge\n");
    else
      utils::logmesg(lmp, "[Pair] Calculate without the Gaussian Charge\n\n");
    if (self_flag == true)
      utils::logmesg(lmp, "[Pair] Include the nonsingular self-energy\n");
    else
      utils::logmesg(lmp, "[Pair] Exclude the nonsingular self-energy\n");
  }
}

/* ---------------------------------------------------------------------- */
void PairLJCutPointLong::init_special_tables()
{
  if (force->kspace == nullptr) {
    error->all(FLERR, "No table loopup for Lj/Cut/Gauss/long without Kspace style!");
  }
  // XXX_vec store the pointer to the tabulated Coulomb interaction
  // XXX_list store the raw data/value for the tabulated Coulomb interaction
  int n = atom->ntypes;
  int n_gauss, i, j, k, ii, atype_i, atype_j;
  int* GIndex_to_Atype;
  int* Atype_to_GIndex;
  int ntable = 1;
  for (int i = 0; i < ncoultablebits; i++)
    ntable *= 2;

  memory->create(GIndex_to_Atype, n + 1, "pair:GIndex_to_Atype");
  memory->create(Atype_to_GIndex, n + 1, "pair:Atype_to_GIndex");
  n_gauss = 1;
  for (size_t i = 1; i <= n; i++) {
    if (isGCharge[i] != 0) {
      GIndex_to_Atype[n_gauss] = i;
      Atype_to_GIndex[i]       = n_gauss;
      n_gauss++;
    } else {
      Atype_to_GIndex[i] = -1;
    }
  }
  n_gauss--; // make n_gauss as the number of gauss charges

  // n_gauss_pair = n_gauss + 1 + n_gauss * (n_gauss + 1) / 2;
  // The first term account for (n_gauss) Gauss charges to Points charge
  // interaction plus one Point to Point interaction; The second term account
  // for n_gauss * (n_gauss + 1) / 2 of Gauss to Gauss interactions;
  int n_gauss_pair = 1 + n_gauss * (n_gauss + 3) / 2;
  int n_pair       = (n + 1) * (n + 1);

  ftable_vec  = (double**)malloc(sizeof(double*) * n_pair);
  dftable_vec = (double**)malloc(sizeof(double*) * n_pair);
  etable_vec  = (double**)malloc(sizeof(double*) * n_pair);
  detable_vec = (double**)malloc(sizeof(double*) * n_pair);
  ctable_vec  = (double**)malloc(sizeof(double*) * n_pair);
  dctable_vec = (double**)malloc(sizeof(double*) * n_pair);
  for (i = 0; i < n_pair; i++) {
    ftable_vec[i] = dftable_vec[i] = etable_vec[i] = ctable_vec[i] = dctable_vec[i] = nullptr;
  }
  if (me == 0) utils::logmesg(lmp, "[Pair] n_gauss_pair = {}; n_pair = {}\n", n_gauss_pair, n_pair);
  double g_ewald_store = force->kspace->g_ewald;
  // printf("g_ewald_store = %f;\n", g_ewald_store);

  memory->create(ftable_list, n_gauss_pair, ntable, "pair:ftable");
  memory->create(dftable_list, n_gauss_pair, ntable, "pair:ctable");

  memory->create(etable_list, n_gauss_pair, ntable, "pair:etable");
  memory->create(detable_list, n_gauss_pair, ntable, "pair:dftable");

  memory->create(ctable_list, n_gauss_pair, ntable, "pair:dctable");
  memory->create(dctable_list, n_gauss_pair, ntable, "pair:detable");

  // P<->P interction  ii_max = 1
  init_tables(cut_coul, nullptr);
  std::copy(ftable, ftable + ntable, ftable_list[0]);
  std::copy(dftable, dftable + ntable, dftable_list[0]);
  std::copy(etable, etable + ntable, etable_list[0]);
  std::copy(detable, detable + ntable, detable_list[0]);
  std::copy(ctable, ctable + ntable, ctable_list[0]);
  std::copy(dctable, dctable + ntable, dctable_list[0]);
  // Generated tabulated P<->G interction into table_list
  for (i = 1; i <= n_gauss; i++) {
    force->kspace->g_ewald = eta[GIndex_to_Atype[i]];
    init_tables(cut_coul, nullptr);
    for (ii = 0; ii < ntable; ii++) {
      ftable_list[i][ii]  = ftable_list[0][ii] - ftable[ii];
      dftable_list[i][ii] = dftable_list[0][ii] - dftable[ii];
      etable_list[i][ii]  = etable_list[0][ii] - etable[ii];
      detable_list[i][ii] = detable_list[0][ii] - detable[ii];
      ctable_list[i][ii]  = ftable[ii] - ctable[ii];
      dctable_list[i][ii] = dftable[ii] - dctable[ii];
    }
  }

  // Generated tabulated G<->G interction into table_list
  // first 0~n_gauss column stored for one P<->P and n_gauss P<->G interactions
  // Next column start with idx = n_gauss + 1
  int idx = n_gauss + 1, idx_ij, idx_ji;
  for (i = 1; i <= n_gauss; i++) {
    for (j = 1; j <= i; j++) {
      atype_i = GIndex_to_Atype[i];
      atype_j = GIndex_to_Atype[j];

      force->kspace->g_ewald = eta_mix_table[atype_i][atype_j];
      init_tables(cut_coul, nullptr);
      for (ii = 0; ii < ntable; ii++) {
        ftable_list[idx][ii]  = ftable_list[0][ii] - ftable[ii];
        dftable_list[idx][ii] = dftable_list[0][ii] - dftable[ii];
        etable_list[idx][ii]  = etable_list[0][ii] - etable[ii];
        detable_list[idx][ii] = detable_list[0][ii] - detable[ii];
        ctable_list[idx][ii]  = ftable[ii] - ctable[ii];
        dctable_list[idx][ii] = dftable[ii] - dctable[ii];
      }

      idx_ij = atype_i * (n + 1) + atype_j;
      idx_ji = atype_j * (n + 1) + atype_i;

      // printf("     i = %d;       j = %d; n_gauss = %d\n", i, j, n_gauss);
      // printf("  at_i = %d;    at_j = %d; idx = %d\n", atype_i, atype_j, idx);
      // printf("idx_ij = %d;  idx_ji = %d;\n", idx_ij, idx_ji);

      ftable_vec[idx_ij]  = ftable_list[idx];
      dftable_vec[idx_ij] = dftable_list[idx];
      etable_vec[idx_ij]  = etable_list[idx];
      detable_vec[idx_ij] = detable_list[idx];
      ctable_vec[idx_ij]  = ctable_list[idx];
      dctable_vec[idx_ij] = dctable_list[idx];

      ftable_vec[idx_ji]  = ftable_list[idx];
      dftable_vec[idx_ji] = dftable_list[idx];
      etable_vec[idx_ji]  = etable_list[idx];
      detable_vec[idx_ji] = detable_list[idx];
      ctable_vec[idx_ji]  = ctable_list[idx];
      dctable_vec[idx_ji] = dctable_list[idx];
      idx++;
    }
  }

  for (i = 1; i <= n; i++) {
    for (j = 1; j <= n; j++) {
      idx_ij = i * (n + 1) + j;
      idx_ji = j * (n + 1) + i;
      if (isGCharge[i] == 1 && isGCharge[j] == 1) {
        // printf("idx_ij = %d; idx_ji = %d;\n", idx_ij, idx_ji);
        if (ftable_vec[idx_ji] == nullptr || ftable_vec[idx_ij] == nullptr) {
          error->all(FLERR, "No table loopup for Lj/Cut/Gauss/long without Kspace style!");
        }
        // ftable_vec[idx_ji], ftable_vec[idx_ij]);
      } else {
        // GIndex of [1,n_gauss] are also the columns of
        // P<->G interaction in table_list
        if (isGCharge[i] == 1) {
          idx = Atype_to_GIndex[i];
        } else if (isGCharge[j] == 1) {
          idx = Atype_to_GIndex[j];
        } else {
          idx = 0; // P<->P interaction in the 0th column
        }
        idx                 = 0;
        ftable_vec[idx_ij]  = ftable_list[idx];
        dftable_vec[idx_ij] = dftable_list[idx];
        etable_vec[idx_ij]  = etable_list[idx];
        detable_vec[idx_ij] = detable_list[idx];
        ctable_vec[idx_ij]  = ctable_list[idx];
        dctable_vec[idx_ij] = dctable_list[idx];

        ftable_vec[idx_ji]  = ftable_list[idx];
        dftable_vec[idx_ji] = dftable_list[idx];
        etable_vec[idx_ji]  = etable_list[idx];
        detable_vec[idx_ji] = detable_list[idx];
        ctable_vec[idx_ji]  = ctable_list[idx];
        dctable_vec[idx_ji] = dctable_list[idx];
      }
    }
  }
  force->kspace->g_ewald = g_ewald_store;
  memory->destroy(GIndex_to_Atype);
  memory->destroy(Atype_to_GIndex);
}

/* ---------------------------------------------------------------------- */
void PairLJCutPointLong::compute(int eflag, int vflag)
{
  /*
  int i = 1, j = 1000, k = 1010;
  double* q = atom->q;
  q[i] = q[j] = q[k] = 0.5;
  int* type          = atom->type;
  printf("i, j, k = %d %d %d\n", i, j, k);
  printf("type = %d %d %d\n", type[i], type[j], type[k]);
  double rsq = 1.57, fforce, value = 0;

  value = single(j, k, 1, 1, rsq, 1, 0, fforce);
  printf("1.0 %f %f\n", value, fforce);

  value = single(j, k, 1, 1, rsq, 0.5, 0, fforce);
  printf("0.5 %f %f\n", value, fforce);

  value = single(j, k, 1, 1, rsq, 0, 0, fforce);
  printf("0.0 %f %f\n", value, fforce);

  value = single(j, k, 3, 1, rsq, 1, 0, fforce);
  printf("1.0 %f %f\n", value, fforce);

  value = single(j, k, 3, 1, rsq, 0.5, 0, fforce);
  printf("0.5 %f %f\n", value, fforce);

  value = single(j, k, 3, 1, rsq, 0, 0, fforce);
  printf("0.0 %f %f\n", value, fforce);

  value = compute_single(1, 1, rsq);
  printf("compute_single %f\n", value * 0.25);

  value = compute_single(3, 1, rsq);
  printf("compute_single %f\n", value * 0.25);
  exit(0);
  */
  // double q[i] = > single(1, 2);

  /*
  int inum         = list->inum;
  int* ilist       = list->ilist;
  int* numneigh    = list->numneigh;
  int** firstneigh = list->firstneigh;
  int* mask        = atom->mask;
  int nlocal       = atom->nlocal;
  // printf("nlocal = %d, inum = %d\n", nlocal, inum);
  for (int ii = 0; ii < inum; ii++) {
    int i      = ilist[ii];
    int* jlist = firstneigh[i];
    int jnum   = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      // printf("jj = %d, %d\n", jj, jnum);
      int j = jlist[jj];
      j &= NEIGHMASK;
      // if (j < i) printf("ii = %d, i = %d j = %d\n", ii, i, j);
    }
  }
  */
  // FUNC_MACRO(0);
  // printf("compute for DEBUG_LOG_LEVEL = %d, %d\n", DEBUG_LOG_LEVEL, FUNCTION_COUNT);
  if (sv_SOL_saved == false || electro == nullptr) {
    if (gauss_flag)
      return compute_Gauss(eflag, vflag);
    else
      return compute_Point(eflag, vflag);
  } else {
    int groupbit = electro->FIX->groupbit;
    // initial compute
    if (eflag || vflag) {
      if (atom->nmax > maxeatom) {
        memory->destroy(cl_atom);
        memory->create(cl_atom, comm->nthreads * atom->nmax, "pair:cl_atom");
      }
      ev_setup(eflag, vflag);
    } else
      evflag = vflag_fdotr = 0;

    if (gauss_flag) {
      compute_Gauss_Group(eflag, vflag, groupbit, true);
      if (sv_GA_flag) compute_Gauss_Group(eflag, vflag, groupbit, false);
    } else {
      compute_Point_Group(eflag, vflag, groupbit, true);
      if (sv_GA_flag) compute_Point_Group(eflag, vflag, groupbit, false);
    }
    add_saved_pair(eflag, vflag, groupbit);
    if (vflag_fdotr) virial_fdotr_compute();
  }
}
/* -------------------------------------------------------------------------
          Compute energy between SOL<=>GA via Coulomb potential
------------------------------------------------------------------------- */

void PairLJCutPointLong::compute_Group(int eflag, int vflag, int FIX_groupbit, bool FLAG)
{
  if (gauss_flag)
    compute_Gauss_Group(eflag, vflag, FIX_groupbit, FLAG);
  else
    compute_Point_Group(eflag, vflag, FIX_groupbit, FLAG);
}

void PairLJCutPointLong::add_saved_pair(int eflag, int vflag, int FIX_groupbit)
{
  FUNC_MACRO(0);

  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double forcecoul, forcelj, r2inv;

  double** f           = atom->f;
  double* q            = atom->q;
  int* type            = atom->type;
  int nlocal           = atom->nlocal;
  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  int* mask            = atom->mask;
  for (int ip = 0; ip < pair_num; ip++) {
    int ip_x2        = ip * 2;
    int ip_x4        = ip * 4;
    int i            = pair_ij[ip_x2 + 0];
    int j            = pair_ij[ip_x2 + 1];
    double ecoulhalf = pair_ecoul[ip];
    evdwl            = pair_evdwl[ip];
    forcelj          = pair_fvdwl[ip];
    delx             = pair_dxyz[ip_x4 + 0];
    dely             = pair_dxyz[ip_x4 + 1];
    delz             = pair_dxyz[ip_x4 + 2];
    r2inv            = pair_dxyz[ip_x4 + 3];

    // bool iGA_Flag = (mask[i] & FIX_groupbit) != 0;
    // bool jGA_Flag = (mask[j] & FIX_groupbit) != 0;
    ecoulhalf *= q[i];
    forcecoul = q[i] * pair_fcoul[ip];
    /*
    if (iGA_Flag) {
      ecoulhalf *= q[i];
      forcecoul = q[i] * pair_fcoul[ip];
    } else {
      // printf("j=%d is not GA\n", j);
      ecoulhalf *= q[j];
      forcecoul = q[j] * pair_fcoul[ip];
    }
    */
    fpair = (forcecoul + forcelj) * r2inv;

    // eatom[i] += 100 * evdwl;
    f[i][0] += delx * fpair;
    f[i][1] += dely * fpair;
    f[i][2] += delz * fpair;
    if (newton_pair || j < nlocal) {
      // eatom[j] += evdwl;
      f[j][0] -= delx * fpair;
      f[j][1] -= dely * fpair;
      f[j][2] -= delz * fpair;
    }

    if (evflag) {
      if (newton_pair || i < nlocal) {
        cl_atom[i] += ecoulhalf;
        // eatom[i] += ecoulhalf + 0.5 * evdwl;
      }
      if (newton_pair || j < nlocal) {
        cl_atom[j] += ecoulhalf;
        // eatom[j] += ecoulhalf + 0.5 * evdwl;
      }
      ev_tally(i, j, nlocal, newton_pair, evdwl, 2 * ecoulhalf, fpair, delx, dely, delz);
    }
  }
}

/* -------------------------------------------------------------------------
          Compute energy between SOL<=>GA via Coulomb potential
------------------------------------------------------------------------- */

void PairLJCutPointLong::compute_Point_Group(int eflag, int vflag, int FIX_groupbit, const bool include_flag)
{
  FUNC_MACRO(0);
  int i, ii, j, jj, inum, jnum, itype, jtype, itable;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double fraction, table;
  double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double grij, expm2, prefactor, erfc;
  int *ilist, *jlist, *numneigh, **firstneigh; // idx;
  double rsq;

  evdwl = ecoul = 0.0;
  double** x    = atom->x;
  double** f    = atom->f;
  int* tag      = atom->tag;

  double* q            = atom->q;
  int* type            = atom->type;
  int nlocal           = atom->nlocal;
  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  const double qqrd2e  = force->qqrd2e;
  double guass_rij, guass_expm2, value, gauss_erfc;

  inum       = list->inum;
  ilist      = list->ilist;
  numneigh   = list->numneigh;
  firstneigh = list->firstneigh;
  int* mask  = atom->mask;

  for (ii = 0; ii < inum; ii++) {
    i             = ilist[ii];
    bool iGA_Flag = (mask[i] & FIX_groupbit);

    // remove all GA interaction
    if (iGA_Flag == include_flag) continue;

    itype = type[i];
    qtmp  = q[i];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    jlist = firstneigh[i];
    jnum  = numneigh[i];

    // printf("ii = %d\n", ii);
    // jGCharge = 0;
    for (jj = 0; jj < jnum; jj++) {
      // printf("jj = %d, %d\n", jj, jnum);
      j           = jlist[jj];
      factor_lj   = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      bool jGA_Flag = (mask[j] & FIX_groupbit);
      if (jGA_Flag == include_flag) continue;
      /*
      if (iGA_Flag != jGA_Flag) {
        continue;
      }
      */

      delx  = xtmp - x[j][0];
      dely  = ytmp - x[j][1];
      delz  = ztmp - x[j][2];
      rsq   = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;
        if (rsq < cut_coulsq) {
          if (!ncoultablebits || rsq <= tabinnersq) {
            r         = sqrt(rsq);
            prefactor = qqrd2e * qtmp * q[j] / r;
            grij      = g_ewald * r;
            expm2     = expmsq(grij);
            erfc      = _my_erfcx(grij) * expm2; // grij > 0 with g_ewald > 0;
            forcecoul = prefactor * (erfc + MY_ISPI4 * grij * expm2);

            // P <-> P interaction
            if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;

          } else {
            // error->all(FLERR,"No table loopup for Lj/Cut/Gauss/long!");
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            itable       = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction  = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            table     = ftable[itable] + fraction * dftable[itable];
            forcecoul = qtmp * q[j] * table;
            if (factor_coul < 1.0) {
              table     = ctable[itable] + fraction * dctable[itable];
              prefactor = qtmp * q[j] * table;
              forcecoul -= (1.0 - factor_coul) * prefactor;
            }
          }
        } else
          forcecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv   = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        } else
          forcelj = 0.0;

        fpair = (forcecoul + factor_lj * forcelj) * r2inv;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) {
          if (rsq < cut_coulsq) {
            if (!ncoultablebits || rsq <= tabinnersq) {
              ecoul = prefactor * erfc;
            } else {
              table = etable[itable] + fraction * detable[itable];
              ecoul = qtmp * q[j] * table;
            }
            if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
          } else
            ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
            evdwl *= factor_lj;
          } else
            evdwl = 0.0;
        }
        if (evflag) {
          double ecoulhalf = 0.5 * ecoul;
          if (newton_pair || i < nlocal) {
            cl_atom[i] += ecoulhalf;
          }
          if (newton_pair || j < nlocal) {
            cl_atom[j] += ecoulhalf;
          }
          ev_tally(i, j, nlocal, newton_pair, evdwl, ecoul, fpair, delx, dely, delz);
        }
      } // if rsq < r_cut
    }   // jj
    /* ------ add the self interaction energy ------ */
    // printf("self_flag0\n");
    if (self_flag) {
      value = qtmp * qtmp * self_Greenf[itype];
      cl_atom[i] += value;
      eatom[i] += value;
      force->pair->eng_coul += value;
    }
    /* ------ add the self interaction energy ------ */
  } // ii
}

/* -------------------------------------------------------------------------
    Compute between SOL<=>GA via the Gauss potential
------------------------------------------------------------------------- */

void PairLJCutPointLong::compute_Gauss_Group(int eflag, int vflag, int FIX_groupbit, const bool include_flag)
{
  FUNC_MACRO(0);
  // GAflag;
  // return compute_Point_Group(eflag, vflag, FIX_groupbit, include_flag);
  int i, ii, j, jj, inum, jnum, itype, jtype, itable;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double fraction, table;
  double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double grij, expm2, prefactor, erfc, qiqj;
  int *ilist, *jlist, *numneigh, **firstneigh, idx;
  double rsq;

  evdwl = ecoul = 0.0;

  double** x = atom->x;
  double** f = atom->f;
  int* tag   = atom->tag;

  double* q            = atom->q;
  int* type            = atom->type;
  int nlocal           = atom->nlocal;
  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  const double qqrd2e  = force->qqrd2e;
  double guass_rij, guass_expm2, gauss_erfc, value;

  inum       = list->inum;
  ilist      = list->ilist;
  numneigh   = list->numneigh;
  firstneigh = list->firstneigh;
  int* mask  = atom->mask;

  for (ii = 0; ii < inum; ii++) {
    i             = ilist[ii];
    bool iGA_Flag = (mask[i] & FIX_groupbit);

    // remove all GA interaction
    if (iGA_Flag == include_flag) continue;

    qtmp  = q[i];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum  = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j           = jlist[jj];
      factor_lj   = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      bool jGA_Flag = (mask[j] & FIX_groupbit);
      if (jGA_Flag == include_flag) continue;

      /*
      if (iGA_Flag != jGA_Flag) {
        continue;
      }
      */

      delx  = xtmp - x[j][0];
      dely  = ytmp - x[j][1];
      delz  = ztmp - x[j][2];
      rsq   = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;
        if (rsq < cut_coulsq) {
          if (!ncoultablebits || rsq <= tabinnersq) {
            r         = sqrt(rsq);
            prefactor = qqrd2e * qtmp * q[j] / r;
            grij      = g_ewald * r;
            expm2     = expmsq(grij);
            erfc      = my_erfcx(grij) * expm2;
            forcecoul = prefactor * (erfc + MY_ISPI4 * grij * expm2);

            // P <-> P interaction
            if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
              gauss_erfc = 0;
              if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;
              // P <-> G or G <-> G interacion
            } else {
              guass_rij          = eta_mix_table[itype][jtype] * r;
              guass_expm2        = expmsq(guass_rij);
              gauss_erfc         = my_erfcx(guass_rij) * guass_expm2;
              double f_prefactor = prefactor * factor_coul;

              forcecoul -= f_prefactor * (gauss_erfc + MY_ISPI4 * guass_rij * guass_expm2);
              if (factor_coul < 1.0) forcecoul -= prefactor - f_prefactor;
            }
          } else {
            // error->all(FLERR,"No table loopup for Lj/Cut/Gauss/long!");
            union_int_float_t rsq_lookup;
            idx = itype * n_plus + jtype;
            // printf("%d, %d %d\n", itype, jtype, idx);
            rsq_lookup.f = rsq;
            itable       = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            qiqj     = qtmp * q[j];
            if (factor_coul < 1.0) {
              double gtable  = ctable_vec[idx][itable] + fraction * dctable_vec[idx][itable];
              double c_table = ctable[itable] + fraction * dctable[itable];

              table = ftable[itable] + fraction * dftable[itable] + factor_coul * gtable;
              table -= (1.0 - factor_coul) * c_table;
              prefactor = qiqj * c_table;
            } else {
              table = ftable_vec[idx][itable] + fraction * dftable_vec[idx][itable];
            }
            forcecoul = qiqj * table;
          }
        } else
          forcecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv   = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        } else
          forcelj = 0.0;

        fpair = (forcecoul + factor_lj * forcelj) * r2inv;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) {
          if (rsq < cut_coulsq) {
            if (!ncoultablebits || rsq <= tabinnersq) {
              ecoul = prefactor * (erfc - factor_coul * gauss_erfc);
            } else {
              if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
                table = etable[itable] + fraction * detable[itable];
                ecoul = qiqj * table;
              } else {
                double table0 = etable[itable] + fraction * detable[itable];
                table         = etable_vec[idx][itable] + fraction * detable_vec[idx][itable];
                table         = table0 + factor_coul * (table - table0);
                ecoul         = qiqj * table;
              }
            }
            if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
          } else
            ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
            evdwl *= factor_lj;
          } else
            evdwl = 0.0;
        }
        if (evflag) {
          double ecoulhalf = 0.5 * ecoul;
          if (newton_pair || i < nlocal) {
            cl_atom[i] += ecoulhalf;
          }
          if (newton_pair || j < nlocal) {
            cl_atom[j] += ecoulhalf;
          }
          ev_tally(i, j, nlocal, newton_pair, evdwl, ecoul, fpair, delx, dely, delz);
        }
      } // if rsq < r_cut
    }   // jj
    /* ------ add the self interaction energy ------ */
    // printf("self_flag0\n");
    if (self_flag) {
      value = qtmp * qtmp * self_Greenf[itype];
      cl_atom[i] += value;
      eatom[i] += value;
      force->pair->eng_coul += value;
    }
    /* ------ add the self interaction energy ------ */
  } // ii
}

/* -------------------------------------------------------------------------
    Compute between All<=>All via the Coulomb potential
------------------------------------------------------------------------- */

void PairLJCutPointLong::compute_Point(int eflag, int vflag)
{
  FUNC_MACRO(0);
  int i, ii, j, jj, inum, jnum, itype, jtype, itable;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double fraction, table;
  double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double grij, expm2, prefactor, erfc;
  int *ilist, *jlist, *numneigh, **firstneigh; // idx;
  double rsq;

  evdwl = ecoul = 0.0;
  if (eflag || vflag) {
    if (atom->nmax > maxeatom) {
      memory->destroy(cl_atom);
      memory->create(cl_atom, comm->nthreads * atom->nmax, "pair:cl_atom");
    }
    ev_setup(eflag, vflag);
  } else
    evflag = vflag_fdotr = 0;

  double** x = atom->x;
  double** f = atom->f;
  int* tag   = atom->tag;

  double* q            = atom->q;
  int* type            = atom->type;
  int nlocal           = atom->nlocal;
  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  const double qqrd2e  = force->qqrd2e;
  double guass_rij, guass_expm2, value, gauss_erfc;

  inum       = list->inum;
  ilist      = list->ilist;
  numneigh   = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i     = ilist[ii];
    qtmp  = q[i];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum  = numneigh[i];
    // printf("ii = %d\n", ii);
    // jGCharge = 0;
    for (jj = 0; jj < jnum; jj++) {
      // printf("jj = %d, %d\n", jj, jnum);
      j           = jlist[jj];
      factor_lj   = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx  = xtmp - x[j][0];
      dely  = ytmp - x[j][1];
      delz  = ztmp - x[j][2];
      rsq   = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;
        if (rsq < cut_coulsq) {
          if (!ncoultablebits || rsq <= tabinnersq) {
            r         = sqrt(rsq);
            prefactor = qqrd2e * qtmp * q[j] / r;
            grij      = g_ewald * r;
            expm2     = expmsq(grij);
            erfc      = _my_erfcx(grij) * expm2; // grij > 0 with g_ewald > 0;
            forcecoul = prefactor * (erfc + MY_ISPI4 * grij * expm2);

            // P <-> P interaction
            if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;

          } else {
            // error->all(FLERR,"No table loopup for Lj/Cut/Gauss/long!");
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            itable       = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction  = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            table     = ftable[itable] + fraction * dftable[itable];
            forcecoul = qtmp * q[j] * table;
            if (factor_coul < 1.0) {
              table     = ctable[itable] + fraction * dctable[itable];
              prefactor = qtmp * q[j] * table;
              forcecoul -= (1.0 - factor_coul) * prefactor;
            }
          }
        } else
          forcecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv   = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        } else
          forcelj = 0.0;

        fpair = (forcecoul + factor_lj * forcelj) * r2inv;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) {
          if (rsq < cut_coulsq) {
            if (!ncoultablebits || rsq <= tabinnersq) {
              ecoul = prefactor * erfc;
            } else {
              table = etable[itable] + fraction * detable[itable];
              ecoul = qtmp * q[j] * table;
            }
            if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
          } else
            ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
            evdwl *= factor_lj;
          } else
            evdwl = 0.0;
        }
        if (evflag) {
          double ecoulhalf = 0.5 * ecoul;
          if (newton_pair || i < nlocal) {
            cl_atom[i] += ecoulhalf;
          }
          if (newton_pair || j < nlocal) {
            cl_atom[j] += ecoulhalf;
          }
          ev_tally(i, j, nlocal, newton_pair, evdwl, ecoul, fpair, delx, dely, delz);
        }
      } // if rsq < r_cut
    }   // jj

    /* ------ add the self interaction energy ------ */
    // printf("self_flag0\n");
    if (self_flag) {
      value = qtmp * qtmp * self_Greenf[itype];
      cl_atom[i] += value;
      eatom[i] += value;
      force->pair->eng_coul += value;
    }
    /* ------ add the self interaction energy ------ */
  } // ii

  if (vflag_fdotr) virial_fdotr_compute();
}

/* -------------------------------------------------------------------------
    Compute between All<=>All via the Gauss potential
------------------------------------------------------------------------- */

void PairLJCutPointLong::compute_Gauss(int eflag, int vflag)
{
  FUNC_MACRO(0);
  // GAflag;
  // return compute_Point(eflag, vflag);
  int i, ii, j, jj, inum, jnum, itype, jtype, itable;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double fraction, table;
  double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double grij, expm2, prefactor, erfc, qiqj;
  int *ilist, *jlist, *numneigh, **firstneigh, idx;
  double rsq;

  evdwl = ecoul = 0.0;
  if (eflag || vflag) {
    if (atom->nmax > maxeatom) {
      memory->destroy(cl_atom);
      memory->create(cl_atom, comm->nthreads * atom->nmax, "pair:cl_atom");
    }
    ev_setup(eflag, vflag);
  } else
    evflag = vflag_fdotr = 0;

  double** x = atom->x;
  double** f = atom->f;
  int* tag   = atom->tag;

  double* q            = atom->q;
  int* type            = atom->type;
  int nlocal           = atom->nlocal;
  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  const double qqrd2e  = force->qqrd2e;
  double guass_rij, guass_expm2, gauss_erfc, value;

  inum       = list->inum;
  ilist      = list->ilist;
  numneigh   = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i     = ilist[ii];
    qtmp  = q[i];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum  = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j           = jlist[jj];
      factor_lj   = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx  = xtmp - x[j][0];
      dely  = ytmp - x[j][1];
      delz  = ztmp - x[j][2];
      rsq   = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;
        if (rsq < cut_coulsq) {
          if (!ncoultablebits || rsq <= tabinnersq) {
            r         = sqrt(rsq);
            prefactor = qqrd2e * qtmp * q[j] / r;
            grij      = g_ewald * r;
            expm2     = expmsq(grij);
            erfc      = my_erfcx(grij) * expm2;
            forcecoul = prefactor * (erfc + MY_ISPI4 * grij * expm2);

            // P <-> P interaction
            if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
              gauss_erfc = 0;
              if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;
              // P <-> G or G <-> G interacion
            } else {
              guass_rij          = eta_mix_table[itype][jtype] * r;
              guass_expm2        = expmsq(guass_rij);
              gauss_erfc         = my_erfcx(guass_rij) * guass_expm2;
              double f_prefactor = prefactor * factor_coul;

              forcecoul -= f_prefactor * (gauss_erfc + MY_ISPI4 * guass_rij * guass_expm2);
              if (factor_coul < 1.0) forcecoul -= prefactor - f_prefactor;
            }
          } else {
            // error->all(FLERR,"No table loopup for Lj/Cut/Gauss/long!");
            union_int_float_t rsq_lookup;
            idx = itype * n_plus + jtype;
            // printf("%d, %d %d\n", itype, jtype, idx);
            rsq_lookup.f = rsq;
            itable       = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            qiqj     = atom->q[i] * atom->q[j];
            if (factor_coul < 1.0) {
              double gtable  = ctable_vec[idx][itable] + fraction * dctable_vec[idx][itable];
              double c_table = ctable[itable] + fraction * dctable[itable];

              table = ftable[itable] + fraction * dftable[itable] + factor_coul * gtable;
              table -= (1.0 - factor_coul) * c_table;
              prefactor = qiqj * c_table;
            } else {
              table = ftable_vec[idx][itable] + fraction * dftable_vec[idx][itable];
            }
            forcecoul = qiqj * table;
          }
        } else
          forcecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv   = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        } else
          forcelj = 0.0;

        fpair = (forcecoul + factor_lj * forcelj) * r2inv;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) {
          if (rsq < cut_coulsq) {
            if (!ncoultablebits || rsq <= tabinnersq) {
              ecoul = prefactor * (erfc - factor_coul * gauss_erfc);
            } else {
              if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
                table = etable[itable] + fraction * detable[itable];
                ecoul = qiqj * table;
              } else {
                double table0 = etable[itable] + fraction * detable[itable];
                table         = etable_vec[idx][itable] + fraction * detable_vec[idx][itable];
                table         = table0 + factor_coul * (table - table0);
                ecoul         = qiqj * table;
              }
            }
            if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
          } else
            ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
            evdwl *= factor_lj;
          } else
            evdwl = 0.0;
        }
        if (evflag) {
          double ecoulhalf = 0.5 * ecoul;
          if (newton_pair || i < nlocal) {
            cl_atom[i] += ecoulhalf;
          }
          if (newton_pair || j < nlocal) {
            cl_atom[j] += ecoulhalf;
          }
          ev_tally(i, j, nlocal, newton_pair, evdwl, ecoul, fpair, delx, dely, delz);
        }
      } // if rsq < r_cut
    }   // jj
    /* ------ add the self interaction energy ------ */
    // printf("self_flag0\n");
    if (self_flag) {
      value = qtmp * qtmp * self_Greenf[itype];
      cl_atom[i] += value;
      eatom[i] += value;
      force->pair->eng_coul += value;
    }
    /* ------ add the self interaction energy ------ */
  } // ii

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutPointLong::allocate()
{
  // printf("PairLJCutPointLong::allocate()\n");
  allocated = 1;
  int n     = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  memory->create(cut_lj, n + 1, n + 1, "pair:cut_lj");
  memory->create(cut_ljsq, n + 1, n + 1, "pair:cut_ljsq");
  memory->create(epsilon, n + 1, n + 1, "pair:epsilon");
  memory->create(sigma, n + 1, n + 1, "pair:sigma");
  memory->create(lj1, n + 1, n + 1, "pair:lj1");
  memory->create(lj2, n + 1, n + 1, "pair:lj2");
  memory->create(lj3, n + 1, n + 1, "pair:lj3");
  memory->create(lj4, n + 1, n + 1, "pair:lj4");
  memory->create(offset, n + 1, n + 1, "pair:offset");
  memory->create(cl_atom, maxeatom, "pair:cl_atom");
  printf("end of function()\n");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutPointLong::coeff(int narg, char** arg)
{
  if (narg < 4 || narg > 5) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  double sigma_one   = utils::numeric(FLERR, arg[3], false, lmp);

  double cut_lj_one = cut_lj_global;
  if (narg == 5) cut_lj_one = utils::numeric(FLERR, arg[4], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j]   = sigma_one;
      cut_lj[i][j]  = cut_lj_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutPointLong::init_style()
{
  // printf("init_style()\n");
  if (!atom->q_flag) error->all(FLERR, "Pair style lj/cut/Gauss/long requires atom attribute q");

  // request regular or rRESPA neighbor list

  int irequest;
  int respa = 0;
  int me    = comm->me;

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
    if (((Respa*)update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa*)update->integrate)->level_middle >= 0) respa = 2;
  }

  irequest = neighbor->request(this, instance_me);

  // neighbor->requests[irequest]->newton = 2; // 2 for off;

  if (respa >= 1) {
    neighbor->requests[irequest]->respaouter = 1;
    neighbor->requests[irequest]->respainner = 1;
  }
  if (respa == 2) neighbor->requests[irequest]->respamiddle = 1;

  cut_coulsq = cut_coul * cut_coul;

  // set rRESPA cutoffs

  if (utils::strmatch(update->integrate_style, "respa") && ((Respa*)update->integrate)->level_inner >= 0)
    cut_respa = ((Respa*)update->integrate)->cutoff;
  else
    cut_respa = nullptr;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == nullptr) {
    error->all(FLERR, "Pair style may require a KSpace style");
    g_ewald = 0;
  } else
    g_ewald = force->kspace->g_ewald;

  int n = atom->ntypes;
  double qscale;
  if (self_flag == 1) {
    qscale = force->qqrd2e * MY_ISPI4;
    // For real units,  1.0 eV/A -> 23.0605419453 kcal/mol-A and 1 hatree/A
    // -> 27.21138602 eV/A
    double qe2f = (27.21138602 * force->qe2f);
    if (me == 0) {
      utils::logmesg(lmp, "Pair-lj/cut/gauss/long: Self_Capacitance = 1 / k_Self:\n");
    }
    for (int atype = 1; atype <= n; ++atype) {
      // self_Greenf should be halved, to be similar to the defination of
      // capacitance
      self_Greenf[atype] *= 0.5 * qscale;
      if (me == 0) {
        if (eta[atype] == 0.0)
          utils::logmesg(lmp,
                         "Point-Q [{}]: eta =     inf, k_Self:{:12.4f}  [Energy/e^2] "
                         "Self-Cap.:{:10.4f} [e^2/Hatree]\n",
                         atype, self_Greenf[atype], qe2f / self_Greenf[atype]);
        else
          utils::logmesg(lmp,
                         "Gauss-Q [{}]: eta = {:8.5g} ,   k_Self:{:12.4f}  [Energy/e^2] "
                         "Self-Cap.:{:10.4f}[e^2/Hatree]\n",
                         atype, eta[atype], self_Greenf[atype], qe2f / self_Greenf[atype]);
      }
    }
    self_flag = 2; // Set to 2 to avoid the double multiply
  }
  // setup force tables
  if (ncoultablebits) {
    // init_tables(cut_coul, cut_respa);
    if (gauss_flag) init_special_tables();
    init_tables(cut_coul, cut_respa);

    /*
    int ntable = 1;
    for (int i = 0; i < ncoultablebits; i++)
      ntable *= 2;
    for (size_t i = 0; i < ntable; i++) {
      printf("%d %.12f %.12f\n", i, rtable[i], ctable[i]);
    }
    */
  }
  if (comm->me == 0) utils::logmesg(lmp, "[Pair] ndisptablebits = {}\n", ncoultablebits);
}

/* ----------------------------------------------------------------------
   compute SOL to GA potential
------------------------------------------------------------------------- */

double PairLJCutPointLong::pair_SOLtoGA_energy(const int FIX_groupbit, const bool GG_Flag, const bool POT_Flag)
{
  FUNC_MACRO(0);
  // printf("sv_SOL_flag = %d, electro = %x\n", sv_SOL_flag, electro);
  if (sv_SOL_flag && electro != nullptr) {
    // printf("allocate\n");
    int* numneigh = list->numneigh;
    int localGA   = electro->FIX->localGA_num;
    pair_num      = 0;
    for (int i = 0; i < localGA; i++) {
      pair_num += numneigh[i];
    }
    pair_num = 1.2 * pair_num;
    if (force->newton_pair) {
      pair_num *= 2;
    }
    if (pair_num > pair_num_allocate) {
      memory->create(pair_ij, pair_num * 2, "PairLJCutPointLong::pair_ij");
      memory->create(pair_fcoul, pair_num, "PairLJCutPointLong::pair_fcoul");
      memory->create(pair_fvdwl, pair_num, "PairLJCutPointLong::pair_fvdwl");
      memory->create(pair_ecoul, pair_num, "PairLJCutPointLong::pair_coul");
      memory->create(pair_evdwl, pair_num, "PairLJCutPointLong::pair_evdwl");
      memory->create(pair_dxyz, pair_num * 4, "PairLJCutPointLong::pair_dxyz");
      pair_num_allocate = pair_num;
    }
    pair_num = 0;
  }

  if (gauss_flag)
    return pair_SOLtoGA_energy_Gauss(FIX_groupbit, GG_Flag, POT_Flag);
  else
    return pair_SOLtoGA_energy_Point(FIX_groupbit, GG_Flag, POT_Flag);
}

/* ----------------------------------------------------------------------
   compute SOL to GA potential using a Coulomb interaction;
   bool GG_Flag:  controls the GA<=>GA interaction;
   bool POT_Flag: controls output as potential of energy
------------------------------------------------------------------------- */

double PairLJCutPointLong::pair_SOLtoGA_energy_Point(const int FIX_groupbit, const bool GG_Flag, const bool POT_Flag)
{
  bool pair_eflag = update->eflag_atom == update->ntimestep;
  int iseq, iGA;
  int i, ii, j, jj, inum, jnum, itype, jtype, itable;
  double qtmp_half, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double fraction, table;
  double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double grij, expm2, prefactor, erfc;
  int *ilist, *jlist, *numneigh, **firstneigh; // idx;
  bool iGA_Flag, jGA_Flag;
  double rsq;

  double** x = atom->x;
  double* q  = atom->q;
  int* tag   = atom->tag;
  int* type  = atom->type;
  int* mask  = atom->mask;

  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  const double qqrd2e  = force->qqrd2e;
  double guass_rij, guass_expm2, value; // gauss_erfc

  inum         = list->inum;
  ilist        = list->ilist;
  numneigh     = list->numneigh;
  firstneigh   = list->firstneigh;
  bool ij_flag = false;

  int* ij_ptr       = pair_ij + pair_num * 2;
  double* ecoul_ptr = pair_ecoul + pair_num;
  double* evdwl_ptr = pair_evdwl + pair_num;
  double* fcoul_ptr = pair_fcoul + pair_num;
  double* fvdwl_ptr = pair_fvdwl + pair_num;
  double* dxyz_ptr  = pair_dxyz + pair_num * 4;

  for (ii = 0; ii < inum; ii++) {
    i     = ilist[ii];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum  = numneigh[i];

    iGA_Flag = (mask[i] & FIX_groupbit) != 0;

    if (GG_Flag) {
      ecoul = self_Greenf[itype] * q[i];
    } else {
      ecoul = 0;
    }

    if (POT_Flag) {
      cl_atom[i] += ecoul;
    } else {
      cl_atom[i] += ecoul * q[i];
    }

    qtmp_half = q[i] * 0.5;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype    = type[j];
      jGA_Flag = (mask[j] & FIX_groupbit) != 0;

      // GA<=>SOL or SOL<=>GA
      if (iGA_Flag != jGA_Flag) {
        ij_flag = true;
      } else {
        // GA<=>GA or SOL<=>SOL
        // iGA && GG_Flag => GA<=>GA if GA_Flag is true
        if (!(iGA_Flag && GG_Flag)) {
          continue;
        }
        ij_flag = false;
      }
      // factor_lj   = special_lj[sbmask(j)];
      // factor_coul = special_coul[sbmask(j)];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq  = delx * delx + dely * dely + delz * delz;

      bool coulsq_flag = true;
      if (rsq < cut_coulsq) {
        if (!ncoultablebits || rsq <= tabinnersq) {
          r         = sqrt(rsq);
          prefactor = qqrd2e / r;
          grij      = g_ewald * r;
          expm2     = expmsq(grij);
          erfc      = _my_erfcx(grij) * expm2; // grij > 0 with g_ewald > 0;
          ecoul     = prefactor * erfc;
          if (sv_SOL_flag) {
            forcecoul = prefactor * (erfc + MY_ISPI4 * grij * expm2);
          }
        } else {
          union_int_float_t rsq_lookup;
          rsq_lookup.f = rsq;
          itable       = rsq_lookup.i & ncoulmask;
          itable >>= ncoulshiftbits;
          fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
          ecoul    = etable[itable] + fraction * detable[itable];
          if (sv_SOL_flag) {
            forcecoul = ftable[itable] + fraction * dftable[itable];
          }
        }
      } else {
        if (sv_SOL_flag == false) continue;
        coulsq_flag = false;
        ecoul = forcecoul = 0.0;
      }

      if (sv_SOL_flag == true) {
        if (rsq < cut_ljsq[itype][jtype]) {
          r2inv   = 1 / rsq;
          r6inv   = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
          if (pair_eflag) {
            evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
          } else {
            evdwl = 0.0;
          }
        } else {
          if (coulsq_flag == false) {
            continue;
          }
          forcelj = evdwl = 0.0;
        }

        *(evdwl_ptr++) = evdwl;
        *(fvdwl_ptr++) = forcelj;
      }
      // printf("ecoul = %.12f, qtmp = %.12f, %.12f, %.12f\n", ecoul, qtmp,
      // self_Greenf[type[i]], self_Greenf[type[j]]);
      double ecoul_i, ecoul_j;
      if (POT_Flag) {
        ecoul_i = 0.5 * q[j] * ecoul;
        ecoul_j = qtmp_half * ecoul;
        cl_atom[i] += ecoul_i;
        cl_atom[j] += ecoul_j;
      } else {
        if (iGA_Flag) {
          ecoul_i = ecoul * q[j];
          ecoul   = ecoul_i * qtmp_half;
          ecoul_i = 0.5 * ecoul_i;
        } else {
          ecoul_j = ecoul * qtmp_half;
          ecoul   = ecoul_j * q[j];
        }
        // ecoul *= qtmp_half * q[j];
        cl_atom[i] += ecoul;
        cl_atom[j] += ecoul;
      }

      if (sv_SOL_flag && ij_flag) {

        if (iGA_Flag) {
          *(ecoul_ptr++) = ecoul_i;
          forcecoul *= q[j];
          *(ij_ptr++)   = i;
          *(ij_ptr++)   = j;
          *(dxyz_ptr++) = delx;
          *(dxyz_ptr++) = dely;
          *(dxyz_ptr++) = delz;
        } else {
          *(ecoul_ptr++) = ecoul_j;
          forcecoul *= q[i];
          *(ij_ptr++)   = j;
          *(ij_ptr++)   = i;
          *(dxyz_ptr++) = -delx;
          *(dxyz_ptr++) = -dely;
          *(dxyz_ptr++) = -delz;
        }
        *(dxyz_ptr++)  = r2inv;
        *(fcoul_ptr++) = forcecoul;
      }
    } // jj
  }   // ii
  if (sv_SOL_flag) {
    sv_SOL_saved = true;
    pair_num     = ecoul_ptr - pair_ecoul;
    if (pair_num > pair_num_allocate) {
      error->one(FLERR, "The total pair num exceeds the pre-allocated size");
    }
  }
  return 0.0;
}

/* ----------------------------------------------------------------------
   compute SOL to GA potential using a Gauss interaction;
   bool GG_Flag:  controls the GA<=>GA interaction;
   bool POT_Flag: controls output as potential of energy
------------------------------------------------------------------------- */

double PairLJCutPointLong::pair_SOLtoGA_energy_Gauss(const int FIX_groupbit, const bool GG_Flag, const bool POT_Flag)
{
  // GAflag;
  // return pair_SOLtoGA_energy_Point(FIX_groupbit, GG_Flag, POT_Flag);
  bool pair_eflag = update->eflag_atom == update->ntimestep;
  int iseq, iGA;
  int i, ii, j, jj, inum, jnum, itype, jtype, itable;
  double qtmp_half, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double fraction, table;
  double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double grij, expm2, prefactor, erfc, gauss_erfc;
  int *ilist, *jlist, *numneigh, **firstneigh, idx;
  bool iGA_Flag, jGA_Flag;
  double rsq;

  double** x = atom->x;
  double* q  = atom->q;
  int* tag   = atom->tag;
  int* type  = atom->type;
  int* mask  = atom->mask;

  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  const double qqrd2e  = force->qqrd2e;
  double guass_rij, guass_expm2, value; // gauss_erfc

  inum         = list->inum;
  ilist        = list->ilist;
  numneigh     = list->numneigh;
  firstneigh   = list->firstneigh;
  bool ij_flag = false;

  int* ij_ptr       = pair_ij + pair_num * 2;
  double* ecoul_ptr = pair_ecoul + pair_num;
  double* evdwl_ptr = pair_evdwl + pair_num;
  double* fcoul_ptr = pair_fcoul + pair_num;
  double* fvdwl_ptr = pair_fvdwl + pair_num;
  double* dxyz_ptr  = pair_dxyz + pair_num * 4;

  for (ii = 0; ii < inum; ii++) {
    i     = ilist[ii];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum  = numneigh[i];

    iGA_Flag = (mask[i] & FIX_groupbit) != 0;

    if (GG_Flag) {
      ecoul = self_Greenf[itype] * q[i];
    } else {
      ecoul = 0;
    }

    if (POT_Flag) {
      cl_atom[i] += ecoul;
    } else {
      cl_atom[i] += ecoul * q[i];
    }

    qtmp_half = q[i] * 0.5;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype    = type[j];
      jGA_Flag = (mask[j] & FIX_groupbit) != 0;

      // GA<=>SOL or SOL<=>GA
      if (iGA_Flag != jGA_Flag) {
        ij_flag = true;
      } else {
        // GA<=>GA or SOL<=>SOL
        // iGA && GG_Flag => GA<=>GA if GA_Flag is true
        if (!(iGA_Flag && GG_Flag)) {
          continue;
        }
        ij_flag = false;
      }
      // factor_lj   = special_lj[sbmask(j)];
      // factor_coul = special_coul[sbmask(j)];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq  = delx * delx + dely * dely + delz * delz;

      bool coulsq_flag = true;
      if (rsq < cut_coulsq) {
        if (!ncoultablebits || rsq <= tabinnersq) {
          r         = sqrt(rsq);
          prefactor = qqrd2e / r;
          grij      = g_ewald * r;
          expm2     = expmsq(grij);
          erfc      = _my_erfcx(grij) * expm2; // grij > 0 with g_ewald > 0;
          if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
            gauss_erfc = 0;
          } else {
            guass_rij   = eta_mix_table[itype][jtype] * r;
            guass_expm2 = expmsq(guass_rij);
            gauss_erfc  = my_erfcx(guass_rij) * guass_expm2;
          }
          ecoul = prefactor * (erfc - gauss_erfc);
          if (sv_SOL_flag) {
            forcecoul = prefactor * (erfc + MY_ISPI4 * grij * expm2);
          }
        } else {
          union_int_float_t rsq_lookup;
          rsq_lookup.f = rsq;
          idx          = itype * n_plus + jtype;
          itable       = rsq_lookup.i & ncoulmask;
          itable >>= ncoulshiftbits;
          fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
          ecoul    = etable_vec[idx][itable] + fraction * detable_vec[idx][itable];

          if (sv_SOL_flag) {
            forcecoul = ftable_vec[idx][itable] + fraction * dftable_vec[idx][itable];
          }
        }
      } else {
        if (sv_SOL_flag == false) continue;
        coulsq_flag = false;
        ecoul = forcecoul = 0.0;
      }

      if (sv_SOL_flag == true) {
        if (rsq < cut_ljsq[itype][jtype]) {
          r2inv   = 1 / rsq;
          r6inv   = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
          if (pair_eflag) {
            evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
          } else {
            evdwl = 0.0;
          }
        } else {
          if (coulsq_flag == false) {
            continue;
          }
          forcelj = evdwl = 0.0;
        }

        *(evdwl_ptr++) = evdwl;
        *(fvdwl_ptr++) = forcelj;
      }
      // printf("ecoul = %.12f, qtmp = %.12f, %.12f, %.12f\n", ecoul, qtmp,
      // self_Greenf[type[i]], self_Greenf[type[j]]);
      double ecoul_i, ecoul_j;
      if (POT_Flag) {
        ecoul_i = 0.5 * q[j] * ecoul;
        ecoul_j = qtmp_half * ecoul;
        cl_atom[i] += ecoul_i;
        cl_atom[j] += ecoul_j;
      } else {
        if (iGA_Flag) {
          ecoul_i = ecoul * q[j];
          ecoul   = ecoul_i * qtmp_half;
          ecoul_i = 0.5 * ecoul_i;
        } else {
          ecoul_j = ecoul * qtmp_half;
          ecoul   = ecoul_j * q[j];
        }
        // ecoul *= qtmp_half * q[j];
        cl_atom[i] += ecoul;
        cl_atom[j] += ecoul;
      }

      if (sv_SOL_flag && ij_flag) {

        if (iGA_Flag) {
          *(ecoul_ptr++) = ecoul_i;
          forcecoul *= q[j];
          *(ij_ptr++)   = i;
          *(ij_ptr++)   = j;
          *(dxyz_ptr++) = delx;
          *(dxyz_ptr++) = dely;
          *(dxyz_ptr++) = delz;
        } else {
          *(ecoul_ptr++) = ecoul_j;
          forcecoul *= q[i];
          *(ij_ptr++)   = j;
          *(ij_ptr++)   = i;
          *(dxyz_ptr++) = -delx;
          *(dxyz_ptr++) = -dely;
          *(dxyz_ptr++) = -delz;
        }
        *(dxyz_ptr++)  = r2inv;
        *(fcoul_ptr++) = forcecoul;
      }
    } // jj
  }   // ii
  if (sv_SOL_flag) {
    sv_SOL_saved    = true;
    sv_SOL_timestep = update->ntimestep;
    pair_num        = ecoul_ptr - pair_ecoul;
    if (pair_num > pair_num_allocate) {
      error->one(FLERR, "The total pair num exceeds the pre-allocated size");
    }
  }
  return 0.0;
}

/* ----------------------------------------------------------------------
   compute GA to GA potential using a Gauss interaction;
------------------------------------------------------------------------- */

double PairLJCutPointLong::pair_GAtoGA_energy(const int FIX_groupbit, const bool POT_Flag)
{
  if (gauss_flag)
    return pair_GAtoGA_energy_Gauss(FIX_groupbit, POT_Flag);
  else
    return pair_GAtoGA_energy_Point(FIX_groupbit, POT_Flag);
}

/* ----------------------------------------------------------------------
   compute GA to GA potential using a Coulomb interaction;
   bool POT_Flag: controls output as potential of energy
------------------------------------------------------------------------- */

double PairLJCutPointLong::pair_GAtoGA_energy_Point(const int FIX_groupbit, const bool POT_Flag)
{
  int iseq, iGA;
  int i, ii, j, jj, inum, jnum, itype, jtype, itable;
  double qtmp_half, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double fraction, table;
  double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double grij, expm2, prefactor, erfc;
  int *ilist, *jlist, *numneigh, **firstneigh; // idx;
  bool iGA_Flag, jGA_Flag;
  double rsq, qi_qj;

  double** x = atom->x;
  double* q  = atom->q;
  int* tag   = atom->tag;
  int* type  = atom->type;
  int* mask  = atom->mask;

  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  const double qqrd2e  = force->qqrd2e;
  double guass_rij, guass_expm2, value; // gauss_erfc

  inum       = list->inum;
  ilist      = list->ilist;
  numneigh   = list->numneigh;
  firstneigh = list->firstneigh;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    // qtmp = q[i];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum  = numneigh[i];

    if (!(mask[i] & FIX_groupbit)) continue;

    qtmp_half = 0.5 * q[i];
    ecoul     = self_Greenf[itype] * q[i];
    if (POT_Flag) {
      cl_atom[i] += ecoul;
    } else {
      cl_atom[i] += ecoul * q[i];
    }

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (!(mask[j] & FIX_groupbit)) continue;
      factor_lj   = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq  = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_coulsq) {
        if (!ncoultablebits || rsq <= tabinnersq) {
          r         = sqrt(rsq);
          prefactor = qqrd2e / r;
          grij      = g_ewald * r;
          expm2     = expmsq(grij);
          erfc      = _my_erfcx(grij) * expm2; // grij > 0 with g_ewald > 0;
          ecoul     = prefactor * erfc;
        } else {
          union_int_float_t rsq_lookup;
          rsq_lookup.f = rsq;
          itable       = rsq_lookup.i & ncoulmask;
          itable >>= ncoulshiftbits;
          fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
          ecoul    = etable[itable] + fraction * detable[itable];
        }
      } else {
        continue;
      }

      if (POT_Flag) {
        cl_atom[i] += 0.5 * q[j] * ecoul;
        cl_atom[j] += qtmp_half * ecoul;
      } else {
        ecoul *= qtmp_half * q[j];
        cl_atom[i] += ecoul;
        cl_atom[j] += ecoul;
      }
    } // jj
  }   // ii
  return 0.0;
}

/* ----------------------------------------------------------------------
   compute GA to GA potential using a Gauss interaction;
   bool POT_Flag: controls output as potential of energy
------------------------------------------------------------------------- */

double PairLJCutPointLong::pair_GAtoGA_energy_Gauss(const int FIX_groupbit, const bool POT_Flag)
{
  // GAflag;
  // return pair_GAtoGA_energy_Point(FIX_groupbit, POT_Flag);
  int iseq, iGA;
  int i, ii, j, jj, inum, jnum, itype, jtype, itable;
  double qtmp_half, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double fraction, table;
  double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double grij, expm2, prefactor, erfc;
  int *ilist, *jlist, *numneigh, **firstneigh, idx;
  bool iGA_Flag, jGA_Flag;
  double rsq, qi_qj;

  double** x = atom->x;
  double* q  = atom->q;
  int* tag   = atom->tag;
  int* type  = atom->type;
  int* mask  = atom->mask;

  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  int newton_pair      = force->newton_pair;
  const double qqrd2e  = force->qqrd2e;
  double guass_rij, guass_expm2, value, gauss_erfc;

  inum       = list->inum;
  ilist      = list->ilist;
  numneigh   = list->numneigh;
  firstneigh = list->firstneigh;
  for (ii = 0; ii < inum; ii++) {
    i     = ilist[ii];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum  = numneigh[i];

    if (!(mask[i] & FIX_groupbit)) continue;

    qtmp_half = 0.5 * q[i];
    ecoul     = self_Greenf[itype] * q[i];
    if (POT_Flag) {
      cl_atom[i] += ecoul;
    } else {
      cl_atom[i] += ecoul * q[i];
    }

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (!(mask[j] & FIX_groupbit)) continue;
      factor_lj   = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq  = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_coulsq) {
        jtype = type[j];
        if (!ncoultablebits || rsq <= tabinnersq) {
          r         = sqrt(rsq);
          prefactor = qqrd2e / r;
          grij      = g_ewald * r;
          expm2     = expmsq(grij);
          erfc      = _my_erfcx(grij) * expm2; // grij > 0 with g_ewald > 0;
          if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
            gauss_erfc = 0;
          } else {
            guass_rij   = eta_mix_table[itype][jtype] * r;
            guass_expm2 = expmsq(guass_rij);
            gauss_erfc  = my_erfcx(guass_rij) * guass_expm2;
          }
          ecoul = prefactor * (erfc - gauss_erfc);
        } else {
          union_int_float_t rsq_lookup;
          idx          = itype * n_plus + jtype;
          rsq_lookup.f = rsq;
          itable       = rsq_lookup.i & ncoulmask;
          itable >>= ncoulshiftbits;
          fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
          ecoul    = etable_vec[idx][itable] + fraction * detable_vec[idx][itable];
        }
      } else {
        continue;
      }

      if (POT_Flag) {
        cl_atom[i] += 0.5 * q[j] * ecoul;
        cl_atom[j] += qtmp_half * ecoul;
      } else {
        ecoul *= qtmp_half * q[j];
        cl_atom[i] += ecoul;
        cl_atom[j] += ecoul;
      }
    } // jj
  }   // ii
  return 0.0;
}

/* ----------------------------------------------------------------------
   compute pair_wise potential between two charge of itype and jtype
------------------------------------------------------------------------- */

double PairLJCutPointLong::compute_single(int itype, int jtype, double rsq)
{
  double ecoul, gauss_erfc;
  const double qqrd2e = force->qqrd2e;
  int idx, itable;
  if (rsq == 0) {
    return self_Greenf[itype];
  }
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq) {
      double r         = sqrt(rsq);
      double prefactor = qqrd2e / r;
      double grij      = g_ewald * r;
      double expm2     = expmsq(grij);
      double erfc      = _my_erfcx(grij) * expm2; // grij > 0 with g_ewald > 0;
      if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
        gauss_erfc = 0;
      } else {
        double guass_rij   = eta_mix_table[itype][jtype] * r;
        double guass_expm2 = expmsq(guass_rij);
        gauss_erfc         = my_erfcx(guass_rij) * guass_expm2;
      }
      ecoul = prefactor * (erfc - gauss_erfc);
    } else {
      if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
        union_int_float_t rsq_lookup;
        rsq_lookup.f = rsq;
        itable       = rsq_lookup.i & ncoulmask;
        itable >>= ncoulshiftbits;
        double fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
        ecoul           = etable[itable] + fraction * detable[itable];
      } else {
        union_int_float_t rsq_lookup;
        idx          = itype * n_plus + jtype;
        rsq_lookup.f = rsq;
        itable       = rsq_lookup.i & ncoulmask;
        itable >>= ncoulshiftbits;
        double fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
        ecoul           = etable_vec[idx][itable] + fraction * detable_vec[idx][itable];
      }
    }
  } else {
    ecoul = 0.0;
  }
  return ecoul;
};

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutPointLong::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j], sigma[i][i], sigma[j][j]);
    sigma[i][j]   = mix_distance(sigma[i][i], sigma[j][j]);
    cut_lj[i][j]  = mix_distance(cut_lj[i][i], cut_lj[j][j]);
  }

  // include TIP4P qdist in full cutoff, qdist = 0.0 if not TIP4P

  double cut     = MAX(cut_lj[i][j], cut_coul + 2.0 * qdist);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);

  if (offset_flag && (cut_lj[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio, 12.0) - pow(ratio, 6.0));
  } else
    offset[i][j] = 0.0;

  cut_ljsq[j][i] = cut_ljsq[i][j];
  lj1[j][i]      = lj1[i][j];
  lj2[j][i]      = lj2[i][j];
  lj3[j][i]      = lj3[i][j];
  lj4[j][i]      = lj4[i][j];
  offset[j][i]   = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && MIN(cut_lj[i][j], cut_coul) < cut_respa[3]) error->all(FLERR, "Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int* type  = atom->type;
    int nlocal = atom->nlocal;

    double count[2], all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count, all, 2, MPI_DOUBLE, MPI_SUM, world);

    double sig2 = sigma[i][j] * sigma[i][j];
    double sig6 = sig2 * sig2 * sig2;
    double rc3  = cut_lj[i][j] * cut_lj[i][j] * cut_lj[i][j];
    double rc6  = rc3 * rc3;
    double rc9  = rc3 * rc6;
    etail_ij    = 8.0 * MY_PI * all[0] * all[1] * epsilon[i][j] * sig6 * (sig6 - 3.0 * rc6) / (9.0 * rc9);
    ptail_ij    = 16.0 * MY_PI * all[0] * all[1] * epsilon[i][j] * sig6 * (2.0 * sig6 - 3.0 * rc6) / (9.0 * rc9);
  }

  return cut;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutPointLong::write_restart(FILE* fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j], sizeof(double), 1, fp);
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&cut_lj[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutPointLong::read_restart(FILE* fp)
{
  read_restart_settings(fp);

  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &epsilon[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut_lj[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut_lj[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutPointLong::write_restart_settings(FILE* fp)
{
  fwrite(&cut_lj_global, sizeof(double), 1, fp);
  fwrite(&cut_coul, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  fwrite(&tail_flag, sizeof(int), 1, fp);
  fwrite(&ncoultablebits, sizeof(int), 1, fp);
  fwrite(&tabinner, sizeof(double), 1, fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutPointLong::read_restart_settings(FILE* fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &cut_lj_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &cut_coul, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &ncoultablebits, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &tabinner, sizeof(double), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_lj_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cut_coul, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&ncoultablebits, 1, MPI_INT, 0, world);
  MPI_Bcast(&tabinner, 1, MPI_DOUBLE, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJCutPointLong::write_data(FILE* fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g\n", i, epsilon[i][i], sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJCutPointLong::write_data_all(FILE* fp)
{
  fprintf(fp, "cut_lj_global: %g cut_coul: %g\n", cut_lj_global, cut_coul);
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g\n", i, eta[i], sigma_width[i]);
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g\n", i, j, epsilon[i][j], sigma[i][j], cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */
void* PairLJCutPointLong::extract(const char* str, int& dim)
{
  dim = 2;
  if (strcmp(str, "epsilon") == 0) return (void*)epsilon;
  if (strcmp(str, "sigma") == 0) return (void*)sigma;
  if (strcmp(str, "cut_ljsq") == 0) return (void*)cut_ljsq;
  if (strcmp(str, "cut_lj") == 0) return (void*)cut_lj;
  if (strcmp(str, "eta_mix_table") == 0) return (void*)eta_mix_table;

  dim = 1;
  if (strcmp(str, "eta") == 0) return (void*)eta;
  if (strcmp(str, "sigma_width") == 0) return (void*)sigma_width;
  if (strcmp(str, "self_Greenf") == 0) {
    return (void*)self_Greenf;
  }

  dim = 0;
  if (strcmp(str, "cut_coul") == 0) return (void*)&cut_coul;
  if (strcmp(str, "cut_coulsq") == 0) return (void*)&cut_coulsq;
  if (strcmp(str, "g_ewald") == 0) return (void*)&g_ewald;

  return nullptr;
}

/* ---------------------------------------------------------------------- */

double PairLJCutPointLong::single(int i, int j, int itype, int jtype, double rsq, double factor_coul, double factor_lj, double& fforce)
{
  double r2inv, r6inv, r, grij, expm2, erfc, prefactor;
  double guass_rij, guass_expm2, gauss_erfc, qiqj;
  double fraction, table, forcecoul, forcelj, phicoul, philj;
  int itable, idx;

  r2inv = 1.0 / rsq;
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq) {
      r         = sqrt(rsq);
      prefactor = force->qqrd2e * atom->q[i] * atom->q[j] / r;

      // if(g_ewald != 0) {
      grij      = g_ewald * r;
      expm2     = expmsq(grij);
      erfc      = _my_erfcx(grij) * expm2;
      forcecoul = prefactor * (erfc + MY_ISPI4 * grij * expm2 - 1.0);
      // P <-> P interaction
      if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
        gauss_erfc = 0;
        if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;
        // P <-> G or G <-> G interacion
      } else {
        guass_rij          = eta_mix_table[itype][jtype] * r;
        guass_expm2        = expmsq(guass_rij);
        gauss_erfc         = my_erfcx(guass_rij) * guass_expm2;
        double f_prefactor = prefactor * factor_coul;

        forcecoul -= f_prefactor * (gauss_erfc + MY_ISPI4 * guass_rij * guass_expm2);
        if (factor_coul < 1.0) forcecoul -= prefactor - f_prefactor;
      }

    } else {
      if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
        union_int_float_t rsq_lookup;
        rsq_lookup.f = rsq;
        itable       = rsq_lookup.i & ncoulmask;
        itable >>= ncoulshiftbits;
        table       = ftable[itable] + fraction * dftable[itable];
        double qiqj = atom->q[i] * atom->q[j];
        forcecoul   = qiqj * table;
        if (factor_coul < 1.0) {
          table     = ctable[itable] + fraction * dctable[itable];
          prefactor = qiqj * table;
          forcecoul -= (1.0 - factor_coul) * prefactor;
        }
      } else {
        union_int_float_t rsq_lookup;
        idx = itype * n_plus + jtype;
        // printf("%d, %d %d\n", itype, jtype, idx);
        rsq_lookup.f = rsq;
        itable       = rsq_lookup.i & ncoulmask;
        itable >>= ncoulshiftbits;
        fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
        qiqj     = atom->q[i] * atom->q[j];
        if (factor_coul < 1.0) {
          double gtable  = ctable_vec[idx][itable] + fraction * dctable_vec[idx][itable];
          double c_table = ctable[itable] + fraction * dctable[itable];

          table = ftable[itable] + fraction * dftable[itable] + factor_coul * gtable;
          table -= (1.0 - factor_coul) * c_table;
          prefactor = qiqj * c_table;
        } else {
          table = ftable_vec[idx][itable] + fraction * dftable_vec[idx][itable];
        }
        forcecoul = qiqj * table;
      }
    }
  } else
    forcecoul = 0.0;

  double eng = 0.0;
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq)
      phicoul = prefactor * (erfc - factor_coul * gauss_erfc);
    else {
      // error->all(FLERR,"No table loopup for Lj/Cut/Gauss/long!");
      if (isGCharge[itype] == P_CHARGE && isGCharge[jtype] == P_CHARGE) {
        table   = etable[itable] + fraction * detable[itable];
        phicoul = qiqj * table;
      } else {
        double table0 = etable[itable] + fraction * detable[itable];
        table         = etable_vec[idx][itable] + fraction * detable_vec[idx][itable];
        table         = table0 + factor_coul * (table - table0);
        phicoul       = qiqj * table;
      }
    }
    if (factor_coul < 1.0) phicoul -= (1.0 - factor_coul) * prefactor;
    eng += phicoul;
  }

  /* -------- End of Skip calculation interaction ------------ */

  if (rsq < cut_ljsq[itype][jtype]) {
    r6inv   = r2inv * r2inv * r2inv;
    forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
  } else
    forcelj = 0.0;

  fforce = (forcecoul + factor_lj * forcelj) * r2inv;

  if (rsq < cut_ljsq[itype][jtype]) {
    philj = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
    eng += factor_lj * philj;
  }

  return eng;
}
