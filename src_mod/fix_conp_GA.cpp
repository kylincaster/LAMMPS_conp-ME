/* ---------------------------------------------------------------------
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

   FixStyle("conp/GA", FixConpGA)
   Fix to deform the SR-CPM simulation.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Version: Sep/22/2014
   Zhenxing Wang(KU)
   Version: Oct/25/2016
   Kylin JIANG(WUST)
   Reduce the memory cost by implement a shared-memory distribution through MPI
communicator. Convert the large array from 2D into 1D also accelerate the
initialtion time. The extra setup flag was added, i.e. the checkpoint for binary
I/O

   Version: 1.06 Data: Jan/16/2019
   Fixed the Bugs in the shared-memory distribution through MPI communicator
   and update the performance via MPI_scatterv()
   remove the data copy for cn and sn arrays into my_cn and my_sn

  Version: 1.06 Data: June/16/2019
  External Modification Add an cl_atom in atom.h to store pure electro_coulomibc
energy

  Move "int bordergroup" in comm.h from protected membemer to public one

  Version: 1.07 Data: June/16/2020
  Add a validation check for electrode atom potential.
  fix small bug by added cl_atom communication with "newton on" case

  Version: 1.08 Data: Sep/24/2020
  Fix the bugs on the potential validation modulus
  that Lammps only calcualte the per-atom energy at eflag/2 > 0
  eflag is controlled by integrate.cpp and envoked by "compute"
  Fix the bugs on AAA matrix calculation.
  The self-interaction energy is removed from calculated AAA matrix.

  Version: 1.27 Data: April/02/2021
  Double check the following two run to make use of the same kspace calculation.

------------------------------------------------------------------------- */
#include "fix_conp_GA.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "kspace.h"
#include "math.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "modify.h"
#include "mpi.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "respa.h"
#include "stddef.h"
#include "stdlib.h"
#include "update.h"
#include "variable.h"

#include "debug_func.h"
#include "electro_control_GA.h"

#include "iostream"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
// #include <gsl/gsl_sf.h>
#ifdef SCALAPACK
// #include "blacs.h"
#endif

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum { REV_Q, REV_COUL, REV_COUL_RM, REV_FORCE };

// #define INTEL_MPI
#define FIX_CONPGA_VERSION 1.68
#define xstr(s) str(s)
#define str(s) #s

//#define EWALD_F   1.12837917 // EWALD_F = 2/sqrt(Pi)
//#define INITIAL_VECTOR_SIZE 200

extern "C" {
// #ifndef MKL_ILP64
// For sequential blas
void dgetrf_(const MKL_INT* M, const MKL_INT* N, double* A, const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info);
void dgetri_(const MKL_INT* N, double* A, const MKL_INT* lda, const MKL_INT* ipiv, double* work, const MKL_INT* lwork, MKL_INT* info);
void dsysv_(char* uplo, MKL_INT* n, MKL_INT* nrhs, double* a, MKL_INT* lda, MKL_INT* ipiv, double* b, MKL_INT* ldb, double* work, MKL_INT* lwork,
            MKL_INT* info);
void dgemv_(char* ta, MKL_INT* m, MKL_INT* n, double* alpha, double* a, MKL_INT* tda, double* x, MKL_INT* incx, double* beta, double* y, MKL_INT* incy);
void dgesv_(const MKL_INT* N, const MKL_INT* NRHS, const double* A, const MKL_INT* lda, const MKL_INT* ipiv, double* B, const MKL_INT* LDB, MKL_INT* info);
void dgels_(const char* trans, MKL_INT* m, MKL_INT* n, MKL_INT* nrhs, double* a, MKL_INT* lda, double* b, MKL_INT* ldb, double* work, MKL_INT* lwork,
            MKL_INT* info);
void dgelsd_(MKL_INT* m, MKL_INT* n, MKL_INT* nrhs, double* a, MKL_INT* lda, double* b, MKL_INT* ldb, double* s, double* rcond, MKL_INT* rank, double* work,
             MKL_INT* lwork, MKL_INT* iwork,
             MKL_INT* info); // #endif

#ifdef SCALAPACK
void Cblacs_pinfo(MKL_INT*, MKL_INT*);
void Cblacs_get(MKL_INT, MKL_INT, MKL_INT*);
void Cblacs_barrier(MKL_INT, const char*);
void Cblacs_gridinfo(MKL_INT, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*);
void Cblacs_gridinit(MKL_INT*, char[], MKL_INT, MKL_INT);
void Cblacs_gridmap(MKL_INT*, MKL_INT*, MKL_INT, MKL_INT, MKL_INT);
void Cblacs_abort(MKL_INT, MKL_INT);
void Cblacs_gridexit(MKL_INT);

MKL_INT Csys2blacs_handle(MPI_Comm);
MKL_INT Cblacs_pnum(MKL_INT, MKL_INT, MKL_INT);
MKL_INT Csys2blacs_handle(MPI_Comm);

void descinit_(MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*);
MKL_INT numroc_(const MKL_INT*, const MKL_INT*, MKL_INT*, const MKL_INT*, const MKL_INT*);

void pdgetrf_(const MKL_INT* M, const MKL_INT* N, double* A, const MKL_INT* IA, const MKL_INT* JA, MKL_INT* desca, MKL_INT* ipiv, MKL_INT* info);
void pdgetri_(const MKL_INT* N, double* A, const MKL_INT* IA, const MKL_INT* JA, MKL_INT* desca, MKL_INT* ipiv, double* work, const MKL_INT* lwork,
              MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info);
void pdgeadd_(char* TRANS, const MKL_INT* M, const MKL_INT* N, double* ALPHA, double* A, const MKL_INT* IA, const MKL_INT* JA, MKL_INT* DESCA, double* BETA,
              double* C, const MKL_INT* IC, const MKL_INT* JC, MKL_INT* DESCC);
#endif
}

static char* strnstr_l(const char* s, int N)
{
  int len_s = strnlen(s, N);
  for (int i = len_s; i >= 0; --i) {
    if (s[i] == '.') return (char*)(s + i);
  }
  return nullptr;
}

// comp function of sparsify_invAAA_EX
static bool fabs_comp(const std::pair<int, double>& a, const std::pair<int, double>& b) { return fabs(a.second) > fabs(b.second); }

static bool fabs_dcomp(const double& a, const double& b) { return fabs(a) > fabs(b); }

static bool val_comp(const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second < b.second; }

// comp function of sparsify_invAAA_EX
static bool int_GT(const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.first < b.first; }

// from https://stackoverflow.com/questions/23999797/implementing-strnstr
char* strnstr(const char* haystack, const char* needle, size_t len)
{
  int i;
  size_t needle_len;

  if (0 == (needle_len = strnlen(needle, len))) return (char*)haystack;

  for (i = 0; i <= (int)(len - needle_len); i++) {
    if ((haystack[0] == needle[0]) && (0 == strncmp(haystack, needle, needle_len))) return (char*)haystack;

    haystack++;
  }
  return nullptr;
}

FixConpGA::FixConpGA(LAMMPS* lmp, int narg, char** arg) : Fix(lmp, narg, arg)
{
  FUNC_MACRO(0)
  me = comm->me;

  if (me == 0) {
#ifndef CONP_NO_DEBUG
    utils::logmesg(lmp, "[ConpGA] start Fix/conpGA {} with -DEBUG- version\n", xstr(FIX_CONPGA_VERSION));
#else
    utils::logmesg(lmp, "[ConpGA] start Fix/conpGA {} with RELEASE verion, no debug \n", xstr(FIX_CONPGA_VERSION));
#endif

#if __cplusplus < 201103L // For C++11
    utils::logmesg(lmp, "[ConpGA] Using std::map from C++\n");
#else
    utils::logmesg(lmp, "[ConpGA] Using std::unordered_map from C++11\n");
#endif

#ifdef MKL_ILP64
    utils::logmesg(lmp, "[ConpGA] Using the MKL_ILP64 library\n");
#else
    utils::logmesg(lmp, "[ConpGA] Using the default 32bit matrix library for MKL and other BLAS library\n");
#endif

#ifdef SCALAPACK
    utils::logmesg(lmp, "[ConpGA] Linked with SCALAPACK\n");
#else
    utils::logmesg(lmp, "[ConpGA] Linked with LAPACK\n");
#endif

#ifdef INTEL_MPI
    utils::logmesg(lmp, "[ConpGA] Using the predifined INTEL_MPI macro\n");
#endif
  } // me == 0

  int i;
  double value;
  char info[512], *save_format;
  int n              = atom->ntypes;
  comm_forward       = 1; // very important
  comm_reverse       = 1;
  pcg_mat_sel        = PCG_MAT::NORMAL;
  newton_pair        = force->newton_pair;
  gewald_tune_factor = 1;
  work_me            = -1;
  maxiter            = 100;
  tolerance          = 0.000001;
  pseuo_tolerance    = 0.000001;
  tol_style          = TOL::NONE;

  // Deprecated
  /*
  nonneutral_removel_flag = false;
  self_Xi_flag            = false;
  */

  // Set AS false
  __env_alpha_flag    = false;
  postreat_flag       = false;
  calc_EleQ_flag      = false;
  force_analysis_flag = false;
  clear_force_flag    = false;
  valid_check_flag    = false;

  // Default setup
  symmetry_flag      = SYM::NONE;
  neutrality_flag    = true;
  update_Vext_flag   = false;
  unit_flag          = 1; // Default unit for [U] as eV
  thermo_energy      = 1;
  pair_flag          = true;
  firstgroup_flag    = true;
  selfGG_flag        = false;
  fast_bvec_flag     = true;
  Smatrix_flag       = false;
  Uchem_extract_flag = false;
  INFO_flag          = false;
  DEBUG_SOLtoGA      = true;
  DEBUG_GAtoGA       = true;

  Ele_GAnum           = nullptr;
  Ele_Cell_Model      = CELL::NONE;
  Ele_num             = 0;
  Ele_iter            = 0;
  valid_check_firstep = 5;
  check_tol           = 1e-8;
  Xi_self_index = setup_run = 0;
  pcg_neighbor_size_local   = 0;
  scalapack_num             = 10000;
  sparse_pcg_num            = 1000;
  size_map                  = 1000000;
  max_liwork                = std::numeric_limits<MKL_INT>::max();
  max_lwork                 = std::numeric_limits<MKL_INT>::max();
  inv_qe2f                  = 1;

  Uchem = Ushift     = 0.0;
  update_Q_totaltime = update_Q_time = 0.0;
  update_K_totaltime = update_K_time = 0.0;
  update_P_totaltime = update_P_time = 0.0;
  me                                 = comm->me;
  nprocs                             = comm->nprocs;
  pcg_shift = pcg_shift_scale = ppcg_invAAA_cut = 0;
  env_alpha_stepsize                            = 0;
  P_PPCG = q_iters  = nullptr;
  DEBUG_fxyz_store  = nullptr;
  DEBUG_eatom_store = nullptr;
  BBB = BBB_localGA = block_ave_value = nullptr;
  Ele_qsum = Ele_Uchem = Ele_qave = Ele_qsq = nullptr;
  B0_prev = B0_store = Q_GA_store = nullptr;
  extraGA_num = extraGA_disp = nullptr;
  all_ghostGA_ind = comm_recv_arr = nullptr;
  // pcg_localGA_firstneigh = pcg_localGA_firstneigh_localIndex = nullptr;
  // pcg_localGA_firstneigh_inva = nullptr;
  ppcg_GAindex     = nullptr;
  pcg_out          = nullptr;
  localGA_Vext     = nullptr;
  localGA_eleIndex = nullptr;
  Ele_matrix = Ele_D_matrix = nullptr;
  kspace_g_ewald            = -1;
  kspace_nn_pppm[0]         = -1;
  kspace_nn_pppm[1]         = -1;
  kspace_nn_pppm[2]         = -1;
  localGA_num = totalGA_num = 0;
  bisect_level              = 0;
  bisect_num                = 1;
  size_subAAA               = 100;
  B0_refresh_time = qsqsum_refresh_time = 100;
  electro                               = nullptr;

#ifndef CONP_NO_DEBUG
  B0_refresh_time     = 1;
  qsqsum_refresh_time = 1;
#endif
  scalapack_nb    = 256;
  scalar_flag     = 1;
  vector_flag     = 1;
  size_vector     = 10; // Number of thermo output vector
  DEBUG_LOG_LEVEL = 0;

  // GA_Vext.assign(n + 1, 0);
  // memory->create(GA_Vext, n + 1, "FixConpGA::GA_Flags")
  GA_Vext_str = nullptr;
  GA_Flags = GA_Vext_info = nullptr;
  GA_Vext_var             = nullptr;
  Ele_Control_Type        = nullptr;
  Ele_To_aType            = nullptr;
iseq:

  memory->create(GA_Flags, 2 * (n + 1), "FixConpGA::GA_Flags");
  GA_Vext_info = GA_Flags + (n + 1);
  memory->create(GA_Vext_type, (n + 1), "FixConpGA::GA_Vext_type");
  GA_Vext_ATOM_data = (double**)memory->smalloc(sizeof(double*) * (n + 1), "FixConpGA::GA_Vext_ATOM_data");
  GA_Vext_str       = (char**)memory->smalloc(sizeof(char*) * (n + 1), "FixConpGA::GA_Vext_str");
  memory->create(GA_Vext_var, (n + 1), "FixConpGA::GA_Vext_var");
  memory->create(Ele_Control_Type, n + 1, "FixConpGA::Elyt_Control_Type");
  memory->create(Ele_To_aType, n + 1, "FixConpGA::Ele_To_aType");
  for (size_t i = 0; i <= n; i++) {
    GA_Flags[i]          = 0; // 0 For non-control electrolyte atoms
    GA_Vext_str[i]       = nullptr;
    GA_Vext_info[i]      = -1;            // Index of variable
    GA_Vext_type[i]      = varTYPE::NONE; // the Default type is constant
    GA_Vext_var[i]       = std::numeric_limits<double>::quiet_NaN();
    Ele_Control_Type[i]  = CONTROL::NONE;
    Ele_To_aType[i]      = -1;
    GA_Vext_ATOM_data[i] = nullptr;
  }

  // self_ECpotential = Xi_self = 0.0;
  cut_coul = cut_coulsq = cut_coul_add = 0.0;
  // memory->create(self_GreenCoef, n+1, "fix::Conp:self_GreenCoef");

  AAA_save[0] = '\0';

  everynum = utils::inumeric(FLERR, arg[3], false, lmp);
  if (everynum < 0) error->all(FLERR, "Illegal fix conp command for fix every step");
  if (strcmp(arg[4], "inv") == 0) {
    minimizer = MIN::INV;
  } else if (strcmp(arg[4], "cg") == 0) {
    minimizer = MIN::CG;
    tol_style = TOL::REL_B;
  } else if (strcmp(arg[4], "pcg") == 0) {
    minimizer   = MIN::PCG;
    pcg_mat_sel = PCG_MAT::FULL;
    tol_style   = TOL::REL_B;
  } else if (strcmp(arg[4], "ppcg") == 0) {
    minimizer       = MIN::PCG_EX;
    pcg_mat_sel     = PCG_MAT::FULL;
    tol_style       = TOL::REL_B;
    ppcg_invAAA_cut = 1.0e-4;
    pcg_shift_scale = 1;
  } else if (strcmp(arg[4], "near") == 0) {
    minimizer = MIN::NEAR;
    tol_style = TOL::MAX_Q;
  } else if (strcmp(arg[4], "none") == 0) {
    minimizer = MIN::NONE;
    tol_style = TOL::REL_B;
  } else
    error->all(FLERR, "Illegal fix conp command for unknown minimization method");

  if (me == 0) utils::logmesg(lmp, "[ConpGA] update electrode charge every {} loop.\n", everynum);

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "tol") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for tol option.");
      tolerance = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (tolerance < 0)
        error->all(FLERR, "Illegal fix conp/GA command for tol option, "
                          "tolerance value only valid for positive value.");
      iarg += 2;
    } else if (strcmp(arg[iarg], "B0_refresh") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for B0_refresh option.");
      B0_refresh_time = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (B0_refresh_time <= 0)
        error->all(FLERR, "Illegal fix conp/GA command for tol option, "
                          "B0_refresh_time value only valid for positive value.");
      iarg += 2;

    } else if (strcmp(arg[iarg], "qsq_refresh") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for qsq_refresh option.");
      qsqsum_refresh_time = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (qsqsum_refresh_time <= 0)
        error->all(FLERR, "Illegal fix conp/GA command for tol option, "
                          "qsqsum_refresh_time value only valid for positive value.");
      iarg += 2;

    } else if (strcmp(arg[iarg], "tol_style") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for tol option.");
      if (strcmp(arg[iarg + 1], "res") == 0) {
        tol_style = TOL::RES;
        if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG using the criteria for minimal residual\n");
      } else if (strcmp(arg[iarg + 1], "pe") == 0) {
        tol_style = TOL::PE;
        if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG using the criteria for minimal delta eneergy\n");
      } else if (strcmp(arg[iarg + 1], "max_Q") == 0) {
        tol_style = TOL::MAX_Q;
        if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG using the criteria for minimal delta Q\n");
      } else if (strcmp(arg[iarg + 1], "std_Q") == 0) {
        tol_style = TOL::STD_Q;
        if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG using the criteria for minimal std Q\n");
      } else if (strcmp(arg[iarg + 1], "rel_Q") == 0) {
        tol_style = TOL::STD_RQ;
        if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG using the criteria for minimal std RQ\n");
      } else if (strcmp(arg[iarg + 1], "abs_Q") == 0) {
        tol_style = TOL::ABS_Q;
        if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG using the criteria for minimal ABS RQ\n");
      } else if (strcmp(arg[iarg + 1], "rel_B") == 0) {
        tol_style = TOL::REL_B;
        if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG using the criteria for minimal B0 * q\n");
      } else {
        error->all(FLERR, "Illegal fix conp/GA command for tol_style option, tolerance value "
                          "only valid for res, pe, max_Q, std_Q, rel_Q and abs_Q");
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "max") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for max option.");
      maxiter = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (maxiter < 0)
        error->all(FLERR, "Illegal fix conp/GA command for max option, maxiter "
                          "only valid for positive value.");
      iarg += 2;
    } else if (strcmp(arg[iarg], "metal") == 0 || strcmp(arg[iarg], "charged") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal fix conp/GA command for metal option.");
      i = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (i > n || i < 0) {
        sprintf(info,
                "Illegal fix conp/GA command for metal option, Illegal metal "
                "atom type for metal %.200s %.200s\n",
                arg[iarg + 1], arg[iarg + 2]);
        error->all(FLERR, info);
      }
      if (strstr(arg[iarg + 2], "v_")) {
        int n_str      = strlen(&arg[iarg + 2][2]) + 1;
        GA_Vext_str[i] = (char*)memory->smalloc(n_str * sizeof(char), "GA_Vext_str[i]"); // new char[n_str];
        strcpy(GA_Vext_str[i], &arg[iarg + 2][2]);
      } else {
        value          = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
        GA_Vext_var[i] = value;
      }
      Ele_num++;
      GA_Flags[i]               = Ele_num;
      Ele_To_aType[Ele_num - 1] = i;
      if (strcmp(arg[iarg], "metal") == 0) Ele_Control_Type[Ele_num] = CONTROL::METAL;
      if (strcmp(arg[iarg], "charged") == 0) Ele_Control_Type[Ele_num] = CONTROL::CHARGED;
      iarg += 3;
    } else if (strcmp(arg[iarg], "Smat") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for Smat option.");
      if (strcmp(arg[iarg + 1], "on") == 0 || strcmp(arg[iarg + 1], "1") == 0) {
        Smatrix_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0 || strcmp(arg[iarg + 1], "0") == 0) {
        Smatrix_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for Smat option, the "
                          "available options are on or off");
      if (minimizer != MIN::INV) Smatrix_flag = false;
      iarg += 2;
    } else if (strcmp(arg[iarg], "printf_me") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for printf_me option.");
      i = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (i < -1)
        error->all(FLERR, "Illegal fix conp/GA command for printf_me option, Illegal "
                          "proc index for print_me, only valid with proc index >= -1");
      work_me = i;
      iarg += 2;

    } else if (strcmp(arg[iarg], "scalapack") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for scalapack option.");
      scalapack_num = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
#ifndef SCALAPACK
      error->warning(FLERR, "SCALAPACK routine is not compiled!");
#endif
    } else if (strcmp(arg[iarg], "SCALAPACK_BSize") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for scalapack option.");
      scalapack_nb = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
#ifndef SCALAPACK
      error->warning(FLERR, "SCALAPACK routine is not compiled!");
#endif
    } else if (strcmp(arg[iarg], "Vext_update") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for Vext_update option.");
      if (strcmp(arg[iarg + 1], "on") == 0 || strcmp(arg[iarg + 1], "1") == 0) {
        update_Vext_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0 || strcmp(arg[iarg + 1], "0") == 0) {
        update_Vext_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for Vext_update option, "
                          "the available options are on or off");
      iarg += 2;
    } else if (strcmp(arg[iarg], "max_lwork") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for max_lwork option.");
      max_lwork = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "max_liwork") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for max_liwork option.");
      max_liwork = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "pcg_atoms") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for pcg_atoms option.");
      sparse_pcg_num = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "ppcg_cut") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for printf_me option.");
      ppcg_invAAA_cut = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (ppcg_invAAA_cut <= 0)
        error->all(FLERR, "Illegal fix conp/GA command for ppcg_cut, the value "
                          "should be >= 0.");
      iarg += 2;
    } else if (strcmp(arg[iarg], "pshift_scale") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for pshift_scale option.");
      pcg_shift_scale = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      // if (pcg_shift_scale < 0) error->all(FLERR,"Illegal fix conp/GA command
      // for ppcg_cut, the value should be >= 0.");
      iarg += 2;
    } else if (strcmp(arg[iarg], "PCG-DC_block_size") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for pcg-DC_block_size option.");
      size_subAAA = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "PCG-DC_map_size") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for pcg-DC_map_size option.");
      i = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      // sscanf(arg[iarg + 1], "%zu", &size_map);
      if (i != 1) {
        error->all(FLERR, "Illegal fix conp/GA command for pcg-DC_map_size "
                          "option, cannot read it as a size_t");
      }
      // size_map = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "check_first") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal fix conp/GA command for check_first option.");
      i         = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      check_tol = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      if (i < 0)
        error->all(FLERR, "Illegal fix conp/GA command for check_first option, Illegal proc "
                          "index for check_first, only valid with step index >= 0");
      valid_check_firstep = i;
      iarg += 3;
    } else if (strcmp(arg[iarg], "Xi_index") == 0) {
      Xi_self_index = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (Xi_self_index < 0)
        error->all(FLERR, "Illegal fix conp/GA command for xi_index option, "
                          "only valid for a positive integer.");
      iarg += 2;
    } else if (strcmp(arg[iarg], "check") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for check option");
      if (strcmp(arg[iarg + 1], "on") == 0 || strcmp(arg[iarg + 1], "1") == 0) {
        valid_check_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0 || strcmp(arg[iarg + 1], "0") == 0) {
        valid_check_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for check option, the "
                          "available options are on or off");
      iarg += 2;
    } else if (strcmp(arg[iarg], "DEBUG_GAtoGA") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for check option");
      if (strcmp(arg[iarg + 1], "on") == 0 || strcmp(arg[iarg + 1], "1") == 0) {
        DEBUG_GAtoGA = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0 || strcmp(arg[iarg + 1], "0") == 0) {
        DEBUG_GAtoGA = false;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for DEBUG_GAtoGA "
                          "option, the available options are on or off");
      iarg += 2;
    } else if (strcmp(arg[iarg], "DEBUG_SOLtoGA") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for check option");
      if (strcmp(arg[iarg + 1], "on") == 0 || strcmp(arg[iarg + 1], "1") == 0) {
        DEBUG_SOLtoGA = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0 || strcmp(arg[iarg + 1], "0") == 0) {
        DEBUG_SOLtoGA = false;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for DEBUG_SOLtoGA "
                          "option, the available options are on or off");
      iarg += 2;
    } else if (strcmp(arg[iarg], "pcg_analysis") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for pcg_analysis option.");
      pcg_out = (FILE*)this;
#ifndef CONP_NO_DEBUG
      if (me == 0) {
        pcg_out = fopen(arg[iarg + 1], "w");
        if (pcg_out == nullptr) {
          error->warning(FLERR, "cannot open the file for the pcg_analysis!\n");
        }
        utils::logmesg(lmp, "[ConpGA] save the convergent info into {}.", arg[iarg + 1]);
      }

#else
      error->warning(FLERR, "pcg_analysis is only implemented in the debug version!\n");
#endif
      iarg += 2;
    } else if (strcmp(arg[iarg], "debug") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for debug option.");
      i = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (i < -1)
        error->all(FLERR, "Illegal fix conp/GA command for debug option, Illegal proc "
                          "index for debug, only valid with proc index >= -1");
      DEBUG_LOG_LEVEL = i;
      iarg += 2;

    } else if (strcmp(arg[iarg], "force_analysis") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for force_analysis option.");
      if (strcmp(arg[iarg + 1], "on") == 0 || strcmp(arg[iarg + 1], "1") == 0) {
        force_analysis_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0 || strcmp(arg[iarg + 1], "0") == 0) {
        force_analysis_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for bvec option, the "
                          "available options are on/{1} or off/{0}");
      iarg += 2;

    } else if (strcmp(arg[iarg], "bvec") == 0) {

      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for bvec option.");
      if (me == 0) error->warning(FLERR, "bvec option has been deprecated!");
      /*
      if (strcmp(arg[iarg+1],"fast") == 0 || strcmp(arg[iarg+1],"1") == 0) {
        fast_bvec_flag = true;
      } else if (strcmp(arg[iarg+1],"normal") == 0 || strcmp(arg[iarg+1],"0") ==
      0) { fast_bvec_flag = false; } else error->all(FLERR,"Illegal fix conp/GA
      command for bvec option, the available options are fast/{1} or
      normal/{0}");
      */
      iarg += 2;
    } else if (strcmp(arg[iarg], "selfGG") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for selfGG option.");
      if (strcmp(arg[iarg + 1], "on") == 0 || strcmp(arg[iarg + 1], "1") == 0) {
        selfGG_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0 || strcmp(arg[iarg + 1], "0") == 0) {
        selfGG_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for selfGG option, the "
                          "available options are on or off");
      iarg += 2;
    } else if (strcmp(arg[iarg], "INFO") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for INFO option.");
      if (strcmp(arg[iarg + 1], "on") == 0 || strcmp(arg[iarg + 1], "1") == 0) {
        INFO_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0 || strcmp(arg[iarg + 1], "0") == 0) {
        INFO_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for selfGG option, the "
                          "available options are on or off");
      iarg += 2;
    } else if (strcmp(arg[iarg], "const_sol") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for bvec option.");
      error->warning(FLERR, "const_sol option to generate S matrix has been replaced as Smat!");
      /*
      if (strcmp(arg[iarg+1],"fast") == 0 || strcmp(arg[iarg+1],"1") == 0) {
        fast_bvec_flag = true;
      } else if (strcmp(arg[iarg+1],"normal") == 0 || strcmp(arg[iarg+1],"0") ==
      0) { fast_bvec_flag = false; } else error->all(FLERR,"Illegal fix conp/GA
      command for bvec option, the available options are fast/{1} or
      normal/{0}");
      */
      iarg += 2;
    } else if (strcmp(arg[iarg], "delta_cut") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal fix conp/GA command for coul_cut option.");
      else {
        cut_coul_add = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      }
      iarg += 2;
    }

    else if (strcmp(arg[iarg], "matrix") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal fix conp/GA command for matrix");
      strncpy(AAA_save, arg[iarg + 1], FIX_CONP_GA_WIDTH);
      save_format = strnstr_l(AAA_save,
                              FIX_CONP_GA_WIDTH); // strnstr(AAA_save, ".", FIX_CONP_GA_WIDTH);
      if (save_format == nullptr) error->all(FLERR, "Cannot find matrix_save format for fix conp/GA command");
      if (strcmp(save_format, ".binmat") == 0)
        AAA_save_format = 1;
      else if (strcmp(save_format, ".txtmat") == 0)
        AAA_save_format = 0;
      else
        error->all(FLERR, "Unknow matrix format, the valid formats are .binmat or .txtmat");

      if (strcmp(arg[iarg + 2], "save") == 0)
        AAA_save_format %= 10;
      else if (strcmp(arg[iarg + 2], "load") == 0) {
        AAA_save_format += 10;
        std::ifstream infile(AAA_save);
        if (!infile.good()) {
          sprintf(info, "Cannot find the target AAA matrix data file: %.300s", AAA_save);
          error->all(FLERR, info);
        }
      } else if (strcmp(arg[iarg + 2], "auto") == 0) {
        std::ifstream infile(AAA_save);
        if (!infile.good()) {
          AAA_save_format %= 10;
          if (me == 0) utils::logmesg(lmp, "[ConpGA] Save GA matrix to {}\n", AAA_save);
        } else {
          if (me == 0) utils::logmesg(lmp, "[ConpGA] load GA matrix from {}\n", AAA_save);
          AAA_save_format += 10;
        }
      } else if (strcmp(arg[iarg + 2], "none") == 0) {
        AAA_save_format = -1;
      } else
        error->all(FLERR, "Illegal fix conp command for AAAA matrix operatiton, only "
                          "valid option of save, load or auto!");
      iarg += 3;

    } else if (strcmp(arg[iarg], "neutrality") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for neutrality option!");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        neutrality_flag = true;
        qsum_set        = 0;
        iarg += 2;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        neutrality_flag = false;
        iarg += 2;
      } else if (strcmp(arg[iarg + 1], "init") == 0) {
        neutrality_flag = true;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for neutrality option, "
                          "the valid options are on or off!");

    } else if (strcmp(arg[iarg], "env_alpha") == 0) {
      // TODO change to accept one integer parameter with
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for env_alpha option!");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        __env_alpha_flag = true;
        if (iarg + 3 > narg) error->all(FLERR, "Illegal fix conp/GA command for env_alpha option with [on]!");
        env_alpha_stepsize = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
        if (env_alpha_stepsize < 1) error->all(FLERR, "Invalid stepsize to calculate the optimal alpha!\n");
        iarg += 3;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        __env_alpha_flag = false;
        iarg += 2;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for env_alpha option, "
                          "the valid options are on or off!");

    } else if (strcmp(arg[iarg], "ppcg_block") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for ppcg_block option!");
      bisect_num = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (bisect_num <= 0)
        error->all(FLERR, "Illegal fix conp/GA command for bisect_num option, "
                          "the valid value are positive!");
      bisect_num -= 1;
      bisect_level = 0;
      while (bisect_num > 0) {
        bisect_num >>= 1;
        bisect_level++;
      }
      bisect_num = 1 << bisect_level;
      if (me == 0) utils::logmesg(lmp, "[ConpGA] set the number of block in PPCG method as {} as 2^{{{}}}}\n", bisect_num, bisect_level);
      iarg += 2;

    } else if (strcmp(arg[iarg], "symmetry") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for symmetry option!");
#ifndef CONP_NO_DEBUG
      if (strcmp(arg[iarg + 1], "on") == 0) {
        symmetry_flag = SYM::MEAN;
        iarg += 2;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        symmetry_flag = SYM::NONE;
        iarg += 2;
      } else if (strcmp(arg[iarg + 1], "sqrt") == 0) {
        symmetry_flag = SYM::SQRT;
        iarg += 2;
      } else if (strcmp(arg[iarg + 1], "tran") == 0) {
        symmetry_flag = SYM::TRAN;
        iarg += 2;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for symmetry option, "
                          "the valid options are on or off!");
#else
      iarg += 2;
#endif
    } else if (strcmp(arg[iarg], "pcg_mat") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for pcg_matrix option!");
      if (strcmp(arg[iarg + 1], "full") == 0) {
        pcg_mat_sel = PCG_MAT::FULL;
        iarg += 2;
      } else if (strcmp(arg[iarg + 1], "normal") == 0) {
        pcg_mat_sel = PCG_MAT::NORMAL;
        iarg += 2;
      } else if (strcmp(arg[iarg + 1], "DC") == 0) {
        if (iarg + 4 > narg)
          error->all(FLERR, "Illegal fix conp/GA command for pcg_matrix-DC options, \
            the valid options are full, normal or DC!");
        pcg_mat_sel        = PCG_MAT::DC;
        gewald_tune_factor = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
        pseuo_tolerance    = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
        if (me == 0) utils::logmesg(lmp, "[ConpGA] DC_gewald = {:f}, tolerance {:e}\n", gewald_tune_factor, pseuo_tolerance);
        iarg += 4;
      } else
        error->all(FLERR, "Illegal fix conp command for pcg_matrix options, "
                          "only valid options of full, normal, DC");

    } else if (strcmp(arg[iarg], "units") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for units option.");
      if (strcmp(arg[iarg + 1], "lmp") == 0) {
        unit_flag = 0;
      } else if (strcmp(arg[iarg + 1], "eV") == 0) {
        unit_flag = 1;
      } else
        error->all(FLERR, "Illegal fix conp/GA command for units options,  the "
                          "valid options are lmp or eV");
      iarg += 2;
    } else if (strcmp(arg[iarg], "first") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for first option.");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        firstgroup_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        firstgroup_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp command for first options, the "
                          "valid options are on, off");
      // if (me == 0) utils::logmesg(lmp, "[ConpGA] First option has been
      // deprecated, by setting FALSE for all input!\n");
      iarg += 2;
      // Calulcate the GA <-> GA interaction via the internal
      // fix_pair_compute();
    } else if (strcmp(arg[iarg], "pair") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for pair option.");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        pair_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        pair_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp command for pair options, the "
                          "valid options are on and off.");
      iarg += 2;
    } else if (strcmp(arg[iarg], "Uchem_extract") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix conp/GA command for Uchem_show option.");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        Uchem_extract_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        Uchem_extract_flag = false;
      } else
        error->all(FLERR, "Illegal fix conp command for Uchem_extract options, "
                          "the valid options are on and off.");
      iarg += 2;
    } else {
      sprintf(info, "Illegal fix conp command for symbol %.200s\n", arg[iarg]);
      error->all(FLERR, info);
    }
  } // while

  if (firstgroup_flag) {
    // atom->sortfreq = 0;
    if (atom->firstgroupname != nullptr) {
      int firstgroup = group->find(atom->firstgroupname);
      if (firstgroup != igroup) {
        error->all(FLERR, "First group in ATOM class has already be set to be "
                          "another group of atoms");
      }
    }
    if (me == 0) utils::logmesg(lmp, "[ConpGA] Group [{}] atoms is sorted on the top of local atom list\n", arg[1]);
  }

  /*
  if (pair_flag == true and set_cut_coul != 0.0) {
    error->all(FLERR, "cut_coul for demostration is inconsistent with a **true**
  pair_flag");
  }
  */
  switch (minimizer) {
    case MIN::PCG:
      break;
    case MIN::CG:
      if (tol_style > TOL::RES) {
        /*
        tol_style = TOL::REL_B;
        if (me == 0)
          utils::logmesg(lmp, "[ConpGA] the stop criterion is not implemented in CG "
                          "method, using [res_B] instead!\n");
        */
        error->all(FLERR, "[ConpGA] the stop criterion is not implemented in CG "
                          "method, using [res_B] instead!\n");
        if (force_analysis_flag == true) {
          force_analysis_flag = false;
          error->warning(FLERR, "[ConpGA] the force analysis is only feasible "
                                "for the iterative method!\n");
        }
      }
      break;
    case MIN::PCG_EX:
      if (tol_style > TOL::RES) {
        /*
        tol_style = TOL::REL_B;
        if (me == 0)
          utils::logmesg(lmp, "[ConpGA] the stop criterion is not implemented in "
                          "PPCG method, using [res_B] instead!\n");
        */
        error->all(FLERR, "[ConpGA] the stop criterion is not implemented in CG "
                          "method, using [res_B] instead!\n");
      }
      if (bisect_level > 0) {
        memory->create(block_ave_value, bisect_num, "Fix:ConpGA for block_ave_value");
      }
      break;
    default:
      tol_style = TOL::NONE;
  }

  if (tol_style == TOL::RES || tol_style == TOL::STD_Q || tol_style == TOL::STD_RQ) {
    tolerance *= tolerance;
  }

  if (pair_flag == false and fast_bvec_flag == true) {
    // error->all(FLERR, "To use fast Bvec routine, the pairFlag should be turn
    // on! (current pair is off)");
  }

  list      = nullptr;
  AAA       = nullptr;
  AAA_sum1D = nullptr; // q_store = nullptr;
  localGA_EachProc.reserve(nprocs);
  localGA_EachProc_Dipl.reserve(nprocs);

  // default save AAA as .binmat
  if (strnlen(AAA_save, FIX_CONP_GA_WIDTH) == 0) {
    strncpy(AAA_save, "a.binmat", FIX_CONP_GA_WIDTH - 1);
    AAA_save_format = 1;
  }

  if (unit_flag == 1) {        // Using the eV unit
    double qe2f = force->qe2f; // For real units,  1.0 eV/A -> 23.0605419453 kcal/mol-A
    inv_qe2f    = 1 / qe2f;
  } else {
    inv_qe2f = 1; // Using the internal energy units
  }

  setup_CELL();
  /*
  if (minimizer == MIN::PCG && cut_coul_add != 0) {
    pcg_localGA_firstneigh            = new std::vector<int>;
    pcg_localGA_firstneigh_localIndex = new std::vector<int>;
    pcg_localGA_firstneigh_inva       = new std::vector<double>;
  } else {
    pcg_localGA_firstneigh            = &localGA_firstneigh;
    pcg_localGA_firstneigh_localIndex = &localGA_firstneigh_localIndex;
    pcg_localGA_firstneigh_inva       = &localGA_firstneigh_inva;
  }
  */
}

void FixConpGA::setup_CELL()
{
  FUNC_MACRO(0);
  setup_Vext();
  update_Vext();
  int n             = atom->ntypes;
  int n_plus        = n + 1;
  bool charged_flag = false;
  bool metal_flag   = false;
  for (int iEle = 1; iEle <= Ele_num; iEle++) {
    int iType = Ele_To_aType[iEle];
    if (Ele_Control_Type[iEle] == CONTROL::METAL) {
      if (me == 0)
        utils::logmesg(lmp,
                       "[ConpGA] Atom Type {} is set as {}th electrode with POTENTIAL "
                       "control at {:f}\n",
                       iType, iEle, GA_Vext_var[iType]);
      metal_flag = true;
    } else if (Ele_Control_Type[iEle] == CONTROL::CHARGED) {
      if (me == 0)
        utils::logmesg(lmp,
                       "[ConpGA] Atom Type {} as {}th electrode with CHARGE control "
                       "at {:f}\n",
                       iType, iEle, GA_Vext_var[iType]);
      charged_flag = true;
      if (GA_Vext_type[iType + n_plus] == varTYPE::ATOM) {
        char buf[256];
        sprintf("Cannot apply the Charged constraint of %s for Atom:[%d]\n", GA_Vext_str[iType], iType);
        error->all(FLERR, buf);
      }
    }
  }

  if (metal_flag == true && charged_flag == true) {
    error->all(FLERR, "Cannot applied the mixture model with both METAL an "
                      "CHARGED electrode control\n");
  } else {
    if (metal_flag == true) {
      Ele_Cell_Model = CELL::POT;
    }
    if (charged_flag == true) {
      Ele_Cell_Model = CELL::Q;
    }
  }

  postreat_flag = calc_EleQ_flag = false;
  switch (Ele_Cell_Model) {
    case CELL::POT:
      if (me == 0)
        utils::logmesg(lmp, "[ConpGA] The electrochemcial simulation cell in on "
                            "POT_CONTROL model.\n");
      if (neutrality_flag == true && Smatrix_flag == false) {
        postreat_flag = true;
      }
      if (neutrality_flag == false && Smatrix_flag == true) {
        error->all(FLERR, "The Smatrix option is only valid with the neutrality constraint!");
        Smatrix_flag = false;
      }
      if (neutrality_flag == false && Uchem_extract_flag == true) {
        error->warning(FLERR, "The extraction of Uchem contribution is only "
                              "valid with the neutrality constraint!");
        Uchem_extract_flag = false;
      }
      break;

    case CELL::Q:
      if (me == 0)
        utils::logmesg(lmp, "[ConpGA] The electrochemcial simulation cell in on "
                            "Q_CONTROL model.\n");
      postreat_flag = true;
      if (Uchem_extract_flag == true) {
        error->warning(FLERR, "The extraction of Uchem contribution is only "
                              "compatible with the CELL_Q_CONTROL");
        Uchem_extract_flag = false;
      }
      calc_EleQ_flag = true;
      /* code */
      break;
    default:
      error->all(FLERR, "Cannot find the feasible CELL solver model for current "
                        "electrode controls\n");
      break;
  }

  memory->create(Ele_qsum, Ele_num * 4, "fix_Conp/GA::setup_CELL()::Ele_qsum");

  Ele_Uchem = Ele_qsum + Ele_num;
  Ele_qave  = Ele_Uchem + Ele_num;
  Ele_qsq   = Ele_qave + Ele_num;
}

/* ----------------------------------------------------------------------
Check the first group option in atom_modify to be same with fix GA Conp
Ensure all electrode atom in such group
------------------------------------------------------------------------- */
void FixConpGA::firstgroup_check()
{
  /*
  char info[FIX_CONP_GA_WIDTH+1];
  // atom->firstgroup == -1 for atom Module without firstgroup set
  if(atom->firstgroup == -1 || groupbit != group->bitmask[atom->firstgroup])
    error->all(FLERR, "[ConpGA] Inconsistent first group in atom_modify and
  FixConpGA"); int nlocal = atom->nlocal; int *type  = atom->type, *mask =
  atom->mask, *tag = atom->tag; for (size_t i = 0; i < nlocal; i++)
  {
    if(GA_Flags[type[i]] && !(mask[i] & g)){
      sprintf(info, "[ConpGA] CPU[%d]: ID: %lu[%d], type = %d, is not included
  in electrode group\n", me, i, tag[i], type[i]); error->one(FLERR, info);
    }
  }
  if(me) utils::logmesg(lmp, "[ConpGA] electrode atom group is resorted as
  the top of atom list\n");
  */
}

/* ---------------------------------------------------------------------- */
/*
void FixConpGA::init_list(int id, NeighList *ptr)
{
  list = ptr;
}
*/

/* ---------------------------------------------------------------------- */

FixConpGA::~FixConpGA()
{
  FUNC_MACRO(0);
  if (__env_alpha_flag) calc_env_alpha();
  // memory->destroy(q_store);
  memory->destroy(AAA);
  memory->destroy(AAA_sum1D);
  memory->destroy(GA_Flags);
  memory->destroy(localGA_Vext);
  memory->destroy(extraGA_num);
  memory->destroy(extraGA_disp);

  if (minimizer == MIN::INV) {
    memory->destroy(BBB);
    memory->destroy(BBB_localGA);
  }
  if (minimizer == MIN::PCG_EX) {
    memory->destroy(ppcg_block);
    memory->destroy(ppcg_GAindex);
    memory->destroy(P_PPCG);
    /*
    for (int i = 0; i < localGA_num; ++i)
    {
      free(ppcg_GAindex[i]);
      free(P_PPCG[i]);
    }
    free(ppcg_GAindex);
    free(P_PPCG);
    */
  }
  memory->destroy(Q_GA_store);
  if (minimizer == MIN::PCG) {
#ifndef CONP_NO_DEBUG
    if (pcg_out != nullptr) {
      memory->destroy(q_iters);
      memory->destroy(pcg_res_norm);
      if (me == 0) fclose(pcg_out);
    }
#endif
    if (tol_style == TOL::REL_B) {
      memory->destroy(B0_prev);
      memory->destroy(B0_next);
      if (force_analysis_flag == true) memory->destroy(B0_store);
    }

    /*
    if (cut_coul_add != 0) {
      delete pcg_localGA_firstneigh;
      delete pcg_localGA_firstneigh_localIndex;
      delete pcg_localGA_firstneigh_inva;
    }
    */
  }
  memory->destroy(Ele_GAnum);
  memory->destroy(Ele_matrix);
  memory->destroy(Ele_D_matrix);
  memory->destroy(Ele_qsum);
  memory->destroy(localGA_eleIndex);

  memory->destroy(Ele_Control_Type);
  memory->destroy(Ele_To_aType);
  memory->destroy(GA_Vext_var);

  std::map<int, double*>::iterator it;
  for (it = VextID_to_ATOM_store.begin(); it != VextID_to_ATOM_store.end(); it++) {
    double* ptr = it->second;
    memory->destroy(ptr);
    it->second = nullptr;
  }

  int n = atom->ntypes;
  for (int i = 0; i <= n; i++) {
    if (GA_Vext_str[i]) memory->sfree(GA_Vext_str[i]);
  }
  memory->sfree(GA_Vext_str);
  memory->sfree(GA_Vext_ATOM_data);
  if (electro) delete electro;
  // memory->destroy(self_GreenCoef);
  if (firstgroup_flag) {
    memory->destroy(all_ghostGA_ind);
    // memory->destroy(all_ghostGA_seq);
  }
}

/* ---------------------------------------------------------------------- */

int FixConpGA::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  // mask |= THERMO_ENERGY;
  // mask |= POST_FORCE;
  // mask |= END_OF_STEP;
  mask |= PRE_REVERSE;
  return mask;
}

/* ---------------------------------------------------------------------- */
void FixConpGA::post_run()
{
  double inv_Ele_iter = 1.0 / Ele_iter;
  double double_qe2f  = 2 * force->qe2f;

  if (me == 0) {
    switch (Ele_Cell_Model) {
      case CELL::POT:
        if (update_Vext_flag) break;
        utils::logmesg(lmp, "\n---------- [Conp] Electrode charge fluctuation for {} MD invoking ---------------\n", Ele_iter);
        for (int i = 0; i < Ele_num; i++) {
          int iType    = Ele_To_aType[i];
          double q_ave = Ele_qave[i] * inv_Ele_iter;
          double q_std = sqrt(Ele_qsq[i] * inv_Ele_iter - q_ave * q_ave);
          double U     = (GA_Vext_var[iType] - Uchem) * double_qe2f; // GA  - Uchem
          utils::logmesg(lmp, "[Ele{:2d}] <Q> = {:.12f}; std{{Q}} = {:.12f} @ Pot {:.6f} V\n", i + 1, q_ave, q_std, U);
        }
        utils::logmesg(lmp, "\n");
        for (int i = 0; i < Ele_num; i++) {
          double q_ave = Ele_qave[i] * inv_Ele_iter / Ele_GAnum[i];
          double q_std = sqrt(Ele_qsq[i] * inv_Ele_iter - q_ave * q_ave) / Ele_GAnum[i];
          double U     = (localGA_Vext[i] - Uchem) * double_qe2f;
          utils::logmesg(lmp, "[Ele{:2d}] <q_i> = {:.12f}; std{{q_i}} = {:.12f}; Ele. Atom = {}\n", i + 1, q_ave, q_std, Ele_GAnum[i]);
        }
        utils::logmesg(lmp, "-------------End [Conp] Electrode charge fluctuation -------------\n\n");
        break;
      case CELL::Q:
        if (update_Vext_flag) break;
        utils::logmesg(lmp, "\n---------- [Conp] Electrode Potential fluctuation for {} invoking ---------------\n", Ele_iter);
        for (int i = 0; i < Ele_num; i++) {
          int iType    = Ele_To_aType[i];
          double U_ave = Ele_qave[i] * inv_Ele_iter;
          double U_std = sqrt(Ele_qsq[i] * inv_Ele_iter - U_ave * U_ave) * double_qe2f;
          double Q     = GA_Vext_var[iType];
          U_ave *= double_qe2f;
          utils::logmesg(lmp, "[Ele{:2d}] <U> = {:.12f} V; std{{U}} = {:.12f} V @ Charge {:.6f} e\n", i + 1, U_ave, U_std, Q);
        }
        utils::logmesg(lmp, "-------------End [Conp] Electrode Potential fluctuation -------------\n\n");
        break;
      default:;
    }
  }
}

void FixConpGA::energy_calc(int eflag, int vflag, int comm_type = 0)
{
  force->pair->compute(eflag, vflag);
  electro->compute(vflag, vflag);
  if (newton_pair) {
    if (comm_type == 0) {
      fix_reverse_comm(REV_COUL_RM);
    } else if (comm_type == 1) {
      fix_reverse_comm(REV_COUL_RM);
      comm->reverse_comm();
    }
  }
}

void FixConpGA::setup_Vext()
{
  FUNC_MACRO(0);
  char buff[256];

  int n = atom->ntypes;
  // int num_ATOM = 0;
  for (int i = 0; i <= n; i++) {
    if (GA_Vext_str[i]) {
      if (me == 0 && DEBUG_LOG_LEVEL > 0) utils::logmesg(lmp, "[Debug] GA_Vext_str for {}[{}]\n", GA_Vext_str[i], i);
      GA_Vext_info[i] = input->variable->find(GA_Vext_str[i]);
      if (GA_Vext_info[i] < 0) {
        sprintf(buff, "Variable [%s] for fix move does not exist", GA_Vext_str[i]);
        error->all(FLERR, buff);
      }
      if (input->variable->equalstyle(GA_Vext_info[i])) {
        GA_Vext_type[i] = varTYPE::EQUAL;
      } else if (input->variable->atomstyle(GA_Vext_info[i])) {
        if (me == 0 && DEBUG_LOG_LEVEL > 0) utils::logmesg(lmp, "[Debug] Vext: {} for atomType:{} is a ATOM style\n", GA_Vext_str[i], i);
        GA_Vext_type[i]                       = varTYPE::ATOM;
        VextID_to_ATOM_store[GA_Vext_info[i]] = nullptr;
      } else {
        sprintf(buff, "Variable [%s] is not EQUAL or Atom style", GA_Vext_str[i]);
        error->all(FLERR, buff);
      }
    }
  } // i;
}

void FixConpGA::update_Vext()
{
  // printf("open update_Vext\n");
  FUNC_MACRO(0);
  // setup_Vext() TODO if error
  int n                = atom->ntypes;
  int nlocal           = atom->nlocal;
  int* type            = atom->type;
  double half_inv_qe2f = 0.5 * inv_qe2f;

  std::map<int, double*>::iterator it;
  for (it = VextID_to_ATOM_store.begin(); it != VextID_to_ATOM_store.end(); it++) {
    double* ptr = it->second;
    int ID      = it->first;
    memory->grow(ptr, nlocal, "Fix_Conp/GA::update_Vext()::VextID_to_ATOM_store");
    input->variable->compute_atom(ID, igroup, ptr, 1, 0);
    it->second = ptr;
    // printf("it->first = %d, %x %x\n", it->first, ptr,
    // VextID_to_ATOM_store[ID]);
  }

  for (int iEle = 0; iEle < Ele_num; iEle++) {
    int iType = Ele_To_aType[iEle];
    // printf("compute %f, iType = %d\n", iEle, iType);
    if (GA_Vext_type[iType] == varTYPE::EQUAL) {
      double value = input->variable->compute_equal(GA_Vext_info[iType]);
      if (me == 0 && DEBUG_LOG_LEVEL > -1) utils::logmesg(lmp, "[Debug] Equal Set atomType:{} as [{:f}]\n", iType, value);
      // GA_Vext_var[iType] = half_inv_qe2f * value + Uchem;
      GA_Vext_var[iType] = value;
    } else if (GA_Vext_type[iType] == varTYPE::ATOM) {
      // printf("iType = %d, %x\n", iType,
      // VextID_to_ATOM_store[GA_Vext_info[iType]]);
      GA_Vext_ATOM_data[iType] = VextID_to_ATOM_store[GA_Vext_info[iType]];
      // printf("iType = %d, %x\n", iType, GA_Vext_ATOM_data[iType]);
      // printf("compute %f, iType = %d\n", iEle, iType);
      // memset(GA_Vext_ATOM_Store[iType],0,localGA_num * sizeof(double));
      // printf("GA_Vext_ATOM_Store[%d] = %x\n", iType,
      // GA_Vext_ATOM_Store[iType]); printf("end of compute\n");
      /*
      for (size_t i = 0; i < extGA_num; i++)
      {
        printf("extGA_num = %d: %.15g\n", i, GA_Vext_ATOM_Store[iType][i]);
      }
      */
    } else {
      if (me == 0 && DEBUG_LOG_LEVEL > 0) utils::logmesg(lmp, "[Debug] CONST Set atomType:{} as [{:f}]\n", iType, GA_Vext_var[iType]);
    }
    // utils::logmesg(lmp, "[Debug] Type[%d] Set atomType:%d as [%f]\n",
    // GA_Vext_str_Type[iType], iType, GA_Vext_var[iType]);
  }

  const int* const GA_seq_arr = &localGA_seq.front();
  for (int iGA = 0; iGA < localGA_num; iGA++) {
    int iseq  = GA_seq_arr[iGA];
    int iType = type[iseq];
    int iEle  = GA_Flags[iType];
    if (Ele_Control_Type[iEle] == CONTROL::METAL) {
      if (GA_Vext_type[iType] != varTYPE::ATOM) {
        localGA_Vext[iGA] = half_inv_qe2f * GA_Vext_var[iType] + Uchem;
        // printf("iGA2 = %d\n", iGA);
        // if (iGA == 0) printf("%d, localGA_Vext[iGA] = %f\n", iGA,
        // localGA_Vext[iGA]);
      } else {
        // printf("iseq = %d\n", iseq);
        localGA_Vext[iGA] = half_inv_qe2f * GA_Vext_ATOM_data[iType][iseq] + Uchem;
        // if (GA_Vext_ATOM_Store[iType][iGA] != 0.5) printf("wrong = %f!\n",
        // GA_Vext_ATOM_Store[iType][iGA]); printf("iGA = %d\n", iGA);
      }
    }
  }
  // printf("end\n");
  // printf("result = %.12g\n", GA_Vext_var[3]+GA_Vext_var[4]);
}

/* ---------------------------------------------------------------------- */
// analysis the force in the iterative method
void FixConpGA::force_analysis(int eflag, int vflag)
{
  FUNC_MACRO(0);
  if (eflag / 2 != 1) eflag += 2;

  int nlocal = atom->nlocal;
  int nlocal_SOL, ntotal_SOL;
  double** f = atom->f;
  f_store_vec.reserve(3 * nlocal);
  double* const f_store  = &f_store_vec.front();
  double* q              = atom->q;
  double tolerance_store = tolerance;
  double fres_local, fsqr_local, fmax_local = -1, v_sqr, v_res, v_max;
  double fres_total, fsqr_total, fmax_total;

  double fx, fy, fz;
  TOL tol_style_store = tol_style;
  int curriter_store  = curriter, i, i3;
  int* type           = atom->type;
  int isel            = 1;

  /* store the results in the previous step */
  q_store.reserve(localGA_num);
  double* const q_store_arr = Q_GA_store;
  double Ushift_store       = Ushift;
  // printf("Ushift = %.16f, %.16f\n", Ushift, Uchem);
  if (tol_style == TOL::REL_B) {
    for (i = 0; i < localGA_num; ++i) {
      q_store_arr[i] = q[localGA_seq[i]];
      B0_store[i]    = B0_prev[i];
    }
  } else {
    for (i = 0; i < localGA_num; ++i) {
      q_store_arr[i] = q[localGA_seq[i]];
    }
  }
  force_clear();
  energy_calc(eflag, 0, 1);
  std::copy(f[0], f[0] + nlocal * 3, f_store);

  tolerance = 1e-17;
  tol_style = TOL::MAX_Q;
  switch (minimizer) {
    case MIN::CG:
      update_charge_cg(false);
      break;
    case MIN::PCG:
      update_charge_pcg();
      break;
    case MIN::PCG_EX:
      update_charge_ppcg();
      break;
    default:
      error->all(FLERR, "force_analysis() only valid for MIN::CG, MIN::PCG or MIN::PPCG methods");
  }

  curriter  = curriter_store;
  tolerance = tolerance_store;
  tol_style = tol_style_store;

  force_clear();
  energy_calc(eflag, vflag, 1);

  nlocal_SOL = 0;
  for (i = 0, i3 = 0; i < nlocal; ++i, i3 += 3) {
    if (GA_Flags[type[i]] == 0) {
      fx    = f[i][0];
      fy    = f[i][1];
      fz    = f[i][2];
      v_sqr = fx * fx + fy * fy + fz * fz;
      fsqr_local += v_sqr;

      fx -= f_store[i3 + 0];
      fy -= f_store[i3 + 1];
      fz -= f_store[i3 + 2];
      v_res = fx * fx + fy * fy + fz * fz;
      fres_local += v_res;

      v_max = v_res / v_sqr;
      if (v_max > fmax_local) fmax_local = v_max;
      nlocal_SOL++;
    }
  }

  MPI_Reduce(&fres_local, &fres_total, 1, MPI_DOUBLE, MPI_SUM, 0, world);
  MPI_Reduce(&fsqr_local, &fsqr_total, 1, MPI_DOUBLE, MPI_SUM, 0, world);
  MPI_Reduce(&fmax_local, &fmax_total, 1, MPI_DOUBLE, MPI_MAX, 0, world);
  if (me == 0) {
    force_analysis_res = sqrt(fres_total / fsqr_total);
    force_analysis_max = sqrt(fmax_total);
  }

  // restore back to previous one;
  // printf("Ushift = %.16f\n", Ushift);

  // printf("Ushift = %.16f, %.16f\n", Ushift, Uchem);

  Ushift = Ushift_store;
  if (tol_style == TOL::REL_B) {
    for (i = 0; i < localGA_num; ++i) {
      q[localGA_seq[i]] = q_store_arr[i];
      B0_prev[i]        = B0_store[i];
    }
  } else {
    for (i = 0; i < localGA_num; ++i) {
      q[localGA_seq[i]] = q_store_arr[i];
    }
  }
  std::copy(f_store, f_store + nlocal * 3, f[0]);
  if (newton_pair) {
    fix_forward_comm(0);
    // comm->reverse_comm();
  }
  electro->qsum_qsq();
}

void FixConpGA::calc_Uchem_with_zero_Q_sol()
{
  int iproc, n = atom->ntypes;
  int i, iseq, *type = atom->type;

  double* kspace_eatom = force->kspace->eatom;
  double* cl_atom      = electro->cl_atom();
  double* q            = atom->q;
  double ecoul;
  const int* const GA_seq_arr = &localGA_seq.front();

  for (iproc = 0; iproc < nprocs; iproc++) {
    if (localGA_EachProc_Dipl[iproc] <= Xi_self_index && Xi_self_index < localGA_EachProc_Dipl[iproc] + localGA_EachProc[iproc]) break;
  }
  if (me == iproc) {
    iseq   = GA_seq_arr[Xi_self_index - localGA_EachProc_Dipl[iproc]];
    ecoul  = cl_atom[iseq] + kspace_eatom[iseq];
    Ushift = ecoul / q[iseq] - localGA_Vext[iseq];
  }
  MPI_Bcast(&Ushift, 1, MPI_DOUBLE, iproc, world);
  Uchem += Ushift;

  // printf("Uchem = %.12f, Ushfit = %f, ecoul = %.12f[%.10f,%.10f], q = %.12f\n", Uchem, Ushift, ecoul, cl_atom[iseq], kspace_eatom[iseq], q[iseq]);
  for (i = 0; i < localGA_num; i++) {
    localGA_Vext[i] += Ushift;
  }
}

/* ---------------------------------------------------------------------- */
// void FixConpGA::post_force(int vflag)
// void FixConpGA::end_of_step()

void FixConpGA::pre_reverse(int eflag, int vflag)
{
  if (update->ntimestep % everynum != 0) {
    return;
  };

  FUNC_MACRO(0)
  //

  double ecoul, econtrol, ecoul_error;
  double* q       = atom->q;
  double* eatom   = force->pair->eatom;
  double* cl_atom = electro->cl_atom();
  double** f      = atom->f;
  int i, nlocal = atom->nlocal;
  bool pair_saved = electro->pair->sv_SOL_saved && electro->pair->sv_SOL_timestep == update->ntimestep;
  electro->pre_reverse();

  update_K_totaltime += update_K_time;
  update_P_totaltime += update_P_time;
  // printf("post_force at %lld\n", update->ntimestep);
  // force->kspace->compute(999, 0);

  /* check validation of electric potential on each electrode atoms */
  bigint run_steps = update->ntimestep - update->firststep;
  if (valid_check_flag == true) {
    bigint next_run = update->ntimestep + 1;
    if (next_run % everynum == 0) electro->compute_addstep(next_run);
    if (run_steps >= valid_check_firstep) {
      const int* const GA_seq_arr = &localGA_seq.front();
      double* kspace_eatom        = force->kspace->eatom;

      if (me == 0 and DEBUG_LOG_LEVEL > 0) utils::logmesg(lmp, "[ConpGA] check the validation of electrode potential.\n");

      // energy_clear();
      int nlocal = atom->nlocal;
      if (pair_saved) {
        /// force_clear();
        // fix_reverse_comm(REV_COUL_RM);

        if (!electro->pair->sv_GA_flag) {
          electro->pair->compute_Group(3, 3, groupbit, false);
        }
        if (newton_pair) {
          fix_reverse_comm(REV_COUL_RM);
          // if (pair_saved) comm->reverse_comm();
        }
        // printf("eatom[0] = %.12f, cl_atom[0] = %.12f\n", eatom[0], cl_atom[0]);
        // ;
        // if (newton_pair) comm->reverse_comm();
        memory->grow(DEBUG_eatom_store, nlocal * 2, "DEBUG_nlocal_store");
        memory->grow(DEBUG_fxyz_store, nlocal, 3, "DEBUG_nlocal_store");
        std::copy(eatom, eatom + nlocal, DEBUG_eatom_store);
        std::copy(cl_atom, cl_atom + nlocal, DEBUG_eatom_store + nlocal);
        std::copy(f[0], f[0] + 3 * nlocal, DEBUG_fxyz_store[0]);
        /*
        force_clear();
        electro->pair_GAtoGA();
        electro->calc_GA_potential(0, 0, 0);
        // fix_reverse_comm(REV_COUL_RM);
        for (int i = 0; i < nlocal; i++) {
          // if (i < 10) printf("cl_atom[%d] = %.10f\n", i, cl_atom[i]);
          DEBUG_eatom_store[i] += cl_atom[i];
          DEBUG_eatom_store[i + nlocal] += cl_atom[i];
        }
        */
      }

      // reset for the next one;

      if (pair_saved || eflag / 2 != 1) {
        if (me == 0 && DEBUG_LOG_LEVEL > 0)
          utils::logmesg(lmp,
                         "[ConpGA] Compute the per-atom energy! with eflag = {}, vflag "
                         "= {}\n",
                         eflag, vflag);
        // update the per-atom energy
        force_clear();
        force->pair->compute(eflag + 2, vflag);
        electro->compute(eflag + 2, vflag);
      }

      if (newton_pair) {
        fix_reverse_comm(REV_COUL_RM);
        // if (pair_saved) comm->reverse_comm();
      }
      // Get the contribution of energy from the ghost atom

      int i, iseq, *type = atom->type;

      if (Smatrix_flag == true) {
        calc_Uchem_with_zero_Q_sol();
      }

      for (i = 0; i < localGA_num; ++i) {
        iseq     = GA_seq_arr[i];
        ecoul    = cl_atom[iseq] + kspace_eatom[iseq];
        econtrol = q[iseq] * localGA_Vext[i];
        if (!isclose(econtrol, ecoul, check_tol)) {
          // ecoul_error > check_tol) {
          // #ifndef CONP_NO_DEBUG
          ecoul_error = fabs((ecoul - econtrol) / econtrol);
          if (DEBUG_LOG_LEVEL > -1) {
            utils::logmesg(lmp,
                           "[Proc. #{}] iseq: {}[tag:{}-type:{}]; ecoul: {:.20g}[cl:{:g}/ea:{:g}:ek:{:g}]; "
                           "econtrol: {:.20g} [Q:{:.20g} x V:{:.10g}]; error: {:g}; Uchem: {:g}\n",
                           me, iseq, localGA_tag[i], type[iseq], ecoul, cl_atom[iseq], eatom[iseq], kspace_eatom[iseq], econtrol, q[iseq], localGA_Vext[i],
                           ecoul_error, Uchem);
          }
          // #endif
          error->one(FLERR, "Invalid electrostatic potential by Fix Conp/GA!\n");
        }
      }
      if (pair_saved) {
        if (run_steps == 0 && me == 0) utils::logmesg(lmp, "[first-step] cannot be set to be zero with a saved pair calculation\n");
        for (i = 0; i < nlocal; ++i) {
          double eatom_ctrl = eatom[i];
          double echeck_tol = 10 * check_tol;
          if (!isclose(eatom_ctrl, DEBUG_eatom_store[i], echeck_tol)) {
            double epair_error = fabs((eatom_ctrl - DEBUG_eatom_store[i]) / eatom_ctrl);
            // if (epair_error > check_tol) {
            utils::logmesg(lmp, "[Proc. #%d] eatom[%d] %.12g vs %.12g; cl_atom: %.10f vs %.10f, error: %.10g\n", me, i, eatom_ctrl, DEBUG_eatom_store[i],
                           cl_atom[i], DEBUG_eatom_store[i + nlocal], epair_error);
            error->one(FLERR, "Invalid pair energy by Fix Conp/GA!\n");
          }

          double fx = f[i][0], fy = f[i][1], fz = f[i][2];
          double fcheck_tol = 100 * check_tol;
          if (!isclose(fx, DEBUG_fxyz_store[i][0], fcheck_tol)) {
            // if (fx_e > check_tol) {
            double fx_e = fabs((fx - DEBUG_fxyz_store[i][0]) / fx);
            utils::logmesg(lmp, "[Proc. #{}] fx[{}] {:.12g} vs {:.12g}; error: {:.10g}\n", me, i, fx, DEBUG_fxyz_store[i][0], fx_e);
            error->one(FLERR, "Invalid pair energy by Fix Conp/GA!\n");
          }
          if (!isclose(fy, DEBUG_fxyz_store[i][1], fcheck_tol)) {
            double fy_e = fabs((fy - DEBUG_fxyz_store[i][1]) / fy);
            utils::logmesg(lmp, "[Proc. #{}] fx[{}] {:.12g} vs {:.12g}; error: {:.10g}\n", me, i, fx, DEBUG_fxyz_store[i][1], fy_e);
            error->one(FLERR, "Invalid pair energy by Fix Conp/GA!\n");
          }
          if (!isclose(fz, DEBUG_fxyz_store[i][2], fcheck_tol)) {
            double fz_e = fabs((fz - DEBUG_fxyz_store[i][2]) / fz);
            utils::logmesg(lmp, "[Proc. #{}] fx[{}] {:.12g} vs {:.12g}; error: {:.10g}\n", me, i, fx, DEBUG_fxyz_store[i][2], fz_e);
            error->one(FLERR, "Invalid pair energy by Fix Conp/GA!\n");
          }
        }
      }
    }
    // ----------------  valid_check_flag == true; ---------------------
  } else if (Smatrix_flag == true) {
    if (newton_pair) {
      fix_reverse_comm(REV_COUL_RM);
    }
    calc_Uchem_with_zero_Q_sol();
  }

  if (Uchem_extract_flag) {
    for (i = 0; i < nlocal; ++i) {
      econtrol = Uchem * q[i];
      cl_atom[i] -= econtrol;
      eatom[i] -= econtrol;
    }
  }

  electro->pre_reverse();
}

/* ---------------------------------------------------------------------- */

void FixConpGA::init() { MPI_Comm_rank(world, &me); }

void FixConpGA::Print_INFO()
{
  if (!(me == 0)) {
    return;
  }
  utils::logmesg(lmp, "<--------------------------- Fix Conp/GA INFO "
                      "----------------------------->\n");
  utils::logmesg(lmp, "\n\n\n\n");
  utils::logmesg(lmp, "<-----------------------------------------------------------"
                      "--------------->\n");
}

void FixConpGA::localGA_build()
{
  FUNC_MACRO(0);
  int i, iGA, iextGA, iseq, nlocal = atom->nlocal, nghost = atom->nghost, ntotal;
  int* type = atom->type;
  int* tag  = atom->tag;
  char info[FIX_CONP_GA_WIDTH + 1];
  double *q = atom->q, qlocal_SOL;
  // std::list<int> localGA_tag_list, localGA_seq_list;

  ntotal      = nlocal + nghost;
  localGA_num = 0;
  qlocal_SOL  = 0.0;
  for (i = 0; i < nlocal; ++i) {
    if (GA_Flags[type[i]] != 0) {
      localGA_num++;
      if (q[i] == 0) q[i] = 1e-15;
    } else {
      qlocal_SOL += q[i];
    }
  }
  MPI_Allreduce(&qlocal_SOL, &qtotal_SOL, 1, MPI_DOUBLE, MPI_SUM, world);
  if (me == 0) utils::logmesg(lmp, "[ConpGA] The total charge in solution is {:.16f}\n", qtotal_SOL);
  if (fabs(qtotal_SOL) > 1e-11 && Smatrix_flag == true) {
    error->all(FLERR, "The Smatrix approach is not compatiable with the "
                      "monopolar electrolyte solution (Q^{SOL} \neq 0)");
  }
  // printf("neutrality_flag = %d, Ele_Cell_Model = %d\n",
  //  neutrality_flag, Ele_Cell_Model);
  if (neutrality_flag && Ele_Cell_Model == CELL::Q) {
    double Q_on_electrode = 0;
    for (int i = 0; i < Ele_num; i++) {
      int iType = Ele_To_aType[i];
      Q_on_electrode += GA_Vext_var[iType];
    }
    if (fabs(Q_on_electrode + qtotal_SOL) > 1e-11) {
      if (me == 0) utils::logmesg(lmp, "Charge on GA: {:g}; Charge on SOL: {:g}; Total Q: {:.12g}\n", Q_on_electrode, qtotal_SOL, Q_on_electrode + qtotal_SOL);
      error->all(FLERR, "In consistent charge constraint on the electrode with "
                        "the neutrality requirement.");
    }
  }

  map<int, int> tag_to_num_of_ghost;
  size_t reserve_num = ntotal - nlocal + localGA_num;
  // IF crash check this point
  MAP_RESERVE(tag_to_num_of_ghost, reserve_num);

  ghostGA_num      = 0;
  used_ghostGA_num = 0;
  all_ghostGA_num  = 0;
  for (i = nlocal; i < ntotal; ++i) {
    if (GA_Flags[type[i]] != 0) {
      all_ghostGA_num++;
      int itag = tag[i];
      if (tag_to_num_of_ghost.count(itag) == 0) {
        tag_to_num_of_ghost[itag] = 0;
        if (tag_to_num_of_ghost.size() == reserve_num) {
          reserve_num *= 2;
          MAP_RESERVE(tag_to_num_of_ghost, reserve_num);
        }
      }
      iseq = atom->map(tag[i]);
      used_ghostGA_num++;
      if (iseq >= nlocal) {
        tag_to_num_of_ghost[tag[i]]--; // < 0 if it is a ghostGA belong to other
                                       // core
        ghostGA_num++;
      } else {
        tag_to_num_of_ghost[tag[i]]++; // > 0 if it is a ghostGA belong to me
                                       // core
      }
    }
  }

  if (firstgroup_flag) {
    atom->nfirst = localGA_num;
    // memory->create(all_ghostGA_seq, all_ghostGA_num, "Fix_Conp/GA::localGA_build()::all_ghostGA_seq");
    memory->create(all_ghostGA_ind, all_ghostGA_num, "Fix_Conp/GA::localGA_build()::all_ghostGA_ind");
    for (int iGA = 0, i = nlocal; i < ntotal; ++i) {
      if (GA_Flags[type[i]] != 0) {
        int itag = tag[i];
        // all_ghostGA_seq[iGA] = i;
        all_ghostGA_ind[iGA] = i;
        iGA++;
      }
    }
  }

  // used_ghostGA_num > ghostGA_num
  // extGA_num > usedGA_num
  extGA_num        = localGA_num + used_ghostGA_num;
  usedGA_num       = localGA_num + ghostGA_num;
  dupl_ghostGA_num = used_ghostGA_num - ghostGA_num;

  true_usedGA_num = tag_to_num_of_ghost.size();

  std::vector<std::pair<int, int>> tag_and_size_vec;
  tag_and_size_vec.reserve(true_usedGA_num);
  for (map<int, int>::iterator it = tag_to_num_of_ghost.begin(); it != tag_to_num_of_ghost.end(); it++) {
    tag_and_size_vec.push_back(std::pair<int, int>(it->second, it->first));
  }
  std::sort(tag_and_size_vec.begin(), tag_and_size_vec.end());

  MAP_RESERVE(mul_tag_To_extraGAIndex, true_usedGA_num)
  // memory->create(extraGA_tag,  true_usedGA_num + 1, "FixConpGA:extraGA_num");

  memory->grow(extraGA_num, true_usedGA_num + 1, "FixConpGA:extraGA_num");
  memory->grow(extraGA_disp, true_usedGA_num + 1, "FixConpGA:extraGA_disp");
  memory->grow(localGA_Vext, localGA_num, "FixConpGA:localGA_Vext");
  for (i = 0; i < true_usedGA_num; ++i) {
    // extraGA_tag[i] = tag_and_size_vec[i].second;
    extraGA_num[i] = abs(tag_and_size_vec[i].first);

    mul_tag_To_extraGAIndex[tag_and_size_vec[i].second] = i;

    if (i == 0) {
      extraGA_disp[i] = 0;
    } else {
      extraGA_disp[i]    = extraGA_num[i - 1] + extraGA_disp[i - 1];
      extraGA_num[i - 1] = 0;
      // printf("[%d] total extraGA = %d\n", me, extraGA_disp[i]);
    }
    // if (me == 0) printf("tag = %d; num = %d\n", tag_and_size_vec[i].second,
    // tag_and_size_vec[i].first);
  }
  // printf("[%d] total extraGA = %d\n", me, extraGA_num[true_usedGA_num - 1] +
  // extraGA_disp[true_usedGA_num - 1]);

  extraGA_num[true_usedGA_num - 1] = 0;

  tag_to_num_of_ghost.empty();
  tag_and_size_vec.empty();

  MPI_Barrier(world);
  for (int iproc = 0; iproc < nprocs; ++iproc) {
    if (me == iproc)
      utils::logmesg(lmp,
                     "[ConpGA] CPU[{}]:local: {}, nb_ghost: {}, self_ghost: {}, "
                     "used:{};\n",
                     me, localGA_num, ghostGA_num, dupl_ghostGA_num, usedGA_num);
  }
  MPI_Barrier(world);
  /* ------------------------------------------------------
   Structure of localGA
   |<-localGA_num->|<-ghostGA_num->|<-(usedGA-ghostGA)->|
   localGA_tag save all seq
   localGA_tag |tag|tag|seq|
  --------------------------------------------------------- */
  localGA_tag.reserve(extGA_num + 1);
  localGA_seq.reserve(extGA_num + 1);

  int* const GA_tag    = &localGA_tag.front();
  int* const GA_seq    = &localGA_seq.front();
  int* const extGA_seq = GA_seq + localGA_num;
  int* const extGA_tag = GA_tag + localGA_num;

  iGA = iextGA = 0;

  memory->create(Ele_GAnum, Ele_num, "fix_Conp/GA::setup_CELL()::Ele_qsum");
  set_zeros(Ele_GAnum, 0, Ele_num);
  memory->create(localGA_eleIndex, localGA_num, "fix_Conp/GA::setup_CELL()::localGA_types");
  for (i = 0; i < nlocal; ++i) {
    int iEle = GA_Flags[type[i]];
    if (iEle != 0) {
      GA_seq[iGA]           = i;
      GA_tag[iGA]           = tag[i];
      localGA_eleIndex[iGA] = iEle - 1;
      localGA_Vext[iGA]     = iEle * 0.05;
      Ele_GAnum[iEle - 1]++;
      iGA++;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, Ele_GAnum, Ele_num, MPI_INT, MPI_SUM, world);

  const map<int, int>& tag_To_extraGAIndex = mul_tag_To_extraGAIndex;
  for (iseq = nlocal; iseq < ntotal; ++iseq) {
    if (GA_Flags[type[iseq]] != 0) {
#ifdef CONP_NO_DEBUG
      iGA = mul_tag_To_extraGAIndex[tag[iseq]];
#else
      iGA = tag_To_extraGAIndex.at(tag[iseq]);
#endif
      extGA_seq[extraGA_num[iGA] + extraGA_disp[iGA]] = iseq;
      extGA_tag[extraGA_num[iGA] + extraGA_disp[iGA]] = tag[iseq];
      extraGA_num[iGA] += 1;
      if (extraGA_disp[iGA] >= ghostGA_num) {
        iextGA++;
      }
    }
  }

  if (iextGA != dupl_ghostGA_num) {
    snprintf(info, FIX_CONP_GA_WIDTH, "Wrongggggggg iextGA %d neq dupl_ghostGA_num %d", iextGA, dupl_ghostGA_num);
    error->one(FLERR, info);
  }
  /*
  std::copy(localGA_tag_list.begin(), localGA_tag_list.end(),
  localGA_tag.begin()); std::copy(localGA_seq_list.begin(),
  localGA_seq_list.end(), localGA_seq.begin()); for (i = 0; i < nlocal; ++i) {
    if(GA_typeCheck[type[i]]) {
      localGA_tag_list.push_back(tag[i]);
      localGA_seq_list.push_back(i);
      //tag2seq[tag[i]] = i;
    }
  }
  for (i = nlocal; i < nlocal + nghost; ++i) {
    if(GA_typeCheck[type[i]]) {
      //tag2seq[tag[i]] = i;
    }
  }
  localGA_num = localGA_tag_list.size();

  localGA_tag.reserve(localGA_num+1);
  std::copy(localGA_tag_list.begin(), localGA_tag_list.end(),
  localGA_tag.begin()); localGA_seq.reserve(localGA_num+1);
  std::copy(localGA_seq_list.begin(), localGA_seq_list.end(),
  localGA_seq.begin());
  */
  if (me == 0) utils::logmesg(lmp, "[ConpGA] all_ghostGA_num = {}, used_ghostGA_num = {}\n", all_ghostGA_num, used_ghostGA_num);
}

/* ---------------------------------------------------------------------- */
//
void FixConpGA::setup(int vflag)
{
  FUNC_MACRO(0);
  if (!force->pair->eatom) {
    error->all(FLERR, "fix conp cannot run without a per-atom eatom\n add: "
                      "compute     eatom all pe/atom_GA\n");
  }

  int i, nlocal = atom->nlocal, n = atom->ntypes;
  int* type = atom->type;
  int *buf, *int_ptr;
  double value;
  if (!electro) {
    electro = new Electro_Control_GA(lmp, this);
  }
  electro->DEBUG_GAtoGA  = DEBUG_GAtoGA;  // For true no Debug
  electro->DEBUG_SOLtoGA = DEBUG_SOLtoGA; // For true no Debug model
  neutrality_state       = false;

  // Setup Self-Interaction energy for each atom type or each localGA
  int dim        = -1;
  self_GreenCoef = (double*)force->pair->extract("self_Greenf", dim);
  // printf("%x %d\n", self_GreenCoef, dim);
  if (dim != 1 || self_GreenCoef == nullptr) error->all(FLERR, "Cannot obtain self_GreenCoef from pair function\n");
  for (i = 1; i <= n; i++) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] type:{}, self_GreenCoef: {:f}\n", i, self_GreenCoef[i]);
  }

  double* ptr;
  ptr = (double*)force->pair->extract("cut_coulsq", dim);
  if (dim != 0 || ptr == nullptr) error->all(FLERR, "Cannot obtain cut_coulsq from pair function\n");
  cut_coulsq = *ptr;
  cut_coul   = sqrt(cut_coulsq);

  if (me == 0) utils::logmesg(lmp, "[ConpGA] cut_coul = {:f}; cut_coulsq = {:f}\n", cut_coul, cut_coulsq);
  if (minimizer == MIN::PCG && pcg_mat_sel == PCG_MAT::DC) {
    if (cut_coul_add > neighbor->skin) {
      if (me == 0)
        utils::logmesg(lmp,
                       "[ConpGA] the delta coul cut-off = {:f} in PCG-DC calculation is "
                       "large than the skin of neighbor = {:f}!\n",
                       cut_coul_add, neighbor->skin);
    }
    if (cut_coul_add + cut_coul < 0)
      error->all(FLERR, "The net cutoff for DC P-matrix generation is smaller than zero!\n");
    else if (me == 0)
      utils::logmesg(lmp, "[ConpGA] PCG cut = {:f}\n", cut_coul_add + cut_coul);
  }
  ptr = (double*)force->pair->extract("g_ewald", dim);
  if (dim != 0 || ptr == nullptr) error->all(FLERR, "Cannot obtain g_ewald from pair function\n");
  pair_g_ewald = *ptr;
  if (me == 0) {
    // qqrd2e = qqr2e/dielectric; in "force.cpp:181"
    utils::logmesg(lmp, "[ConpGA] g_ewald = {:.18f}\n", pair_g_ewald);
    utils::logmesg(lmp,
                   "[ConpGA] the Coulomb's constant is {:18.12f} "
                   "[eneryg_unit/lenght_unit]\n",
                   force->qqrd2e);
  }

  /* sort fix group to first */
  if (igroup != 0 && atom->firstgroup == -1) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] Resort Group [{}] at the top automatically!\n", group->names[igroup]);
    atom->firstgroup = igroup;

    comm->exchange();
    atom->first_reorder();
    comm->borders();
    // list->print_attributes();
    // neighbor->requests[0]->newton = 1;

    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;

    atom->firstgroup = -1;
  }

  if (setup_run == 0) {
    localGA_build();
    if (force->kspace == nullptr || pair_flag || minimizer == MIN::PCG) { // Extra Kspace Check
      localGA_neigh_build();                                              //
    }
  } else {
    // localGA_build();
    localGA_atomsort();
    if (__env_alpha_flag) calc_env_alpha();
  }

  Uchem = 0;
  setup_Vext();
  // if (setup_run == 0)
  update_Vext();

  /*
  double factor_coul = 1, factor_lj = 0, fforce;
  for (i = 1; i <= n; i++)
  {
    self_GreenCoef[i] = force->pair->single(1, 1, i, i, 0, factor_coul,
  factor_lj, fforce); if (me == 0) utils::logmesg(lmp, "Fix::Conp:
  type:%d, self_GreenCoef: %f\n", i, self_GreenCoef[i]); if(self_GreenCoef[i] !=
  self_GreenCoef[i]) error->all(FLERR, "Invalid self_GreenCoef result, may be
  the wrong pair potential\n");
  }
  */

  int* const localGA_seq_arr = &localGA_seq.front();
  localGA_GreenCoef_vec.reserve(localGA_num);
  double* const localGA_GreenCoef = &localGA_GreenCoef_vec.front();
  for (size_t i = 0; i < localGA_num; i++) {
    localGA_GreenCoef[i] = self_GreenCoef[type[localGA_seq_arr[i]]];
    // printf("%d %f\n", i, localGA_GreenCoef[i]);
  }

  // printf("CPU%d:end of neighbor_local, GA_build with localGA_num: %d\n", me,
  // localGA_num);
  /*
  std::vector<int> localGA_tag_arr(1), localGA_seq_arr(1);
  if (localGA_num > 0) {
    localGA_tag_arr.reserve(localGA_num);
    localGA_seq_arr.reserve(localGA_num);
    std::copy(localGA_tag.begin(), localGA_tag.end(), localGA_tag_arr.begin());
    std::copy(localGA_seq.begin(), localGA_seq.end(), localGA_seq_arr.begin());
  }
  */
  MPI_Allreduce(&localGA_num, &totalGA_num, 1, MPI_INT, MPI_SUM, world);
  totalGA_tag.reserve(totalGA_num);
  if (neutrality_flag == true) {
    Xi_self_index %= totalGA_num;
  }
  // std::vector<int> totalGA_tag_arr;
  // totalGA_tag_arr.reserve(totalGA_num);
  // totalGA_tag_arr[totalGA_num-1] = 0;
  // printf("%d, totalGA_tag_arr %lu\n", totalGA_num, totalGA_tag_arr.size());
  buf = &localGA_EachProc.front();
  MPI_Allgather(&localGA_num, 1, MPI_INT, buf, 1, MPI_INT, world);
  // printf("localGA_EachProc size %d, %d\n", localGA_EachProc.capacity(),
  // localGA_tag_arr.capacity());
  localGA_EachProc_Dipl[0] = 0;
  for (i = 1; i < localGA_EachProc.capacity(); ++i) {
    localGA_EachProc_Dipl[i] = localGA_EachProc_Dipl[i - 1] + localGA_EachProc[i - 1];
  }
  /*
  if(me == 0) {
    for (int i = 0; i < 2; ++i)
    {
      printf("cpu: %d %d %d\n", i, localGA_EachProc[i],
  localGA_EachProc_Dipl[i]);
    }
  }
  */

  int_ptr = &totalGA_tag.front();
  MPI_Allgatherv(&localGA_tag.front(), localGA_num, MPI_INT, int_ptr, &localGA_EachProc.front(), &localGA_EachProc_Dipl.front(), MPI_INT, world);
  // printf("FixConpGA::setup CPU %d, %d\n", me, totalGA_num);
  // return;
  /*
  for (i = 0; i < localGA_num; ++i)
  {
    //printf("CPU# %d, %d tag:[%d]\n", me, i, localGA_tag[i]);
  }
  */

  // build background potential for env_alpha
  if (__env_alpha_flag) {
    q_store.reserve(nlocal);
    env_alpha.reserve(localGA_num);
    double* q                   = atom->q;
    double* kspace_eatom        = force->kspace->eatom;
    int* type                   = atom->type;
    double* const q_store_arr   = &q_store.front();
    double* const env_alpha_arr = &env_alpha.front();
    localSOL_num                = nlocal - localGA_num;
    MPI_Allreduce(&localSOL_num, &totalSOL_num, 1, MPI_INT, MPI_SUM, world);

    for (i = 0; i < nlocal; ++i) {
      q_store_arr[i] = q[i];
      if (GA_Flags[type[i]] != 0) {
        q[i] = 1;
      } else {
        q[i] = 0;
      }
    }
    time_t t_begin0 = clock();
    // electro->qsum_qsq();
    electro->compute_GAtoGA();
    // electro->compute(3,0);
    // calculation in kspace;
    update_K_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;
    for (i = 0; i < localGA_num; ++i) {
      env_alpha_arr[i]                 = kspace_eatom[localGA_seq_arr[i]];
      kspace_eatom[localGA_seq_arr[i]] = 0.0;
    }
    std::copy(q_store_arr, q_store_arr + nlocal, q);
    // force->kspace->qsum_qsq();
  }
  // std::sort(int_ptr, int_ptr + totalGA_num);

  /* // A test of comm->forward_comm_fix to exchange ghost charge value
  //int nlocal = atom->nlocal;
  int nmax = (atom->nmax) + 10;
  printf("CPU: %d nlocal: %d, nmax: %d\n", me, nlocal, nmax);
  memory->create(d, nmax, "FixConpGA:nmax");
  for (int i = 0; i < nmax; ++i)
  {
    d[i] = 1.0*me;
  }
  printf("CPU: %d, %f %f %f %f %f\n", me, d[0], d[1], d[2], d[3], d[4]);
  MPI_Barrier(world);
  comm_forward = comm_reverse = 1;
  comm->forward_comm_fix(this);
  printf("CPU: %d, %f %f %f %f %f\n", me, d[0], d[1], d[2], d[3], d[4]);
  */
  Ushift   = 0;
  Ele_iter = 0;
  set_zeros(Ele_Uchem, 0, 3 * Ele_num);

  electro->setup();
  if (firstgroup_flag && comm_recv_arr == nullptr) {
    comm_recv_arr = electro->setup_comm(all_ghostGA_ind);
    // comm_recv_arr == nullptr for procs without GA atoms and neighbors
    if (comm_recv_arr == nullptr) comm_recv_arr = reinterpret_cast<int*>(this);
  }
  group_check();
  setup_GG_Flag();
  if (setup_run > 0) {
    // For pairwise interaction check the consistent of kspace setting.
    if (pair_flag) {
      char INFO[512];
      if (fabs(1 - force->kspace->g_ewald / kspace_g_ewald) > 1e-14) {
        sprintf(INFO, "[ConpGA] Inconsistent kspace setup for %28.26f vs %f\n", kspace_g_ewald, force->kspace->g_ewald);
        error->all(FLERR, INFO);
      }
      if (kspace_nn_pppm[0] != force->kspace->nx_pppm || kspace_nn_pppm[1] != force->kspace->ny_pppm || kspace_nn_pppm[2] != force->kspace->nz_pppm) {
        sprintf(INFO, "[ConpGA] Inconsistent kspace setup for %dx%dx%d vs %dx%dx%d\n", kspace_nn_pppm[0], kspace_nn_pppm[1], kspace_nn_pppm[2],
                force->kspace->nx_pppm, force->kspace->ny_pppm, force->kspace->nz_pppm);
        error->all(FLERR, INFO);
      }
    }
  } else {
    setup_once();
    if (INFO_flag) {
      Print_INFO();
    }
  }
  if (tol_style == TOL::REL_B) {
    calc_B0(B0_prev);
    if (B0_store != nullptr) std::copy(B0_prev, B0_prev + localGA_num, B0_store);
  }

  // setup_run > 0;

  // tol_style = TOL::MAX_Q;

  setup_run++;

  // run two times of pre_force to increase accuracy
  update->ntimestep--;
  if (Smatrix_flag == true) {
    neighbor->ago++;
    equalize_Q();
    neighbor->ago--;
    // fix_forward_comm(0);
    // Uchem = 0;
  } else {
    neighbor->ago++;
    equalize_Q();
    neighbor->ago--;
    //
  }

  update->ntimestep++;
  {
    int nall = atom->nlocal;
    if (force->newton) nall += atom->nghost;
    double** f = atom->f;
    for (int i = 0; i < nall; ++i) {
      if (f[i][0] != 0 || f[i][1] != 0 || f[i][2] != 0) {
        clear_force_flag = true;
        break;
      }
    }
    int clear_force_int = int(clear_force_flag), clear_force_sum = 0;
    MPI_Allreduce(&clear_force_int, &clear_force_sum, 1, MPI_INT, MPI_SUM, world);
    if (clear_force_sum > 0) {
      clear_force_flag = true;
      if (me == 0) utils::logmesg(lmp, "[ConpGA] calculate with force clear!\n");
    } else {
      if (me == 0) utils::logmesg(lmp, "[ConpGA] calculate without force clear!\n");
    }
  }
  neighbor->ago++;
  pre_force(1);
  neighbor->ago--;

  // FOr MIN::INV, run two times of INV_AAA to increase accuracy.
  // if (minimizer == MIN::INV) pre_force(1);
  // necessary for force_clear, because the force has been calculated before
  // fix. force_clear();

  // printf("nlocal = %d; nghost = %d\n", nlocal, atom->nghost);

  force_clear();
  electro->calc_qsqsum();
  force->pair->compute(3, 0);
  electro->compute(3, 0);
  if (newton_pair) {
    comm->reverse_comm();
    fix_reverse_comm(REV_COUL_RM);
  }
  /*
  Uchem = 0;
  update_Vext();
  calc_Uchem_with_zero_Q_sol();
  */
  pre_reverse(3, 0);

  // post_force(1);
  /*
  // Copy from post_force()
  double *q = atom->q;
  double *cl_atom = force->pair->cl_atom;
  double *eatom   = force->pair->eatom;
  //fix_reverse_comm(REV_COUL_RM);
  if (neutrality_flag && Uchem_extract_flag) {
    for (i = 0; i < nlocal; ++i) {
      cl_atom[i] -= Uchem * q[i];
      eatom[i] -= Uchem * q[i];
    }
  }
  */
} // void FixConpGA::setup(int vflag)

void FixConpGA::setup_once()
{
  if (tol_style == TOL::REL_B || !GA2GA_flag) {
    memory->create(B0_prev, localGA_num, "FixConp/GA: B0_prev from ");
    memory->create(B0_next, localGA_num, "FixConp/GA: B0_next from ");
    // memory->create(B0_store, localGA_num, "FixConp/GA: B0_next from ");

    if (force_analysis_flag == true && B0_store == nullptr) {
      memory->create(B0_store, localGA_num, "FixConp/GA: B0_store from ");
    }
  }

  kspace_g_ewald    = force->kspace->g_ewald;
  kspace_nn_pppm[0] = force->kspace->nx_pppm;
  kspace_nn_pppm[1] = force->kspace->ny_pppm;
  kspace_nn_pppm[2] = force->kspace->nz_pppm;
  if (me == 0) utils::logmesg(lmp, "[ConpGA] gewald = {:f}, grid = {}x{}x{}\n", kspace_g_ewald, kspace_nn_pppm[0], kspace_nn_pppm[1], kspace_nn_pppm[2]);

  uself_calc();
  if (force_analysis_flag && Q_GA_store == nullptr) {
    memory->create(Q_GA_store, localGA_num, "FixConp/GA: Q_GA_store from ");
  }
  if (minimizer == MIN::INV) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] MIN allocate vector\n");
    gen_invAAA();
    if (Smatrix_flag == true) make_S_matrix();

    memory->create(BBB, totalGA_num, "FixConpGA:BBB");
    memory->create(BBB_localGA, localGA_num, "FixConpGA:BBB");

  } else if (minimizer == MIN::CG || minimizer == MIN::NEAR) {
    // gen_invAAA();
    if (me == 0) utils::logmesg(lmp, "[ConpGA] CG allocate vector\n");

    p_localGA_vec.reserve(localGA_num);
    ap_localGA_vec.reserve(localGA_num);
    res_localGA_vec.reserve(usedGA_num);
  } else if (minimizer == MIN::PCG_EX) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG-EX allocate vector\n");

    p_localGA_vec.reserve(localGA_num);
    ap_localGA_vec.reserve(localGA_num);
#ifndef INTEL_MPI
    res_localGA_vec.reserve(totalGA_num);
#else
    res_localGA_vec.reserve(totalGA_num + localGA_num);
#endif
    z_localGA_vec.reserve(localGA_num);
    z_localGA_prv_vec.reserve(localGA_num);

    if (me == 0) utils::logmesg(lmp, "[ConpGA] ppcg_cut for AAA^{-1} = {:g}\n", ppcg_invAAA_cut);

    gen_invAAA();
    sparsify_invAAA_EX();
  } else if (minimizer == MIN::PCG) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG allocate vector\n");
    p_localGA_vec.reserve(localGA_num);
    ap_localGA_vec.reserve(localGA_num);
    res_localGA_vec.reserve(usedGA_num);
    z_localGA_vec.reserve(usedGA_num);
    z_localGA_prv_vec.reserve(usedGA_num);
#ifndef CONP_NO_DEBUG
    if (pcg_out != nullptr) {
      memory->create(q_iters, maxiter + 1, localGA_num, "FixConp/GA: q_iters from charge analysis");
      memory->create(pcg_res_norm, maxiter + 1, 5, "FixConp/GA: pcg_res_norm from charge analysis");
      if (me == 0) fprintf(pcg_out, "# analysis for #GA: %d\n", totalGA_num);
    }
#endif
    if (pcg_mat_sel == PCG_MAT::NORMAL || pcg_mat_sel == PCG_MAT::FULL) {
      gen_invAAA();
      // To DO generate Sparse invAAA matrix
      sparsify_invAAA();
    } else if (pcg_mat_sel == PCG_MAT::DC) {
      // 25600
      if (totalGA_num < sparse_pcg_num)
        gen_DC_invAAA();
      else
        gen_DC_invAAA_sparse();
    } else {
      //_self_calc();
    }
  } else {
    error->all(FLERR, "Unkown minimum type");
    // TODO errr
  }

  if (1 || neutrality_flag) {
    if (minimizer == MIN::CG || minimizer == MIN::PCG || minimizer == MIN::PCG_EX || minimizer == MIN::NEAR) {
      // Put meanSum_AAA_cg() into uself_calc();
      meanSum_AAA_cg();
    }
  }
  if (Ele_Cell_Model == CELL::Q) {
    make_ELE_matrix();
  }

  if (minimizer == MIN::PCG && pcg_shift_scale > 0) {
    pcg_P_shift();
  }
}

/* -------------------------------------------------------------------------
      env_alpha_calc_cg() to calculate the environment dependent alpha
      // Using the cg mathod cannot obtain a convergent result
------------------------------------------------------------------------- */
/*
void FixConpGA::env_alpha_calc_cg() {
  FUNC_MACRO(0)
  if(me == 0)
    utils::logmesg(lmp, "[ConpGA] Calculate environment dependent alpha.\n");
  int *tag = atom->tag, *type = atom->type;
  int nlocal = atom->nlocal, localSOL_num = nlocal - localGA_num;
  double * const cl_atom = force->pair->cl_atom;
  double * const kspace_eatom = force->kspace->eatom;
  const double x0 = -0.028;
  int* const GA_seq = &localGA_seq.front();

  bool iGA_Flags;
  int i, j, jj, iseq, jseq, jtag, itype, jtype, jGA_index;
  int iproc, AAA_disp = localGA_EachProc_Dipl[me];


  double *x, *r, *pp, *qq, *s, *pe_RML;
  memory->create(pe_RML, nlocal,   "FixConp/GA: PE_store");

  double *q = atom->q;
  double alpha, beta, gamma_prv, gamma, delta, value;
  memory->create(x,  localGA_num,  "FixConp/GA: PE_store");
  memory->create(r,  localSOL_num, "FixConp/GA: PE_store");
  memory->create(pp, localGA_num,  "FixConp/GA: PE_store");
  memory->create(qq, localSOL_num, "FixConp/GA: PE_store");
  memory->create(s,  localGA_num,  "FixConp/GA: PE_store");
  memory->create(pe_RML, nlocal,   "FixConp/GA: PE_store");

  q_store.reserve(nlocal);
  double * const q_store_arr  = &q_store.front();
  for (i = 0; i < nlocal; ++i) {
    q_store_arr[i] = q[i];
    iGA_Flags = GA_Flags[type[i]];
    if (iGA_Flags) {
      q[i] = 1;
    } else {
      q[i] = 0;
    }
    cl_atom[i] = 0.0;
    // kspace_eatom[i] = 0;
  }

  // Compute the background GA<->GA with all unit charge;
  time_t t_begin0 = clock();
  force_clear();
  force->kspace->qsum_qsq();
  force->kspace->compute(3,0);
calculation in kspace; update_K_time += double(clock() - t_begin0) /
CLOCKS_PER_SEC; for (i = 0; i < nlocal; ++i) { iGA_Flags = GA_Flags[type[i]]; if
(iGA_Flags) { pe_RML[i] = kspace_eatom[i]; q[i] = 0; } else { q[i] = 1;
    }
  }

  // Compute the background Solution<->Solution with all unit charge;
  t_begin0 = clock();
  force_clear();
  force->kspace->qsum_qsq();
  force->kspace->compute(3,0);
calculation in kspace; update_K_time += double(clock() - t_begin0) /
CLOCKS_PER_SEC; for (i = 0; i < nlocal; ++i) { iGA_Flags = GA_Flags[type[i]]; if
(iGA_Flags) { q[i] = x0; } else { pe_RML[i] = kspace_eatom[i];
    }
  }

  bool fast_bvec_flag_store = fast_bvec_flag, pair_flag_store = pair_flag;
  fast_bvec_flag = true; pair_flag = true;




  t_begin0 = clock();
  force_clear();
  fix_forward_comm(0);
  force->kspace->qsum_qsq();
  force->kspace->compute(3,0);
calculation in kspace; update_K_time += double(clock() - t_begin0) /
CLOCKS_PER_SEC; fix_localGA_pairenergy(0, false); j = 0; for (i = 0; i < nlocal;
++i) { iGA_Flags = GA_Flags[type[i]]; if (iGA_Flags) {
      ;
    } else {
      r[j] = 1 - (kspace_eatom[i] + cl_atom[i] - pe_RML[i]);
      // r[j] = (kspace_eatom[i] + cl_atom[i] - pe_RML[i]);
      // printf("r[%d-%d] = %.15g\n", j, i, r[j]);
      j++;
    }
  }
  value = norm_mat_p(r, localSOL_num, "r=");

  group_group_compute(true, r, s, pe_RML);

  gamma = 0;
  for (i = 0; i < localGA_num; ++i)
  {
    x[i] = x0;
    value = pp[i] = s[i];
    // printf("s[%d] = %.15g, %.15g\n", i, s[i], pe_RML[i]);
    gamma += value * value;
  }
  value = norm_mat_p(s, localGA_num, "s=");


  while(1) {
    group_group_compute(false, pp, qq, pe_RML);
    delta = 0;
    for (i = 0; i < localSOL_num; ++i) {
      delta += qq[i] * qq[i];
    }
    value = norm_mat_p(qq, localSOL_num, "qq=");

    alpha = gamma/delta;
    for (i = 0; i < localGA_num; ++i) {
      x[i] += alpha * pp[i];
    }
    for (i = 0; i < localSOL_num; ++i) {
      r[i] -= alpha * qq[i];
    }

    group_group_compute(true, r, s, pe_RML);
    value = norm_mat_p(s, localGA_num, "s=");

    gamma_prv = gamma; gamma = 0;
    for (i = 0; i < localGA_num; ++i) {
      gamma += s[i] * s[i];
    }
    beta = gamma / gamma_prv;
    for (i = 0; i < localGA_num; ++i) {
      pp[i] = s[i] + beta * pp[i];
    }
    value = norm_mat_p(pp, localGA_num, "pp=");
    printf("gamma = %.16g\n", gamma);
  } // 1

  for (i = 0; i < nlocal; ++i)
  {
    printf("%d[type =%d], coul = %.15g, kspace = %.15g, pe_RML = %.15g\n", i,
type[i], cl_atom[i], kspace_eatom[i], pe_RML[i]);
  }

  // Recover the setting
  force_clear();

  fast_bvec_flag = fast_bvec_flag_store;
  pair_flag = pair_flag_store;
  std::copy(q_store_arr, q_store_arr + nlocal, q);
  fix_forward_comm(0);
  memory->destroy(x);
  memory->destroy(r);
  memory->destroy(pp);
  memory->destroy(qq);
  memory->destroy(s);
}
*/

/* ----------------------------------------------------------------------
  Compute the group group interaction in kspace
  to_GA_Flag := True  From Electrolyt to Electrode
  to_GA_Flag := False From Electrode  to Electrolyte
---------------------------------------------------------------------- */
/*
void FixConpGA::group_group_compute(bool to_GA_Flag, const double* const q_set,
double* pe_out, const double* const pe_RML) { if(me == 0 and
DEBUG_LOG_LEVEL > 0) utils::logmesg(lmp, "[ConpGA] Calculate environment dependent
alpha.\n"); int *tag = atom->tag, *type = atom->type; int nlocal = atom->nlocal;
  bool iGA_Flags, src_group_Flag;
  double * const cl_atom = force->pair->cl_atom;
  double * const kspace_eatom = force->kspace->eatom;
  double *q = atom->q;
  int i, j, jj, iseq, jseq, jtag, itype, jtype, jGA_index;
  force_clear();
  j = 0;
  for (i = 0; i < nlocal; ++i) {
    iGA_Flags = GA_Flags[type[i]];
    // XOR     to_GA_Flag   1   0
    // iGA_Flags        1   0   1
    //                  0   1   0
    //
    src_group_Flag = (iGA_Flags ^ to_GA_Flag);
    if (src_group_Flag) {
      q[i] = q_set[j];
      j++;
    } else {
      q[i] = 1;
    }
  }

  time_t t_begin0 = clock();
  fix_forward_comm(0);
  force->kspace->qsum_qsq();
  force->kspace->compute(3,0);
calculation in kspace; update_K_time += double(clock() - t_begin0) /
CLOCKS_PER_SEC;

  t_begin0 = clock();
  fix_localGA_pairenergy(0, to_GA_Flag);
  update_P_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;

  j = 0;
  for (i = 0; i < nlocal; ++i) {
    iGA_Flags = GA_Flags[type[i]];
    src_group_Flag = (iGA_Flags ^ to_GA_Flag) ;
    if (src_group_Flag) {
      ;
    } else {
      pe_out[j] = (kspace_eatom[i] + cl_atom[i] - pe_RML[i]);
      j++;
    }
  }
}
*/

// make_ELE_matrix
void FixConpGA::make_ELE_matrix()
{
  memory->create(Ele_matrix, Ele_num, Ele_num, "fix_Conp/GA::make_ELE_matrix()::Ele_matrix");
  memory->create(Ele_D_matrix, localGA_num, Ele_num, "fix_Conp/GA::make_ELE_matrix()::Ele_D_matrix");

  if (AAA != nullptr) {
    return make_ELE_matrix_direct();
  } else {
    switch (minimizer) {
      case MIN::CG:
      case MIN::PCG:
      case MIN::PCG_EX:
      case MIN::NEAR:
        make_ELE_matrix_iterative();
        break;
      default:
        error->all(FLERR, "meanSum_AAA_cg() only valid for MIN::CG, MIN::PCG or "
                          "MIN::PPCG methods");
    }
  }
}

// Direct obtain the Ele interaction matrix between the Electrode
void FixConpGA::make_ELE_matrix_direct()
{
  FUNC_MACRO(0);
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Compute the ELE_matrix directly from the inv_E\n");
  // double** matrix;
  int* totalGA_eleIndex;
  int Ele_num_sqr = Ele_num * Ele_num;
  memory->create(totalGA_eleIndex, totalGA_num, "fix_Conp/GA::make_ELE_matrix_direct()::totalGA_types");
  // memory->create(matrix, Ele_num+1, Ele_num+1,
  // "fix_Conp/GA::make_ELE_matrix_direct()::matrix");
  MPI_Allgatherv(localGA_eleIndex, localGA_num, MPI_INT, totalGA_eleIndex, &localGA_EachProc.front(), &localGA_EachProc_Dipl.front(), MPI_INT, world);

  set_zeros(&Ele_matrix[0][0], 0, Ele_num_sqr);
  set_zeros(&Ele_D_matrix[0][0], 0, localGA_num * Ele_num);
  for (int i = 0; i < localGA_num; i++) {
    int iEle = localGA_eleIndex[i];
    for (int j = 0; j < totalGA_num; j++) {
      int jEle = totalGA_eleIndex[j];
      // printf("iEle = %d, jEle = %d\n", iEle, jEle);
      double value = AAA[i][j];
      Ele_D_matrix[i][jEle] += value;
      Ele_matrix[iEle][jEle] += value;
    }
    /* code */
  }

  MPI_Allreduce(MPI_IN_PLACE, &Ele_matrix[0][0], Ele_num_sqr, MPI_DOUBLE, MPI_SUM, world);

#ifndef CONP_NO_DEBUG
  if (me == 0 && DEBUG_LOG_LEVEL > 0) print_vec(Ele_matrix[0], Ele_num_sqr, "inv_Ele_matrix");
  inv_AAA_LAPACK(Ele_matrix[0], Ele_num);
  if (me == 0 && DEBUG_LOG_LEVEL > 0) print_vec(Ele_matrix[0], Ele_num_sqr, "Ele_matrix");
#endif

  memory->destroy(totalGA_eleIndex);
  // memory->destroy(matrix);
}

void FixConpGA::make_ELE_matrix_iterative()
{
  FUNC_MACRO(0);
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Compute the ELE_matrix iteratively\n");
  int n = atom->ntypes, *type = atom->type, Ele_num_sqr = Ele_num * Ele_num;
  int nlocal                  = atom->nlocal;
  double* q                   = atom->q;
  const int* const GA_seq_arr = &localGA_seq.front();
  double* const cl_atom       = electro->cl_atom();
  double* const ek_atom       = electro->ek_atom();
  q_store.reserve(localGA_num);
  double* const q_store_arr = &q_store.front();

  TOL tol_style_store      = tol_style;
  double tolerance_store   = tolerance;
  bool postreat_flag_store = postreat_flag;

  postreat_flag = false;
  tolerance     = 1e-20;
  tol_style     = TOL::MAX_Q;
  if (me == 0)
    utils::logmesg(lmp,
                   "[ConpGA] Switch to max_Q criterion to calculate Ele_D_matrix with "
                   "the tolerance of {:g}\n",
                   tolerance);

  for (int i = 0; i < localGA_num; i++) {
    int iseq       = GA_seq_arr[i];
    q_store_arr[i] = q[iseq];
    q[iseq]        = 1e-30;
    // qtmp = q[iseq];
    cl_atom[iseq] = 0.0; // (-1 + localGA_Vext[i]); // * qtmp;
  }

  set_zeros(&Ele_matrix[0][0], 0, Ele_num_sqr);
  for (int iEle = 0; iEle < Ele_num; iEle++) {
    for (int i = 0; i < localGA_num; i++) {
      int iseq       = GA_seq_arr[i];
      double delta_i = localGA_eleIndex[i] == iEle;
      // The extra localGA_Vext will be cancelled in the following update charge
      // by Eq = b-X
      ek_atom[iseq] = (localGA_Vext[i] - delta_i);
    }
    switch (minimizer) {
      case MIN::CG:
        update_charge_cg();
        break;
      case MIN::PCG:
        update_charge_pcg();
        break;
      case MIN::PCG_EX:
        update_charge_ppcg();
        break;
      default:
        error->all(FLERR, "make_ELE_matrix_iterative() only valid for MIN::CG, "
                          "MIN::PCG or MIN::PPCG methods");
    }
    // double res = 0;
    for (int i = 0; i < localGA_num; i++) {
      int iseq     = GA_seq_arr[i];
      int i_in_Ele = localGA_eleIndex[i];
      double value = q[iseq];
      // res += value;
      Ele_D_matrix[i][iEle] = value;
      Ele_matrix[iEle][i_in_Ele] += value;
      q[iseq] = 1e-30;
    }
    // printf("res = %.18f\n", res);
    /* code */
  } // iEle

  tolerance     = tolerance_store;
  tol_style     = tol_style_store;
  postreat_flag = postreat_flag_store;

  for (int i = 0; i < localGA_num; i++) {
    int iseq = GA_seq_arr[i];
    q[iseq]  = q_store_arr[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &Ele_matrix[0][0], Ele_num_sqr, MPI_DOUBLE, MPI_SUM, world);
#ifndef CONP_NO_DEBUG
  if (me == 0 && DEBUG_LOG_LEVEL > 0) print_vec(Ele_matrix[0], Ele_num_sqr, "inv_Ele_matrix");
  inv_AAA_LAPACK(Ele_matrix[0], Ele_num);
  if (me == 0 && DEBUG_LOG_LEVEL > 0) print_vec(Ele_matrix[0], Ele_num_sqr, "Ele_matrix");
#endif
}
// Transform the distributed AAA matrix into S matrix for neutral electrolyte
void FixConpGA::make_S_matrix()
{
  FUNC_MACRO(0);
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Generate Smat for neutral electrolyte solution\n");
  int AAA_disp, i, j;
  double AAA_sumsum_sqrt = sqrt(fabs(inv_AAA_sumsum));
  // where c is c'/sqrt(sum(E))
  double *c, *clocal;
  memory->create(clocal, localGA_num,
                 "FixConp/GA: c from the neutralization vector [c] with "
                 "totalGA_num entries");
  memory->create(c, totalGA_num,
                 "FixConp/GA: c from the neutralization vector [c] with "
                 "totalGA_num entries");

  for (i = 0; i < localGA_num; ++i) {
    clocal[i] = AAA_sum1D[i] * AAA_sumsum_sqrt;
  }

  MPI_Allgatherv(clocal, localGA_num, MPI_DOUBLE, c, &localGA_EachProc.front(), &localGA_EachProc_Dipl.front(), MPI_DOUBLE, world);
  /*
  for (i = 0; i < localGA_num; ++i)
  {
    printf("%d = %f, %f\n", i, c[i], AAA_sum1D[i]);
  }
  */

  AAA_disp = localGA_EachProc_Dipl[me];
  // if (inv_AAA_sumsum < 0) {
  // write_mat(c, totalGA_num, "C.mat");

  // write_mat(AAA[0], totalGA_num*totalGA_num, "AAA0.mat");
  for (i = 0; i < localGA_num; ++i, ++AAA_disp) {
    for (j = 0; j < totalGA_num; ++j) {
      AAA[i][j] += c[AAA_disp] * c[j];
    }
  }
  // write_mat(AAA[0], totalGA_num*totalGA_num, "S.mat");

  /*} else {
    for (i = 0; i < localGA_num; ++i, ++AAA_disp) {
      for (j = 0; j < totalGA_num; ++j) {
        AAA[i][j] -= c[AAA_disp]*c[j];
      }
    }
  }
  */
}

double** FixConpGA::pair_matrix(double** pAAA_local)
{
  FUNC_MACRO(0);
  int i, j, ii, jj;
  const int* const localGA_seq_arr = &localGA_seq.front();
  const int* const localGA_tag_arr = &localGA_tag.front();
  const int* const totalGA_tag_arr = &totalGA_tag.front();
  const int* const firstneigh_seq  = &localGA_firstneigh.front();

  const double* const localGA_GreenCoef = &localGA_GreenCoef_vec.front();
  const double* const firstneigh_coeff  = &localGA_firstneigh_coeff.front();

  int iseq, iGA_index, jGA_index, itag, jtag;
  double** pAAA_Global;
  set_zeros(pAAA_local[0], 0, localGA_num * totalGA_num);

  map<int, int> tag2GA_index;
  MAP_RESERVE(tag2GA_index, totalGA_num);
  for (i = 0; i < totalGA_num; i++) {
    tag2GA_index[totalGA_tag[i]] = i;
  }

  int *tag    = atom->tag, jseq;
  int my_disp = localGA_EachProc_Dipl[me];
  for (i = 0; i < localGA_num; i++) {
    iseq                       = localGA_seq_arr[i];
    pAAA_local[i][i + my_disp] = localGA_GreenCoef[i];
    for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj++) {
      jseq      = firstneigh_seq[jj];
      jtag      = jseq;
      jGA_index = tag2GA_index.at(jtag);
      pAAA_local[i][jGA_index] += firstneigh_coeff[jj];
    }
    for (jj = localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
      jtag = firstneigh_seq[jj];
      jseq = atom->map(jtag);

      jGA_index = tag2GA_index.at(jtag);
      // printf("i = %d, iseq = %d, jtag = %d, jseq = %d, jGA_index = %d,
      // %.12f\n", i, iseq, jtag, jseq, firstneigh_coeff[jj], jGA_index);
      pAAA_local[i][jGA_index] += firstneigh_coeff[jj];
    }
  }

  if (me == 0) {
    memory->create(pAAA_Global, totalGA_num, totalGA_num, "FixConp/GA:pAAA_Global");
  } else {
    pAAA_Global = pAAA_local;
  }
  int* size_EachProc = &localGA_EachProc.front();
  int* size_Dipl     = &localGA_EachProc_Dipl.front();
  MPI_Datatype MPI_totalGA;
  MPI_Type_contiguous(totalGA_num, MPI_DOUBLE, &MPI_totalGA);
  MPI_Type_commit(&MPI_totalGA);

  MPI_Gatherv(pAAA_local[0], localGA_num, MPI_totalGA, pAAA_Global[0], size_EachProc, size_Dipl, MPI_totalGA, 0, world);
  if (me == 0) {
    for (int iGA = 0; iGA < totalGA_num; ++iGA) {
      for (int jGA = 0; jGA < iGA; ++jGA) {
        double value_i = pAAA_Global[iGA][jGA];
        double value_j = pAAA_Global[jGA][iGA];
        double value   = (value_i + value_j);
        if (value_j != 0 && value_i != 0) {
          value = 0.5 * (value_i + value_j);
        }
        pAAA_Global[iGA][jGA] = pAAA_Global[jGA][iGA] = value;
      }
    }
  }
  // if(me == 0) {
  //   write_mat(pAAA_Global[0], totalGA_num*totalGA_num,"pppp_AAA.mat");
  // }
  matrix_scatter(pAAA_Global, pAAA_local, totalGA_num, size_EachProc, size_Dipl);
  if (me == 0) {
    memory->destroy(pAAA_Global);
  } else {
    pAAA_Global = 0;
  }
  MPI_Type_commit(&MPI_totalGA);
  return pAAA_local;
}

void FixConpGA::matrix_scatter(double**& mat_all, double**& mat, int N, const int* size_EachProc, const int* size_Disp)
{
  MPI_Datatype MPI_totalN;
  MPI_Type_contiguous(N, MPI_DOUBLE, &MPI_totalN);
  MPI_Type_commit(&MPI_totalN);

  int i;
#ifdef INTEL_MPI
  {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] Start to redistribute inv_AAA for INTEL MPI!\n");
    MPI_Request* request = new MPI_Request[nprocs];
    MPI_Status* status   = new MPI_Status[nprocs];
    for (i = nprocs - 1; i > -1; --i) {
      if (me == 0 && size_EachProc[i] > 0) {
        MPI_Isend(mat_all[size_Disp[i]], size_EachProc[i], MPI_totalN, i, i, world, &request[i]);
      }
      if (me == i && size_EachProc[i] > 0) {
        MPI_Recv(mat[0], size_EachProc[i], MPI_totalN, 0, i, world, &status[i]);
      }
    }
    MPI_Barrier(world);
    delete[] request;
    delete[] status;
  }
#else
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Start to redistribute pAAA.\n");
  MPI_Scatterv(mat_all[0], size, size_Dipl, MPI_totalN, mat[0], size_EachProc[me], MPI_totalN, 0, world);
#endif
  MPI_Type_free(&MPI_totalN);
}

/* -------------------------------------------------------------------------
    calc_B0() to calculate the GA<->GA interaction potential
------------------------------------------------------------------------- */

void FixConpGA::calc_B0(double* B0)
{
  FUNC_MACRO(0);
  if (me == 0 && DEBUG_LOG_LEVEL > 0)
    utils::logmesg(lmp, "[ConpGA] cal_B0: calcualte the potential B0 by the current electrode charge alone with tol_style = {}\n", tol_style);
  int i, j, ii, iter;
  int nlocal = atom->nlocal, *type = atom->type, *tag = atom->tag;
  int* GA_seq_arr = &localGA_seq.front();
  int iseq;
  double value, qtmp, tmp;
  double* q         = atom->q;
  double* ek_atom   = electro->ek_atom();
  double qsum_local = 0, qsqsum_local = 0, qsum, qsqsum;

  t_begin = clock();

  q_store.reserve(nlocal);
  double* const q_store_arr = &q_store.front();
  for (i = 0; i < nlocal; ++i) {
    q_store_arr[i] = q[i];
    q[i]           = 0;
  }

  for (i = 0; i < localGA_num; i++) {
    iseq = GA_seq_arr[i];
    qtmp = q[iseq] = q_store_arr[iseq];
    qsum_local += qtmp;
    qsqsum_local += qtmp * qtmp;
  }
  MPI_Allreduce(&qsum_local, &qsum, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&qsqsum_local, &qsqsum, 1, MPI_DOUBLE, MPI_SUM, world);

  time_t t_begin = clock();
  electro->set_qsqsum(qsum, qsqsum);
  fix_forward_comm(0);
  electro->pair_GAtoGA();
  update_P_time += double(clock() - t_begin) / CLOCKS_PER_SEC;

  t_begin = clock();
  electro->compute_GAtoGA();
  electro->calc_GA_potential(STAT::POT, STAT::POT, SUM_OP::ON);
  update_K_time += double(clock() - t_begin) / CLOCKS_PER_SEC;

  for (i = 0; i < localGA_num; i++) {
    iseq  = GA_seq_arr[i];
    B0[i] = ek_atom[iseq];
    // printf("B0_prev[%d] = %.16f\n", i, B0_prev[i]);
  }

  std::copy(q_store_arr, q_store_arr + nlocal, q);
  electro->qsum_qsq();
  fix_forward_comm(0);
}

/* -------------------------------------------------------------------------
          pcg_P_shift() to shift the value of sparse P by a constant
------------------------------------------------------------------------- */
void FixConpGA::pcg_P_shift()
{
  int i, j, local_nbr_sum, total_nbr_sum;
  double total_inva_sum;
  double* const inva = &localGA_firstneigh_inva.front();

  int inva_size = localGA_firstneigh_inva.capacity();

  double local_inva_sum = 0;
  for (i = localGA_num; i < inva_size; ++i) {
    local_inva_sum += inva[i];
  }

  local_nbr_sum = localGA_firstneigh.capacity();
  if (newton_pair) {
    local_inva_sum *= 2;
    local_nbr_sum *= 2;
  }

  for (i = 0; i < localGA_num; ++i) {
    local_inva_sum += inva[i];
  }

  MPI_Allreduce(&local_nbr_sum, &total_nbr_sum, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&local_inva_sum, &total_inva_sum, 1, MPI_DOUBLE, MPI_SUM, world);
  total_nbr_sum += totalGA_num;

  pcg_shift = (1 / inv_AAA_sumsum - total_inva_sum) / (double(totalGA_num) * totalGA_num - total_nbr_sum);

  if (me == 0) {
    double value = double(total_nbr_sum) / totalGA_num;
    utils::logmesg(lmp, "[ConpGA] Ave. non-zero per column: {:.3f}/{} [{:3f}%%]\n", value, totalGA_num, value * 100 / totalGA_num);
    utils::logmesg(lmp, "[ConpGA] The shift for sparse P: {:16g} (x{:f})\n", pcg_shift, pcg_shift_scale);
  }
  pcg_shift *= pcg_shift_scale;

  if (pcg_mat_sel == PCG_MAT::FULL || pcg_mat_sel == PCG_MAT::NORMAL) {
    for (i = 0; i < inva_size; ++i) {
      inva[i] -= pcg_shift;
    }
  }
  // printf("local = %d\n", local_nbr_sum);
}

void FixConpGA::gen_DC_invAAA_sparse()
{
  FUNC_MACRO(0)
  int i, iseq, itype, itag, jnum, jrow_index;
  int nlocal = atom->nlocal, nghost = atom->nghost, ntotal = nlocal + nghost;
  int jj, j, *jlist, jseq, jtag, jtype, jGA_index, *tag = atom->tag;
  int k, iproc, AAA_nonzero_local, AAA_nonzero, max_neighbor = 0, i_neighbor;
  int neighbor_size_local = pcg_neighbor_size_local; // lo_calGA_firstneigh.capacity(),
  int neighbor_size;
  int AAA_disp = localGA_EachProc_Dipl[me];
  int i_index, j_index, ii, *type = atom->type;
  double xtmp, ytmp, ztmp, qtmp, delx, dely, delz, rsq, value;
  // double ** AAA_tmp;
  double *q = atom->q, **x = atom->x;
  int inum           = list->inum;
  int* ilist         = list->ilist;
  int* numneigh      = list->numneigh;
  int** firstneigh   = list->firstneigh;
  double factor_coul = 1, factor_lj = -2, fforce = gewald_tune_factor;
  const double* const localGA_GreenCoef = &localGA_GreenCoef_vec.front();
  char info[512];
  utils::logmesg(lmp, "[ConpGA] core: {}, neighbor_size_local = {} vs {}\n", me, neighbor_size_local, localGA_firstneigh.capacity());
  localGA_firstneigh_inva.reserve(neighbor_size_local + localGA_num);
  localGA_firstneigh_localIndex.reserve(neighbor_size_local);

  // memory->create(AAA, totalGA_num, totalGA_num, "FixConp/GA: AAA from
  // gen_pseudo_invAAA_select");
  AAA_nonzero_local = neighbor_size_local + localGA_num;

  int *spAAA_nonzero_disp, *spAAA_nonzero;
  memory->create(spAAA_nonzero, nprocs, "FixConp/GA: spAAA_nonzero from gen_DC_invAAA_sparse");
  memory->create(spAAA_nonzero_disp, nprocs + 1, "FixConp/GA: spAAA_nonzero_disp from gen_DC_invAAA_sparse");

  MPI_Allgather(&AAA_nonzero_local, 1, MPI_INT, spAAA_nonzero, 1, MPI_INT, world);
  spAAA_nonzero_disp[0] = 0;
  // printf("iproc[%d] = nonzero %d\n", 0, spAAA_nonzero[0]);
  for (iproc = 1; iproc < nprocs + 1; ++iproc) {
    // printf("iproc[%d] = nonzero %d\n", iproc, spAAA_nonzero[iproc]);
    spAAA_nonzero_disp[iproc] = spAAA_nonzero_disp[iproc - 1] + spAAA_nonzero[iproc - 1];
  }

  AAA_nonzero_local = neighbor_size_local + localGA_num;
  MPI_Allreduce(&AAA_nonzero_local, &AAA_nonzero, 1, MPI_INT, MPI_SUM, world);
  // printf("AAA_nonzero = %d/%d\n", AAA_nonzero_local, AAA_nonzero);
  spMatEle *spAAA, *spAAA_ptr, *spAAA_rev_ptr;
  memory->create(spAAA, AAA_nonzero_local, "FixConp/GA: spAAA from gen_DC_invAAA_sparse");

  const int* const localGA_seq_arr = &localGA_seq.front();
  const int* const localGA_tag_arr = &localGA_tag.front();
  const int* const totalGA_tag_arr = &totalGA_tag.front();
  const int* const firstneigh_seq  = &localGA_firstneigh.front();

  double coul = cut_coul + cut_coul_add, coulsq = coul * coul;
  double* const firstneigh_inva_self = &localGA_firstneigh_inva.front();
  double* const firstneigh_inva      = firstneigh_inva_self + localGA_num;
  int* const firstneigh_localIndex   = &localGA_firstneigh_localIndex.front();

  map<int, int> tag_To_localIndex;
  MAP_RESERVE(tag_To_localIndex, usedGA_num);
  for (i = 0; i < usedGA_num; i++) {
    tag_To_localIndex[localGA_tag_arr[i]] = i;
  }

  map<int, int> tag2GA_index;
  MAP_RESERVE(tag2GA_index, totalGA_num);
  for (i = 0; i < totalGA_num; i++) {
    tag2GA_index[totalGA_tag[i]] = i;
  }

  double N_shift, N;
  N = fit_erfc(gewald_tune_factor, coul, N_shift);
  if (me == 0) {
    utils::logmesg(lmp,
                   "[ConpGA] Generate DC PCG matrix with Dampered_alpha = {:f}, "
                   "compared with PPPM g_ewald = {:f}\n",
                   gewald_tune_factor, pair_g_ewald);
    utils::logmesg(lmp,
                   "[ConpGA] Using Sparse Routine with N = {:f} and N_shift = %f with "
                   "size_subAAA = %d\n",
                   N, N_shift, size_subAAA);
  }

  t_begin = clock();
  double r_dist, qscale = force->qqrd2e / (N * N + 1); // , r_scale;
  double inv_cut_coul_two = (N + 1) / coul, inv_cut_coulsq = 1 / coulsq, inv_cut_coul = 1 / coul;
  double sigma = 1 / gewald_tune_factor;

  spAAA_ptr = spAAA; // + spAAA_nonzero_disp[me];
  // printf("me = %d; disp = %d/%d;\n", me, spAAA_nonzero_disp[me],
  // AAA_nonzero);

  if (me == 0) utils::logmesg(lmp, "[ConpGA] PCG_cut = {:f}\n", coul);
  int local_pair = localGA_num;
  int ghost_pair = 0;
  spAAA_rev_ptr  = spAAA_ptr + spAAA_nonzero[me];

  // printf("localGA_numneigh_max = %d\n", localGA_numneigh_max);
  spMatEle* pair_store;
  int edgeGA_num = 0;

  memory->create(pair_store, localGA_numneigh_max, "FixConp/GA: spAAA_nonzero_disp from gen_DC_invAAA_sparse");
  for (i = 0; i < localGA_num; ++i) {
    iseq             = localGA_seq_arr[i];
    bool ilocal_flag = iseq < nlocal ? true : false;
    qtmp             = q[iseq] * 2; // To half the interaction energy;
    xtmp             = x[iseq][0];
    ytmp             = x[iseq][1];
    ztmp             = x[iseq][2];
    jlist            = firstneigh[iseq];
    jnum             = numneigh[iseq];
    itype            = type[iseq];
    // cl_atom[iseq] += qtmp * qtmp * localGA_GreenCoef[ii];

    itag    = tag[iseq];
    i_index = i + localGA_EachProc_Dipl[me];

    (*spAAA_ptr)(i_index, i_index, localGA_GreenCoef[i]);
    spAAA_ptr++;

    int ipair     = 0;
    bool isEdgeGA = false;
    for (jj = 0; jj < jnum; jj++) {
      jseq = jlist[jj];
      // factor_coul = special_coul[sbmask(j)];
      jseq &= NEIGHMASK;
      jtype = type[jseq];

      if (GA_Flags[jtype] == 0) {
        continue;
      }
      delx = xtmp - x[jseq][0];
      dely = ytmp - x[jseq][1];
      delz = ztmp - x[jseq][2];
      rsq  = delx * delx + dely * dely + delz * delz;
      if (rsq > coulsq) continue;
      jtag    = tag[jseq];
      j_index = tag2GA_index[jtag];

      // r_dist = sqrt(rsq);  value = (N + 0.27296801036977642)/r_dist +
      // r_dist*inv_cut_coulsq - inv_cut_coul_two;
      r_dist = sqrt(rsq);
      // r_dist = coul;
      // r_scale = r_dist * inv_cut_coul;
      // value = gewald_tune_factor * (1 / r_dist -inv_cut_coul);
      //
      value = N_shift / r_dist + r_dist * inv_cut_coulsq - inv_cut_coul_two;
      // printf("r_dist = %f, %.12f\n", coul, value);
      // double value2 = gsl_sf_erfc(gewald_tune_factor * r_dist)/r_dist;
      // MathSpecial::erfcx_y100(400.0/(4.0+ gewald_tune_factor * r_dist));
      // if (i == 10) printf("value = %f;%f;r = %f;\n", value, value2, r_dist);

      bool jlocal_flag = jseq < nlocal ? true : false;
      if (jlocal_flag == false) isEdgeGA = true;
      // if (ilocal_flag && jlocal_flag) local_interaction ++;
      // else ghost_interaction ++;
      // value = gsl_sf_erfc(gewald_tune_factor * r_dist) -
      // gsl_sf_erfc(gewald_tune_factor * coul); value /= r_dist; double qq =
      // r_dist * inv_cut_coul; value =
      // -gsl_sf_erfc((0.5)*gewald_tune_factor)*(-0.5) -
      // gewald_tune_factor*MathSpecial::expmsq(gewald_tune_factor) *
      // 0.5641895835477563; value +=
      // gsl_sf_erfc((qq-0.5)*gewald_tune_factor)*(0.5-qq) +
      // gewald_tune_factor*MathSpecial::expmsq(qq*gewald_tune_factor) *
      // 0.5641895835477563; value =
      // -gsl_sf_erfc(r_scale/sigma)*r_scale+sigma*MathSpecial::expmsq(r_scale/sigma)
      // * 0.5641895835477563; if (i == 10) printf("value = %f;r = %f(%f);\n",
      // value, r_dist, qq); value /= r_dist;

      if (i_index < j_index)
        pair_store[ipair](i_index, j_index, qscale * value);
      else
        pair_store[ipair](j_index, i_index, qscale * value);
      ipair++;
      // AAA_me[i][j_index] = qscale * value;
    } // jj
    if (isEdgeGA) {
      ghost_pair += ipair;
      spAAA_rev_ptr -= ipair;
      std::copy(pair_store, pair_store + ipair, spAAA_rev_ptr);
      edgeGA_num++;
    } else {
      local_pair += ipair;
      std::copy(pair_store, pair_store + ipair, spAAA_ptr);
      spAAA_ptr += ipair;
    }
    // AAA_me[i][i + AAA_disp] = localGA_GreenCoef[i]; // Self-interactions
    // coeff
  } // i
  // printf("core: %d, delta = %d/%d\n", me, spAAA_ptr - spAAA -
  // spAAA_nonzero_disp[me], AAA_nonzero_local); if (spAAA_ptr - (spAAA +
  // spAAA_nonzero_disp[me]) != spAAA_nonzero[me]) {
  utils::logmesg(lmp, "[ConpGA] gen_DC_invAAA_sparse() me = {}, nonzero = {} vs {}\n", me, spAAA_ptr - (spAAA), spAAA_nonzero[me]);
  // }
  t_end = clock();
  if (me == 0 && DEBUG_LOG_LEVEL > 0)
    utils::logmesg(lmp,
                   "[Debug] me = {}, Edge atom [{}/{}], Ghost vs Local = {}[{}] vs {} "
                   "with ghost rate {:.3f}%%;\n",
                   me, edgeGA_num, localGA_num, ghost_pair, spAAA_nonzero[me], local_pair, 100.0 * ghost_pair / spAAA_nonzero[me]);
  if (spAAA_rev_ptr - spAAA_ptr == -1) {
    // TODO delete the ptr position check in future
    ; // printf("error structure of input diff = %d\n", spAAA_rev_ptr -
      // spAAA_ptr);
  }

  spAAA_rev_ptr = nullptr;
  int *num_ghost_pair, *num_ghost_pair_disp;
  memory->create(num_ghost_pair, nprocs, "FixConp/GA: num_ghost_pair @ gen_DC_invAAA_sparse()");
  memory->create(num_ghost_pair_disp, nprocs + 1, "FixConp/GA: num_ghost_pair @ gen_DC_invAAA_sparse()");

  ghost_pair += local_pair;
  spAAA_ptr -= local_pair;
  MPI_Allgather(&ghost_pair, 1, MPI_INT, num_ghost_pair, 1, MPI_INT, world);
  // printf("x01\n");
  int total_ghost_pair   = 0;
  num_ghost_pair_disp[0] = 0;
  for (int i = 0; i < nprocs; i++) {
    total_ghost_pair += num_ghost_pair[i];
    num_ghost_pair_disp[i + 1] = num_ghost_pair_disp[i] + num_ghost_pair[i];
  }
  spMatEle* spAAA_Ghost;
  memory->create(spAAA_Ghost, total_ghost_pair, "FixConp/GA: spAAA_Ghost @ gen_DC_invAAA_sparse()");
  // printf("total_ghost_pair = %d\n", total_ghost_pair);

  if (me == 0) utils::logmesg(lmp, "[ConpGA] 01 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);
  /*
  for (iproc = 0; iproc < nprocs; ++iproc) {
    AAA_disp = spAAA_nonzero_disp[iproc];
    // if (me == 0) printf("core: %d, nonzero %d\n", iproc,
  spAAA_nonzero[iproc]); MPI_Bcast(spAAA+AAA_disp,
  spAAA_nonzero[iproc]*sizeof(spMatEle), MPI_BYTE, iproc, world);
    //
  }
  */
  MPI_Datatype MPI_spMAT;
  MPI_Type_contiguous(sizeof(spMatEle) / sizeof(char), MPI_CHAR, &MPI_spMAT);
  MPI_Type_commit(&MPI_spMAT);
  MPI_Allgatherv(spAAA_ptr, ghost_pair, MPI_spMAT, spAAA_Ghost, num_ghost_pair, num_ghost_pair_disp, MPI_spMAT, world);
  MPI_Type_free(&MPI_spMAT);

  t_end = clock();
  if (me == 0) {
    utils::logmesg(lmp, "[ConpGA] 02 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);
    utils::logmesg(lmp,
                   "[ConpGA] The sparse rate of Ewald interaction for short-ranged is "
                   "{:.12f}\n",
                   (double)(AAA_nonzero) / ((double)(totalGA_num)*totalGA_num));
  }

  AAA_disp = localGA_EachProc_Dipl[me];

  SpMatCLS spMat(spAAA, local_pair, spAAA_Ghost, total_ghost_pair, totalGA_num, AAA_disp, AAA_disp + localGA_num);
  // SpMatCLS spMat(spAAA_Ghost, 0, spAAA_Ghost, total_ghost_pair, totalGA_num,
  // AAA_disp, AAA_disp + localGA_num); printf("x02\n");
  free(spAAA);
  free(spAAA_Ghost);
  free(spAAA_nonzero);
  free(spAAA_nonzero_disp);
  if (me == 0) utils::logmesg(lmp, "[ConpGA] size of spAAA at CPU:0 is {}\n", spMat.size());

  // memory->create(AAA, totalGA_num, totalGA_num, "FixConp/GA: AAA from
  // gen_pseudo_invAAA_select");

  // double* const AAA0 = &AAA[0][0];
  const size_t size_AAA     = (size_t)(totalGA_num)*totalGA_num;
  const size_t count_subAAA = size_AAA / size_subAAA + 1;
  double** AAA_L1;
  bool* AAA_L1_Flag;
  size_t idx_AAA, idx_L1, idx_L2, AAA_L1_nonzero = 0;
  // if (me == 0) utils::logmesg(lmp, "size_AAA = %ld; count_subAAA
  // %ld\n", size_subAAA, count_subAAA); memory->create(AAA_L1, count_subAAA,
  // "FixConp/GA: AAA_level1 from gen_pseudo_invAAA_select");

  if (totalGA_num < size_map) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] allocate AAA_L1 matrix for direct I/O.\n");
    AAA_L1      = (double**)malloc(count_subAAA * sizeof(double*));
    AAA_L1_Flag = (bool*)malloc(count_subAAA * sizeof(bool));
    for (i = 0; i < count_subAAA; i++) {
      AAA_L1_Flag[i] = false;
    }
  } else {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] using STL::MAP for indirect I/O.\n");
  }

  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] 03 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  int** mat_struct = (int**)malloc(sizeof(int*) * localGA_num);
  int *row, *row_size = (int*)malloc(sizeof(int) * localGA_num);
  int i_nonzero, n, idx, irow, icol;

  for (i = AAA_disp; i < AAA_disp + localGA_num; ++i) {
    spMap_type* iAtom = spMat.atom(i);
    i_nonzero         = iAtom->size();

    i_index           = i - AAA_disp;
    row_size[i_index] = i_nonzero;
    if (max_neighbor < i_nonzero) max_neighbor = i_nonzero;
    row = mat_struct[i_index] = (int*)malloc(sizeof(int) * i_nonzero);
    row++;
    for (spMap_type::const_iterator iter = iAtom->begin(); iter != iAtom->end(); ++iter) {
      // printf("k = %d\n", k);
      if (iter->first != i) {
        row[0] = iter->first;
        row++;
      } else
        mat_struct[i_index][0] = i;
    }

    if (totalGA_num < size_map) {
      row = mat_struct[i_index];
      for (j = 0; j < i_nonzero; j++) {
        irow = row[j];
        for (k = 0; k < i_nonzero; k++) {
          icol    = row[k];
          idx_AAA = (size_t)(irow)*totalGA_num + icol;
          idx_L1  = idx_AAA / size_subAAA;
          idx_L2  = idx_AAA % size_subAAA;

          if (AAA_L1_Flag[idx_L1] == false) {
            AAA_L1_nonzero++;
            AAA_L1_Flag[idx_L1] = true;
            AAA_L1[idx_L1]      = (double*)malloc(size_subAAA * sizeof(double));
          }

          AAA_L1[idx_L1][idx_L2] = 0.0;
          // AAA0[idx_AAA] = 0.0;
        }
      }
    }
  } // i
  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] 03.1 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  // memset(AAA0, 0.0, sizeof(double)*totalGA_num*totalGA_num);

  if (totalGA_num < size_map) {
    for (i = 0; i < totalGA_num; ++i) {
      spMap_type* iAtom = spMat.atom(i);
      for (spMap_type::const_iterator iter = iAtom->begin(); iter != iAtom->end(); ++iter) {
        j       = iter->first;
        idx_AAA = (size_t)(i)*totalGA_num + j;
        idx_L1  = idx_AAA / size_subAAA;
        idx_L2  = idx_AAA % size_subAAA;
        if (AAA_L1_Flag[idx_L1] == false) {
          AAA_L1_nonzero++;
          AAA_L1_Flag[idx_L1] = true;
          AAA_L1[idx_L1]      = (double*)malloc(size_subAAA * sizeof(double));
        }
        AAA_L1[idx_L1][idx_L2] = iter->second;
        // AAA0[idx_AAA] = iter->second;
      }
    }
  }

  t_end = clock();
  if (me == 0) {
    utils::logmesg(lmp,
                   "[ConpGA] 04 Elps. Time {:.12f} for short-ranged interaction by "
                   "initiate a zero matrix\n",
                   double(t_end - t_begin) / CLOCKS_PER_SEC);
    utils::logmesg(lmp, "[ConpGA] subAAA_size = {}, Final sparse rate = {:.12f}\n", size_subAAA, (double)AAA_L1_nonzero / count_subAAA);
  }

  std::vector<double> DDD_vec;
  std::vector<double> coef_vec;
  std::vector<int> IPIV_vec;

  /*
  IPIV_vec.reserve(max_neighbor);
  std::vector<double> work_vec;
  int LWORK = max_neighbor * max_neighbor;
  work_vec.reserve(LWORK);
  double* WORK = &work_vec.front();
  int* IPIV = &IPIV_vec.front();
  char UPLO = 'L';
  int ione = 1;
  int INFO = 0;
  */
  /*
  DDD_vec.reserve(max_neighbor * max_neighbor);
  coef_vec.reserve(max_neighbor * 4);
  double * const DDD = &DDD_vec.front();
  double * const coef = &coef_vec.front();
  */
  double *DDD, *coef;
  memory->create(DDD, max_neighbor * max_neighbor, "fix_conp::PCG()::DDD");
  memory->create(coef, max_neighbor * 4, "fix_conp::PCG()::coef");
  map<int, int> tag2row_index;
  MAP_RESERVE(tag2row_index, max_neighbor);
  for (i = 0; i < localGA_num; i++) {
    t_end   = clock();
    i_index = i;
    n       = row_size[i_index];
    row     = mat_struct[i_index];
    idx     = 0;

    if (totalGA_num < size_map) {
      for (j = 0; j < n; j++) {
        irow = row[j];
        for (k = 0; k < n; k++) {
          icol = row[k];

          idx_AAA = (size_t)(irow)*totalGA_num + icol;
          idx_L1  = idx_AAA / size_subAAA;
          idx_L2  = idx_AAA % size_subAAA;

          DDD[idx] = // spMat.at(irow, icol);
              AAA_L1[idx_L1][idx_L2];
          // AAA0[irow * totalGA_num + icol];
          //
          idx++;
        }
      }
    } else {
      for (j = 0; j < n; j++) {
        irow = row[j];
        for (k = 0; k < n; k++) {
          icol     = row[k];
          idx_AAA  = (size_t)(irow)*totalGA_num + icol;
          idx_L1   = idx_AAA / size_subAAA;
          idx_L2   = idx_AAA % size_subAAA;
          DDD[idx] = spMat.at(irow, icol);
          idx++;
        }
      }
    }

    k = solver_CG(DDD, coef, n, pseuo_tolerance, maxiter);

    // dsysv_(&UPLO, &n, &ione, DDD, &n, IPIV, coef, &n, WORK, &LWORK, &INFO);
    /*
    if(i%20 == 0) {
      printf("DDD = ");
      for (j = 0; j < 10; j++) {
      irow = row[j];
      for (k = 0; k < 10; k++) {
        idx = n * j + k;
        printf("%f ", DDD[idx]);
        }
        printf("\n");
      }

      printf("f = ");
      for (j = 0; j < n; j++) {
        value = 0;
        for (k = 0; k < n; k++) {
          idx = j * n + k;
          value += DDD[idx] * coef[k];
        }
        printf("%f ", value);
      }
      printf("\n");
    }
    */
    if (i % 100 == 0 && me == 0) {
      utils::logmesg(lmp,
                     "atom {}/{}, Loops [{}/{}] DDD[0,0] = {:f}, DDD[0,(1:2)/{}] = "
                     "({:f},{:f}), Elps. Time {:.3f}[3]\n",
                     i, localGA_num, k, maxiter, DDD[0], n, DDD[1], DDD[2], double(t_end - t_begin) / CLOCKS_PER_SEC);
    }

    tag2row_index.clear();
    for (j = 0; j < n; j++) {
      jtag                = totalGA_tag_arr[row[j]];
      tag2row_index[jtag] = j;
    }

    // self to selt at D[0][0];
    firstneigh_inva_self[i] = coef[0];
    for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj++) {
      jseq = firstneigh_seq[jj];
      // jtag = tag[jseq];
      jtag       = jseq;
      jGA_index  = tag2GA_index[jtag];
      jrow_index = tag2row_index.count(jtag);
      if (jrow_index == 1) {
        jrow_index          = tag2row_index[jtag];
        firstneigh_inva[jj] = coef[jrow_index];
      } else {
        firstneigh_inva[jj] = 0;
#ifndef CONP_NO_DEBUG
        if (cut_coul_add >= 0) {
          snprintf(info, FIX_CONP_GA_WIDTH,
                   "Wrong structure at i:%d; jtag:%d from DC P-matrix of "
                   "update_charge_ppcg_i\n",
                   i, jtag);
          error->one(FLERR, info);
        }
#endif
      }

      firstneigh_localIndex[jj] = tag_To_localIndex[jtag];
    } // jj for localGA

    for (jj = localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
      jseq      = firstneigh_seq[jj];
      jtag      = jseq;
      jGA_index = tag2GA_index[jtag];

      jrow_index = tag2row_index.count(jtag);
      if (jrow_index == 1) {
        jrow_index          = tag2row_index[jtag];
        firstneigh_inva[jj] = coef[jrow_index];
      } else {
        firstneigh_inva[jj] = 0;
#ifndef CONP_NO_DEBUG
        if (cut_coul_add >= 0) {
          snprintf(info, FIX_CONP_GA_WIDTH, "Wrong structure at i:%d; jtag:%d from DC P-matrix of sparse_j\n", i, jtag);
          error->one(FLERR, info);
        }
#endif
      }

      firstneigh_localIndex[jj] = tag_To_localIndex[jtag];
    } // jj for reall ghostGA
  }   // i = 0 ~ localGA_num

  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] 05 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  if (totalGA_num < size_map) {
    for (i = 0; i < count_subAAA; i++) {
      if (AAA_L1_Flag[i] == true) {
        free(AAA_L1[i]);
      }
    }
    free(AAA_L1);
    free(AAA_L1_Flag);
  }

  memory->destroy(AAA);
  for (i = 0; i < localGA_num; i++) {
    free(mat_struct[i]);
  }
  free(mat_struct);
  free(row_size);
  free(DDD);
  free(coef);
}

void FixConpGA::gen_DC_invAAA()
{
  FUNC_MACRO(0)
  int i, iseq, itype, itag, jnum, jrow_index;
  int nlocal = atom->nlocal, nghost = atom->nghost, ntotal = nlocal + nghost;
  int jj, j, *jlist, jseq, jtag, jtype, jGA_index, *tag = atom->tag;
  int k, iproc, AAA_nonzero_local, AAA_nonzero, max_neighbor = 0, i_neighbor;
  int neighbor_size_local = pcg_neighbor_size_local; // lo_calGA_firstneigh.capacity(),
  int neighbor_size;
  int AAA_disp = localGA_EachProc_Dipl[me];
  int i_index, j_index, ii, *type = atom->type;
  double xtmp, ytmp, ztmp, qtmp, delx, dely, delz, rsq, value;
  // double ** AAA_tmp;
  double *q = atom->q, **x = atom->x;
  int inum           = list->inum;
  int* ilist         = list->ilist;
  int* numneigh      = list->numneigh;
  int** firstneigh   = list->firstneigh;
  double factor_coul = 1, factor_lj = -2, fforce = gewald_tune_factor;
  const double* const localGA_GreenCoef = &localGA_GreenCoef_vec.front();
  const double epsilon                  = std::numeric_limits<double>::epsilon();
  char info[512];

  localGA_firstneigh_inva.reserve(neighbor_size_local + localGA_num);
  localGA_firstneigh_localIndex.reserve(neighbor_size_local);
  // memory->create(AAA_tmp, totalGA_num, totalGA_num, "FixConp/GA: AAA_tmp");
  memory->create(AAA, totalGA_num, totalGA_num, "FixConp/GA: AAA from gen_DC_invAAA");

  double* const AAA0 = &AAA[0][0];
  for (i = 0; i < totalGA_num * totalGA_num; i++) {
    AAA0[i] = 0.0;
  }

  utils::logmesg(lmp, "[ConpGA] core: {}, neighbor_size_local = {} vs {}\n", me, neighbor_size_local, localGA_firstneigh.capacity());

  const int* const localGA_seq_arr = &localGA_seq.front();
  const int* const localGA_tag_arr = &localGA_tag.front();
  const int* const totalGA_tag_arr = &totalGA_tag.front();
  const int* const firstneigh_seq  = &localGA_firstneigh.front();

  double* const firstneigh_inva_self = &localGA_firstneigh_inva.front();
  double* const firstneigh_inva      = firstneigh_inva_self + localGA_num;
  int* const firstneigh_localIndex   = &localGA_firstneigh_localIndex.front();

  map<int, int> tag_To_localIndex;
  MAP_RESERVE(tag_To_localIndex, usedGA_num);
  for (i = 0; i < usedGA_num; i++) {
    tag_To_localIndex[localGA_tag_arr[i]] = i;
  }

  map<int, int> tag2GA_index;
  MAP_RESERVE(tag2GA_index, totalGA_num);
  for (i = 0; i < totalGA_num; i++) {
    tag2GA_index[totalGA_tag[i]] = i;
  }

  double coul = cut_coul + cut_coul_add, coulsq = coul * coul;

  double **AAA_me = AAA + AAA_disp, N_shift, N;
  N               = fit_erfc(gewald_tune_factor, coul, N_shift);
  if (me == 0) {
    utils::logmesg(lmp,
                   "[ConpGA] Generate DC P-matrix with Dampered_alpha = {:f}, compared "
                   "with PPPM g_ewald = {:f}\n",
                   gewald_tune_factor, pair_g_ewald);
    utils::logmesg(lmp,
                   "[ConpGA] Generate DC P-matrix using a Full matrix N = {:f} and "
                   "N_shift = {:f}\n",
                   N, N_shift);
  }

  t_begin = clock();
  double r_dist, qscale = force->qqrd2e / (N * N + 1);
  double inv_cut_coul_two = (N + 1) / coul, inv_cut_coulsq = 1 / coulsq;
  AAA_nonzero_local = 0;
  for (i = 0; i < localGA_num; ++i) {

    for (k = 0; k < totalGA_num; ++k) {
      ; // AAA_me[i][k] = 0;
    }

    iseq  = localGA_seq_arr[i];
    qtmp  = q[iseq] * 2; // To half the interaction energy;
    xtmp  = x[iseq][0];
    ytmp  = x[iseq][1];
    ztmp  = x[iseq][2];
    jlist = firstneigh[iseq];
    jnum  = numneigh[iseq];
    itype = type[iseq];
    // cl_atom[iseq] += qtmp * qtmp * localGA_GreenCoef[ii];

    for (jj = 0; jj < jnum; jj++) {
      jseq = jlist[jj];
      // factor_coul = special_coul[sbmask(j)];
      jseq &= NEIGHMASK;
      jtype = type[jseq];

      if (GA_Flags[jtype] == 0) {
        continue;
      }
      delx = xtmp - x[jseq][0];
      dely = ytmp - x[jseq][1];
      delz = ztmp - x[jseq][2];
      rsq  = delx * delx + dely * dely + delz * delz;
      if (rsq > coulsq) continue;
      AAA_nonzero_local++;
      jtag    = tag[jseq];
      j_index = tag2GA_index[jtag];

      // r_dist = sqrt(rsq);  value = (N + 0.27296801036977642)/r_dist +
      // r_dist*inv_cut_coulsq - inv_cut_coul_two;
      r_dist             = sqrt(rsq);
      value              = N_shift / r_dist + r_dist * inv_cut_coulsq - inv_cut_coul_two;
      AAA_me[i][j_index] = qscale * value;
      // value = force->pair->single(iseq, jseq, itype, jtype, rsq, factor_coul,
      // factor_lj, fforce); AAA_me[i][j_index] = value/(qtmp * q[jseq]);
    }                                                         // jj
    AAA_me[i][i + AAA_disp] = localGA_GreenCoef[i] + epsilon; // Self-interactions coeff
  }                                                           // i

  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] 01 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  AAA_nonzero_local += localGA_num;
  MPI_Allreduce(&AAA_nonzero_local, &AAA_nonzero, 1, MPI_INT, MPI_SUM, world);
  // printf("core:%d, AAA_nonzero_local:%d\n", me, AAA_nonzero_local);
  for (iproc = 0; iproc < nprocs; ++iproc) {
    AAA_disp = localGA_EachProc_Dipl[iproc];
    MPI_Bcast(AAA[AAA_disp], localGA_EachProc[iproc] * totalGA_num, MPI_DOUBLE, iproc, world);
    MPI_Barrier(world);
  }

  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] 02 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  /*
  for (k = 0; k < totalGA_num * totalGA_num; ++k) {
    if (AAA_ptr[k] != 0.0) {
      j = k % totalGA_num;
      i = (k - j) / totalGA_num;
      AAA_ptr[j*totalGA_num + i] = AAA_ptr[k];
    }
  }
  */
  double* AAA_ptr = AAA0;

  for (i = 0; i < totalGA_num; ++i) {
    for (j = 0; j < totalGA_num; ++j) {
      if (*AAA_ptr != 0.0) {
        AAA0[j * totalGA_num + i] = *AAA_ptr;
      }
      AAA_ptr++;
    }
  }

  /*
  for (i = 0; i < totalGA_num; ++i) {
    for (j = 0; j < i; ++j) {
      if(AAA[i][j] != 0) {
        AAA[j][i] = AAA[i][j];
      } else if (AAA[j][i] != 0) {
        AAA[i][j] = AAA[j][i];
      }
    }
  }
  */

  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] 03 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  int** mat_struct = (int**)malloc(sizeof(int*) * localGA_num);
  int *row, *row_size = (int*)malloc(sizeof(int) * localGA_num);
  int i_nonzero, n, idx, irow, icol;
  AAA_disp = localGA_EachProc_Dipl[me];
  for (i = AAA_disp; i < AAA_disp + localGA_num; ++i) {
    i_nonzero = 0;
    for (j = 0; j < totalGA_num; ++j) {
      if (AAA[i][j] != 0) {
        i_nonzero++;
      }
    }

    i_index           = i - AAA_disp;
    row_size[i_index] = i_nonzero;
    if (max_neighbor < i_nonzero) max_neighbor = i_nonzero;
    row = mat_struct[i_index] = (int*)malloc(sizeof(int) * i_nonzero);
    i_nonzero                 = 0;
    // i_atom row has been reshuffled to the head of array;
    for (j = i; j < totalGA_num; ++j) {
      if (AAA[i][j] != 0) {
        row[i_nonzero] = j;
        i_nonzero++;
      }
    }
    for (j = 0; j < i; ++j) {
      if (AAA[i][j] != 0) {
        row[i_nonzero] = j;
        i_nonzero++;
      }
    }
    /*
    for (j = 0; j < AAA_disp; ++j) {
      if(AAA_tmp[i][j] != 0) {
        row[i_nonzero] = j;
        i_nonzero++;
      }
    }
    */
  } // i

  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] 04 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  // neighbor_size = 2*AAA_nonzero/totalGA_num + max_neighbor;
  // neighbor_size *= 2;
  std::vector<double> DDD_vec;
  std::vector<double> coef_vec;
  DDD_vec.reserve(max_neighbor * max_neighbor);
  coef_vec.reserve(max_neighbor * 4);
  double* const DDD  = &DDD_vec.front();
  double* const coef = &coef_vec.front();

  map<int, int> tag2row_index;
  MAP_RESERVE(tag2row_index, max_neighbor);
  t_begin = clock();
  for (i = 0; i < localGA_num; i++) {
    t_end   = clock();
    i_index = i;
    n       = row_size[i_index];
    row     = mat_struct[i_index];
    idx     = 0;
    for (j = 0; j < n; j++) {
      irow = row[j];
      for (k = 0; k < n; k++) {
        icol     = row[k];
        DDD[idx] = AAA[irow][icol];
        idx++;
      }
    }

    k = solver_CG(DDD, coef, n, pseuo_tolerance, maxiter);
    /*
    if(i == 20) {
      printf("DDD = ");
      for (j = 0; j < 10; j++) {
      irow = row[j];
      for (k = 0; k < 10; k++) {
        idx = n * j + k;
        printf("%f ", DDD[idx]);
        }
        printf("\n");
      }

      printf("f = ");
      for (j = 0; j < n; j++) {
        value = 0;
        for (k = 0; k < n; k++) {
          idx = j * n + k;
          value += DDD[idx] * coef[k];
        }
        printf("%f ", value);
      }
      printf("\n");

    }

    if(i%100 == 0 && me == 0) {
      utils::logmesg(lmp, "atom %d/%d, Loops [%d/%d] DDD[0,0] = %f, DDD[0,(1:2)/%d]
    = (%f,%f), Elps. Time %.3f[3]\n", i,  localGA_num, k, maxiter, DDD[0], n,
    DDD[1], DDD[2], double(t_end - t_begin) / CLOCKS_PER_SEC);
    }
    */
    tag2row_index.clear();
    for (j = 0; j < n; j++) {
      jtag                = totalGA_tag_arr[row[j]];
      tag2row_index[jtag] = j;
    }

    // self to selt at D[0][0];
    firstneigh_inva_self[i] = coef[0];
    for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj++) {
      jseq = firstneigh_seq[jj];
      // jtag = tag[jseq];
      jtag      = jseq;
      jGA_index = tag2GA_index[jtag];

      jrow_index = tag2row_index.count(jtag);
      if (jrow_index == 1) {
        jrow_index          = tag2row_index[jtag];
        firstneigh_inva[jj] = coef[jrow_index];
      } else {
        firstneigh_inva[jj] = 0;
#ifndef CONP_NO_DEBUG
        if (cut_coul_add >= 0) {
          snprintf(info, FIX_CONP_GA_WIDTH, "Wrong structure at i:%d; jtag:%d from DC_invAAA_i\n", i, jtag);
          error->one(FLERR, info);
        }
#endif
      } //

      firstneigh_localIndex[jj] = tag_To_localIndex[jtag];
    } // jj for localGA

    for (jj = localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
      jseq       = firstneigh_seq[jj];
      jtag       = jseq;
      jGA_index  = tag2GA_index[jtag];
      jrow_index = tag2row_index.count(jtag);
      if (jrow_index == 1) {
        jrow_index          = tag2row_index[jtag];
        firstneigh_inva[jj] = coef[jrow_index];
      } else {
        firstneigh_inva[jj] = 0;
#ifndef CONP_NO_DEBUG
        if (cut_coul_add >= 0) {
          snprintf(info, FIX_CONP_GA_WIDTH, "Wrong structure at i:%d; jtag:%d from DC_invAAA_j\n", i, jtag);
          error->one(FLERR, info);
        }
#endif
      } //

      firstneigh_localIndex[jj] = tag_To_localIndex[jtag];
    } // jj for reall ghostGA
  }   // i = 0 ~ localGA_num
  memory->destroy(AAA);
  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] 05 Elps. Time {:.12f} for short-ranged interaction\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  for (i = 0; i < localGA_num; i++) {
    free(mat_struct[i]);
  }
  free(mat_struct);
  free(row_size);
}

double FixConpGA::fit_erfc(double alpha, double rcut, double& shift_by_erfc)
{
  double result0 = MathSpecial::erfcx_y100(400.0 / (4.0 + alpha)) * MathSpecial::expmsq(alpha);
  double result1 = MathSpecial::erfcx_y100(400.0 / (4.0 + alpha * rcut)) * MathSpecial::expmsq(alpha * rcut);
  double inv_Rc  = 1 / rcut;
  double a       = result0 - result1;
  double b       = 2 * (inv_Rc - 1);
  double c       = a - inv_Rc * b;
  double t       = sqrt(b * b - 4 * a * c);
  double root0, root1;
  // root0 = (-b - t)/(2 * a);
  root1         = (-b + t) / (2 * a);
  shift_by_erfc = result1 * (root1 * root1 + 1) * 0.5 + root1;
  // printf("%f, %f, %f, %f, %f, %f \n", root0, root1, end_erfc, result0, alpha,
  // rcut);

  return root1;
}

int FixConpGA::solver_CG(double* M, double* x, int n, double eps, int maxiter)
{
  FUNC_MACRO(2)
  int i, j, k, idx, iter;
  double alpha, beta, value, ptap, resnorm = 0, newresnorm;
  double* const res = x + n;
  double* const p   = res + n;
  double* const ap  = p + n;
  /*
  printf("M = \n");
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      idx = i * n + j;
      printf(" %8.3g", M[idx]);
    }
    printf("\n");
  }
  */

  // initial x_vec = [1, 1, 1, ..., 1];
  /*
  for (i = 0; i < n; i++) {
    x[i] = 1/M[i];
  }


  resnorm = 0;
  for (i = 0; i < n; i++) {
    value = 0;
    for (j = 0; j < n; j++) {
      idx = i * n + j;
      value -= M[idx]*x[j];
    }
    p[i] = res[i] = value;
    resnorm *= value*value;
  }
  */

  for (i = 0; i < n; i++) {
    x[i] = res[i] = p[i] = 0;
  }
  resnorm = 1;
  res[0]  = 1; // For b = [1,0,0,0,0];
  p[0]    = 1;

  /* // using the previous result as input no acceleration.
  for (i = 0; i < n; ++i)
  {
    value = 0;
    for (j = 0; j < n; j++) {
      idx = i * n + j;
      value += M[idx] * x[j];
    }
    p[i] = res[i] = -value;
    resnorm += value*value;
  }
  value = p[0] + 1;
  resnorm += value * value - p[0] * p[0];
  res[0] = p[0] = value;
  */

  for (iter = 0; iter < maxiter; iter++) {

    ptap = 0;
    for (i = 0; i < n; i++) {
      value = 0;
      for (j = 0; j < n; j++) {
        idx = i * n + j;
        value += M[idx] * p[j];
      }
      ap[i] = value;
      ptap += value * p[i];
    }

    alpha = resnorm / ptap;

    newresnorm = 0;
    for (i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      res[i] -= alpha * ap[i];
      newresnorm += res[i] * res[i];
    }

    if (fabs(newresnorm) < eps) {
      return iter;
    }

    beta = newresnorm / resnorm;
    for (i = 0; i < n; i++) {
      p[i] = res[i] + beta * p[i];
    }
    resnorm = newresnorm;
  } // iter
  return iter;
}

static inline double cal_1D_std(int n_split, const double* const p0, const double* const p1)
{
  int i, n = p1 - p0, n_right = n - n_split;
  const double* ptr = p0;
  double ave_L = 0, ave_R = 0, ave = 0, sqr_L = 0, sqr_R = 0, tmp;
  for (ptr = p0; ptr < p0 + n_split; ++ptr) {
    tmp = *ptr;
    ave_L += tmp;
    sqr_L += tmp * tmp;
  }
  for (ptr = p0 + n_split; ptr < p1; ++ptr) {
    tmp = *ptr;
    ave_R += tmp;
    sqr_R += tmp * tmp;
  }
  ave_L /= n_split;
  ave_R /= n_right;
  return (sqr_R * n_right - ave_R * ave_R) + (sqr_L * n_split - ave_L * ave_L);
}

int FixConpGA::optimal_split_1D(const double* const p0, const double* const p1)
{
  int i, n = p1 - p0;
  const double* ptr = p0;
  double error[3];
  int split[3] = {n / 10, n / 4, n - 1}, new_split, iter = 0;
  ;
  error[0]  = cal_1D_std(split[0], p0, p1);
  error[1]  = cal_1D_std(split[1], p0, p1);
  error[2]  = cal_1D_std(split[2], p0, p1);
  new_split = quadratic_fit_int(split, error);

  // printf("%f %f %f\n", error[0], error[1], error[2]);

  // printf("%d + %d %d %d\n", new_split, split[0], split[1], split[2]);

  if (new_split > split[1]) {
    split[2] = split[1];
    error[2] = error[1];
  } else {
    split[0] = split[1];
    error[0] = error[1];
  }
  split[1] = new_split;

  while (split[0] != split[2]) {
    error[1] = cal_1D_std(split[1], p0, p1);
    // printf("while %d %d %d\n", split[0], split[1], split[2]);
    new_split = quadratic_fit_int(split, error);
    if (new_split > split[1]) {
      split[2] = split[1];
      error[2] = error[1];
    } else {
      split[0] = split[1];
      error[0] = error[1];
    }
    split[1] = new_split;
    iter++;
  }
  // printf("iter = %d, new_split = %d\n", iter, new_split);
  return new_split;
}

void FixConpGA::sparsify_invAAA_EX()
{
  FUNC_MACRO(0)
  int i, j, iseq, itag, nlocal = atom->nlocal, nghost = atom->nghost, ntotal = nlocal + nghost;
  int jj, jseq, jtag, jGA_index, *tag = atom->tag, i_nbr, local_nbr_sum = 0, total_nbr_sum = 0;
  int neighbor_size_local = localGA_firstneigh.capacity(), neighbor_size;
  int AAA_disp            = localGA_EachProc_Dipl[me];

  double* split;
  memory->create(split, bisect_num + 10, "FixConpGA:split");
  memory->create(block_ave_value, bisect_num + 10, "FixConpGA:i_split");

  double **AAA_Global = AAA, *ptr_AAA, value;
  int totalGA_num_sqr = totalGA_num * totalGA_num, iAAA_start, iBlock;

  const int* const size_EachProc = &localGA_EachProc.front();
  const int* const size_Dipl     = &localGA_EachProc_Dipl.front();

  MPI_Datatype MPI_totalGA;
  MPI_Type_contiguous(totalGA_num, MPI_DOUBLE, &MPI_totalGA);
  MPI_Type_commit(&MPI_totalGA);

  if (bisect_level > 0) {
    if (me == 0) memory->create(AAA_Global, totalGA_num, totalGA_num, "FixConpGA:AAA_Global");
    MPI_Gatherv(AAA[0], localGA_num, MPI_totalGA, AAA_Global[0], size_EachProc, size_Dipl, MPI_totalGA, 0, world);
  }

  if (bisect_level > 0 and me == 0) {
    double* ptr;
    int* i_split;
    memory->create(i_split, bisect_num + 1, "FixConpGA:i_split");

    ptr_AAA = AAA_Global[0];
    std::sort(ptr_AAA, ptr_AAA + totalGA_num_sqr, fabs_dcomp);
    for (i = 0; i < totalGA_num_sqr; ++i) {
      if (fabs(ptr_AAA[i]) < ppcg_invAAA_cut) break;
    }
    iAAA_start = i;

    std::sort(ptr_AAA + iAAA_start, ptr_AAA + totalGA_num_sqr, fabs_dcomp);

    i_split[0] = iAAA_start;
    i_split[1] = totalGA_num_sqr;
    for (int ilevel = 0; ilevel < bisect_level; ++ilevel) {

      int num_level = 1 << ilevel;
      for (i = num_level; i > 0; --i) {
        j          = 2 * i;
        i_split[j] = i_split[i];
      }

      num_level *= 2;
      for (i = 1; i < num_level; i += 2) {
        i_split[i] = optimal_split_1D(ptr_AAA + i_split[i - 1], ptr_AAA + i_split[i + 1]);
        i_split[i] += i_split[i - 1];
      }
    }

    split[0] = ppcg_invAAA_cut;
    for (i = 1; i < bisect_num + 1; ++i) {
      split[i - 1]           = ptr_AAA[i_split[i] - 1];
      block_ave_value[i - 1] = 0;
      for (ptr = ptr_AAA + i_split[i - 1]; ptr < ptr_AAA + i_split[i]; ++ptr) {
        block_ave_value[i - 1] += *ptr;
      }
      block_ave_value[i - 1] /= i_split[i] - i_split[i - 1];

      if (DEBUG_LOG_LEVEL > 0) {
        value = cal_1D_std(1, ptr_AAA + i_split[i - 1], ptr_AAA + i_split[i]);
        if (me == 0)
          utils::logmesg(lmp,
                         "block[{}], range: {}-{}[{}], value = {:.12g}, ave = {:.12g}, "
                         "sqr = {:.12g}\n",
                         i, i_split[i - 1], i_split[i], i_split[i] - i_split[i - 1], split[i - 1], block_ave_value[i - 1], value);
      }
      block_ave_value[i - 1] *= pcg_shift_scale;
    }
    memory->destroy(AAA_Global);
    memory->destroy(i_split);
    AAA_Global = AAA;
  } // (bisect_level > 0 and me == 0)

  MPI_Bcast(split, bisect_num, MPI_DOUBLE, 0, world);
  MPI_Bcast(block_ave_value, bisect_num, MPI_DOUBLE, 0, world);

  // int ilevel, num_level;

  // P_PPCG = (double**) malloc(localGA_num * sizeof(double*));
  // ppcg_GAindex = (int**) malloc(localGA_num * sizeof(int*));
  memory->create(P_PPCG, localGA_num, totalGA_num, "FixConpGA:P_PPCG");
  memory->create(ppcg_GAindex, localGA_num, totalGA_num, "FixConpGA:ppcg_GAindex");

  // int* ppcg_neighbor = (int*) malloc(localGA_num * sizeof(int));
  if (bisect_level == 0) {
    memory->create(ppcg_block, localGA_num, 2, "FixConpGA:ppcg_neighbor");
  } else {
    memory->create(ppcg_block, localGA_num, bisect_num + 1, "FixConpGA:ppcg_neighbor");
  }

  std::vector<std::pair<int, double>> data(totalGA_num, std::pair<int, double>(0, 0.0));

  int ishow = 10;

  double local_shift = 0, total_shift = 0;
  for (i = 0; i < localGA_num; ++i) {
    for (j = 0; j < totalGA_num; ++j) {
      data[j].first  = j;
      data[j].second = AAA[i][j];
      if (fabs(AAA[i][j]) < ppcg_invAAA_cut) {
      }
    }

    std::sort(data.begin(), data.end(), fabs_comp);
    for (j = 0; j < totalGA_num; ++j) {
      if (fabs(data[j].second) < ppcg_invAAA_cut) break;
    }

    ppcg_block[i][0] = i_nbr = j;
    local_nbr_sum += i_nbr;
    std::sort(data.begin(), data.begin() + i_nbr, int_GT);

    std::sort(data.begin() + i_nbr, data.end(), val_comp);

    // for bisect_level > 0, iBlock start with 0;
    for (iBlock = (bisect_level == 0); iBlock < bisect_num; ++iBlock) {
      for (j = ppcg_block[i][iBlock]; j < totalGA_num; ++j) {
        local_shift += data[j].second;
        if (data[j].second >= split[iBlock]) break;
      }
      ppcg_block[i][iBlock + 1] = j;
      std::sort(data.begin() + ppcg_block[i][iBlock], data.begin() + j, int_GT);
    }
    /*
    for (j = i_nbr; j < totalGA_num; ++j)
    {
      local_shift += data[j].second;
    }
    */
    for (j = 0; j < totalGA_num; ++j) {
      ppcg_GAindex[i][j] = data[j].first;
      P_PPCG[i][j]       = data[j].second;
    }
  }

  MPI_Allreduce(&local_shift, &total_shift, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&local_nbr_sum, &total_nbr_sum, 1, MPI_INT, MPI_SUM, world);

  pcg_shift = total_shift / (double(totalGA_num) * totalGA_num - total_nbr_sum);

  if (me == 0) {
    value = double(total_nbr_sum) / totalGA_num;
    utils::logmesg(lmp, "[ConpGA] Ave. non-zero per column: {:.3f}/{} [{:.3f}%%]\n", value, totalGA_num, value * 100 / totalGA_num);
    utils::logmesg(lmp, "[ConpGA] The shift for sparse P: {:.16g} (x{:.f})\n", pcg_shift, pcg_shift_scale);
  }
  memory->destroy(split);

  pcg_shift *= pcg_shift_scale;
  if (bisect_level == 0) {
    for (i = 0; i < localGA_num; ++i) {
      for (j = 0; j < ppcg_block[i][0]; ++j) {
        P_PPCG[i][j] -= pcg_shift;
      }
    }
  }
}

/*
void FixConpGA::check_ppcg_partition(int index, int** range) {
  int i, j;
  double ave_local = 0, ave_total;
  double sqr_local = 0, sqr_total, tmp;
  int num_local = 0, num_total;

  for (i = 0; i < localGA_num; ++i) {
    for (j = range[index][i]; j < range[index+1][i]; ++j) {
      ave_local += P_PPCG[i][j];
      num_local++;
    }
  }


  MPI_Allreduce(&ave_local, &ave_total, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&num_local, &num_total, 1, MPI_INT,    MPI_SUM, world);
  ave_total /= num_total;

  for (i = 0; i < localGA_num; ++i) {
    for (j = range[index][i]; j < range[index+1][i]; ++j) {
      tmp = ave_total - P_PPCG[i][j];
      sqr_local += tmp * tmp;
    }
  }
  MPI_Allreduce(&sqr_local, &sqr_total, 1, MPI_DOUBLE, MPI_SUM, world);
  printf("total_ave = %.16g, num = %d, sqr_local = %.16g\n", ave_total,
num_total, sqrt(sqr_total/num_total));


}
*/

double FixConpGA::quadratic_fit_int(int* x_in, double* y)
{
  double x[3] = {double(x_in[0]), double(x_in[1]), double(x_in[2])};
  return quadratic_fit(x, y);
}

double FixConpGA::quadratic_fit(double* x, double* y)
{
  double xMat[9], value;
  double yMat[3] = {y[0], y[1], y[2]};

  if (fabs((x[1] - x[2]) / x[2]) < 1e-8) return (x[0] + x[2]) / 2;
  if (fabs((x[0] - x[1]) / x[2]) < 1e-8) return (x[0] + x[2]) / 2;

  xMat[0] = x[0] * x[0];
  xMat[1] = x[1] * x[1];
  xMat[2] = x[2] * x[2];

  xMat[3] = x[0];
  xMat[4] = x[1];
  xMat[5] = x[2];
  xMat[6] = xMat[7] = xMat[8] = 1;

  std::vector<double> work_vec;
  std::vector<MKL_INT> IPIV_vec;

  int LWORK = 9;
  MKL_INT n = 3;
  work_vec.reserve(LWORK);
  IPIV_vec.reserve(LWORK);
  double* WORK  = &work_vec.front();
  MKL_INT* IPIV = &IPIV_vec.front();
  MKL_INT ione  = 1;
  MKL_INT INFO  = 0;

  dgesv_(&n, &ione, xMat, &n, IPIV, yMat, &n, &INFO);
  if (INFO != 0) {
    if (me == 0) utils::logmesg(lmp, "x = {:.18g} {:.18g} {:.18g}\n", x[0], x[1], x[2]);
    if (me == 0) utils::logmesg(lmp, "y = {:.18g} {:.18g} {:.18g}\n", y[0], y[1], y[2]);
    error->all(FLERR, "quadratic_fit() cannot solve the 3x3 matrix");
  }
  // printf("y = %g %g %g\n, INFO = %d\n", y[0], y[1], y[2], INFO);
  // exit(0);
  // value = -b/2a
  value = -yMat[1] / (2 * yMat[0]);
  // printf("value = %.16g\n", value);
  if (value < x[0]) {
    value = x[0];
  } else if (value > x[2]) {
    value = x[2];
  }
  // printf("value = %.16g\n", value);
  return value;
}

/* get the optimal position of the P matrix split */
/*
double FixConpGA::ppcg_best_split(int i, int**range, int* num_arr, double*
split_arr) { FUNC_MACRO(1); printf("start at ppcg_best_split at %d\n", i); int*
range_L = range[i]; int* range_M = range[i+1]; int* range_R = range[i+2]; double
split[3]; // = {split_arr[i], split_arr[i+1], split_arr[i+2]}; double error[3];
  int    n_num[3] = {0,     -1,   num_arr[i+2] - num_arr[i]};
  double new_split;
  int n_left_R;
  split[1] = (split_arr[i+0] + split_arr[i+2]) / 2;

  split[0] = (split_arr[0] + split[1]) / 2;
  split[0] = (split[0] + split_arr[0]) / 2;

  split[2] = (split_arr[2] + split[1]) / 2;
  split[2] = (split[2] + split_arr[2]) / 2;

  int n_left, n_right;
  int iter = 0;

  printf("%g %g %g %g %g\n", split_arr[i+0], split[0], split[1], split[2],
split_arr[i+2]);

  split[2] = split_arr[i+2];
  error[2] = ppcg_split_STD(split[2], n_left_R, n_right, range_L, range_M,
range_R); printf("n_Right = %d\n", n_left_R); if (n_left_R == 0) { exit(0);
  }

  n_left = 0; split[0] = split_arr[i+0];
  while (n_left == 0 || n_left == n_left_R) {
    while (n_left == 0) {
      split[0] = (split[0] + split[2]) / 2;
      error[0] = ppcg_split_STD(split[0], n_left, n_right, range_L, range_M,
range_R);
    }
    if (n_left == n_left_R) {
      split[2] = split[0];
      split[0] = split_arr[i+0];
      n_left = 0;
    }
  }
  printf("n_left = %d\n", n_left);


  split[1] = split[0]*0.66 + split[2]*0.34;

  error[1] = ppcg_split_STD(split[1], n_left, n_right, range_L, range_M,
range_R);

  printf("split[1] = %f, %f, n_left = %d\n", split[1], split[2], n_left);

  new_split = quadratic_fit(split, error);

  printf("new_split = %12g  %12g  %12g  %12g\n", new_split, split[0], split[1],
split[2]); printf("error     = %12g  %12g  %12g\n", error[0],  error[1],
error[2]);

  if (new_split < split[1]) {
    split[2] = split[1];
    error[2] = error[1];
    n_num[2] = n_left;

    split[0] = split_arr[i+0];
    error[0] = ppcg_split_STD(split[0], n_left, n_right, range_L, range_M,
range_R); n_num[0] = n_left;

  } else {
    split[0] = split[1];
    error[0] = error[1];
    n_num[0] = n_left;


    split[2] = split_arr[i+2];
    error[2] = ppcg_split_STD(split[2], n_left, n_right, range_L, range_M,
range_R); n_num[2] = n_left;
  }

  split[1] = (split[0] + split[1]) / 2;


  printf("start_plit = %12g  %12g  %12g  %12g\n", new_split, split[0], split[1],
split[2]); printf("error     = %12g  %12g  %12g\n", error[0],  error[1],
error[2]); printf("n_num = %d, %d\n", n_num[0], n_num[2]); while(n_num[0] !=
n_num[2] && iter < 100) {
    // printf("split = %.12g %.12g %.12g\n", split[0], split[1], split[2]);
    // printf("num   = %d %d %d\n", n_num[0], n_num[1], n_num[2]);
    n_left = 0;
    while(n_left == 0) {
      error[1] = ppcg_split_STD(split[1], n_left, n_right, range_L, range_M,
range_R); split[1] = (split[1] + split[2])/2;
    }
    new_split = quadratic_fit(split, error);
    printf("new_split = %12g  %12g  %12g  %12g\n", new_split, split[0],
split[1], split[2]); printf("error     = %12g  %12g  %12g [%d]\n", error[0],
error[1], error[2], n_left);

    if (new_split < split[1]) {
      error[2] = error[1];
      split[2] = split[1];
      n_num[2] = n_left;

      split[1] = new_split;
    } else {
      error[0] = error[1];
      split[0] = split[1];
      n_num[0] = n_left;

      split[1] = new_split;
    }

    split[1] = (split[0] + split[1]) / 2;

    iter ++;
  }


  printf("split = %.12g %.12g %.12g\n", split[0], split[1], split[2]);
  printf("iter = %d, num   = %d %d %d\n", iter, n_num[0], n_num[1], n_num[2]);
  split_arr[i+1] = split[1];
  num_arr[i+1] = n_left + num_arr[i];
  return split[1];
}
*/

/* get the position of the P matrix split */
/*
double FixConpGA::ppcg_split_STD(double split, int& ret_left_n, int&
ret_right_n, const int* range_L, int* range_M, const int* range_R) { int i, j;
  double left_val = 0, right_val = 0;
  double left_val_total, right_val_total;
  double left_std, right_std, tmp;
  int left_n = 0, right_n = 0;
  int left_n_total = 0, right_n_total = 0;

  for (i = 0; i < localGA_num; ++i) {
    for (j = range_L[i]; j < range_R[i]; ++j) {
      // if (i == 0) printf("L %d %f %f\n", j, P_PPCG[i][j], split);
      if (P_PPCG[i][j] <= split) {
        left_val += P_PPCG[i][j];
        left_n   ++;
      } else {
        // printf("j = %d, %f, %f\n", j, P_PPCG[i][j], split);
        break;
      }
    }
    range_M[i] = j;
    for (j = range_M[i]; j < range_R[i]; ++j) {
      // if (i == 0) printf("R %d %f %f\n", j, P_PPCG[i][j], split);
      right_val += P_PPCG[i][j];
      right_n   ++;
    }
  } // i;
  MPI_Allreduce(&left_val,  &left_val_total,  1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&right_val, &right_val_total, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&left_n,    &left_n_total,    1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&right_n,   &right_n_total,   1, MPI_INT, MPI_SUM, world);
  ret_left_n = left_n_total;
  ret_right_n = right_n_total;


  // printf("%f %f %d %d\n", left_val_total, right_val_total, left_n_total,
right_n_total);

  left_val_total   /= left_n_total;
  right_val_total  /= right_n_total;
  if (left_n_total == 0) left_val_total = 0;
  if (right_n_total == 0) right_val_total = 0;

  left_std = right_std = 0.0;
  for (i = 0; i < localGA_num; ++i) {
    for (j = range_L[i]; j < range_M[i]; ++j) {
      tmp = P_PPCG[i][j] - left_val_total;
      left_std += tmp * tmp;
    }

    for (j = range_M[i]; j < range_R[i]; ++j) {
      tmp = P_PPCG[i][j] - right_val_total;
      right_std += tmp * tmp;
    }
  }

  MPI_Allreduce(&left_std,  &left_val,  1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&right_std, &right_val, 1, MPI_DOUBLE, MPI_SUM, world);

  return sqrt((left_val + right_val) / (left_n_total + right_n_total));
}
*/

void FixConpGA::sparsify_invAAA()
{
  FUNC_MACRO(0)
  int i, iseq, itag, nlocal = atom->nlocal, nghost = atom->nghost, ntotal = nlocal + nghost;
  int jj, jseq, jtag, jGA_index, *tag = atom->tag;
  int neighbor_size_local = localGA_firstneigh.capacity(), neighbor_size;
  int AAA_disp            = localGA_EachProc_Dipl[me];
  localGA_firstneigh_inva.reserve(neighbor_size_local + localGA_num);
  localGA_firstneigh_localIndex.reserve(neighbor_size_local);

  double value;

  double* const firstneigh_inva_self = &localGA_firstneigh_inva.front();
  double* const firstneigh_inva      = firstneigh_inva_self + localGA_num;
  int* const firstneigh_localIndex   = &localGA_firstneigh_localIndex.front();
  // const int * const localGA_seq_arr = &localGA_seq.front();
  const int* const firstneigh_seq  = &localGA_firstneigh.front();
  const int* const localGA_seq_arr = &localGA_seq.front();

  // std::vector<int> seq_To_localIndex;
  // seq_To_localIndex.assign(ntotal, -1)
  // const int * const seq_To_localIndex_arr = &seq_To_localIndex.front();

  map<int, int> tag_To_localIndex;
  MAP_RESERVE(tag_To_localIndex, usedGA_num);
  for (i = 0; i < usedGA_num; i++) {
    iseq                         = localGA_seq_arr[i];
    tag_To_localIndex[tag[iseq]] = i;
  }

  map<int, int> tag2GA_index;
  MAP_RESERVE(tag2GA_index, totalGA_num);
  for (i = 0; i < totalGA_num; i++) {
    tag2GA_index[totalGA_tag[i]] = i;
  }

  double** AAA_tmp = AAA; // + AAA_disp;
  for (i = 0; i < localGA_num; i++) {
    // iseq = localGA_seq_arr[i];
    firstneigh_inva_self[i] = AAA_tmp[i][i + AAA_disp];
    for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj++) {
      jseq = firstneigh_seq[jj];
      jtag = jseq;
      // jtag = tag[jseq];
      jGA_index = tag2GA_index[jtag];
#ifndef CONP_NO_DEBUG
      if (jGA_index == -1 || tag_To_localIndex[jtag] == -1) error->all(FLERR, "wrong jGA_index");
#endif
      firstneigh_inva[jj]       = AAA_tmp[i][jGA_index];
      firstneigh_localIndex[jj] = tag_To_localIndex[jtag];

    } // jj -> local atom
    for (jj = localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
      jseq                      = firstneigh_seq[jj];
      jtag                      = jseq;
      jGA_index                 = tag2GA_index[jtag];
      firstneigh_inva[jj]       = AAA_tmp[i][jGA_index];
      firstneigh_localIndex[jj] = tag_To_localIndex[jtag];
#ifndef CONP_NO_DEBUG
      if (jGA_index == -1 || tag_To_localIndex[jtag] == -1) error->all(FLERR, "wrong jGA_index");
#endif
    } // jj -> ghost atom
  }
  if (newton_pair) neighbor_size_local *= 2;
  MPI_Allreduce(&neighbor_size_local, &neighbor_size, 1, MPI_INT, MPI_SUM, world);

  neighbor_size += totalGA_num;

  double average_num = double(neighbor_size) / totalGA_num;
  if (me == 0)
    utils::logmesg(lmp,
                   "[ConpGA]  CPU[{}]:Total size: {}, Ave. neighbor number: "
                   "{:10.7f}/{}, Sparsity:{:3.2f}%%\n",
                   me, neighbor_size, average_num, totalGA_num, average_num * 100 / double(totalGA_num));

  if (me == work_me && DEBUG_LOG_LEVEL > 1) {
    for (i = 0; i < localGA_num; i++) {
      iseq = localGA_seq[i];
      itag = tag[iseq];
      printf("------------ GA atom localIndex: %d, Seq: %d, Tag; %d "
             "------------------\n",
             i, iseq, itag);
      printf("jseq [%d]: ", localGA_numlocalneigh[i]);
      for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
        jseq = firstneigh_seq[jj];
        printf("%d ", jseq);
      }
      printf("\n");
      printf("jtag [%d]: ", localGA_numlocalneigh[i]);
      for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {

        jseq = firstneigh_seq[jj];
        jtag = tag[jseq];
        printf("%d ", jtag);
      }
      printf("\n");
      printf("j_localindex: ");
      for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
        jseq = firstneigh_localIndex[jj];
        printf("%d ", jseq);
      }
      printf("\n");
      printf("invA_ij[%f]: ", firstneigh_inva_self[i]);
      for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
        value = firstneigh_inva[jj];
        printf("%f ", value);
      }
      printf("\n\n");
    }

  } // work_me

  // printf("jj = %d, firstneigh_inva size = %d\n", jj,
  // localGA_firstneigh_inva.capacity());
}

void FixConpGA::gen_invAAA()
{
  FUNC_MACRO(0)
  if (AAA_save_format < 10) {
    if (force->kspace) // Extra Kspace Check
      setup_AAA();
    else
      setup_AAA_Direct();
    if (AAA_save_format < 0) return;
    if (AAA_save_format % 2)
      save_AAA(); // AAA_save_format = 1 for binary file
    else
      savetxt_AAA(); // AAA_save_format = 0 for txt
  } else {
    // _self_calc();
    if (AAA_save_format % 2)
      load_AAA(); // =1 Load AAA from a binary matrix file
    else
      loadtxt_AAA(); // =0 Load AAA from a txt matrix file
  }
}

/* ----------------------------------------------------------------------------
      Calculate the self-interaction coefficient Xi of simulation cell
---------------------------------------------------------------------------- */
void FixConpGA::uself_calc()
{
  FUNC_MACRO(0);
  if (force->kspace == nullptr) { // Extra Kspace Check
    Xi_self = 0;
    return;
  }
  int i, ilocalseq, iseq, nlocal = atom->nlocal;
  int iproc, procs = comm->nprocs;
  double *q                        = atom->q, *q_store_tmp;
  const int* const localGA_seq_arr = &localGA_seq.front();

  memory->create(q_store_tmp, nlocal, "FixConpGA:q_store_tmp");

  // set all charge to zero;
  for (size_t i = 0; i < nlocal; i++) {
    q_store_tmp[i] = q[i];
    q[i]           = 0;
  }

  for (iproc = 0; iproc < procs; iproc++) {
    if (localGA_EachProc_Dipl[iproc] <= Xi_self_index && Xi_self_index < localGA_EachProc_Dipl[iproc] + localGA_EachProc[iproc]) break;
  }
  if (me == 0)
    utils::logmesg(lmp, "[ConpGA] Xi_self_index = {}/{} @ Proc. #{}: [{}<-{}]\n", Xi_self_index, totalGA_num, iproc, localGA_EachProc_Dipl[iproc],
                   localGA_EachProc_Dipl[iproc] + localGA_EachProc[iproc] - 1);

  if (me == iproc) {
    ilocalseq    = localGA_seq[Xi_self_index - localGA_EachProc_Dipl[iproc]];
    q[ilocalseq] = 1.0;
  }

  // electro->qsum_qsq();
  // electro->compute(3, 0);
  electro->compute_GAtoGA();
  if (me == iproc) {
    Xi_self      = -electro->cl_atom()[ilocalseq];
    q[ilocalseq] = 0;
  }
  MPI_Bcast(&Xi_self, 1, MPI_DOUBLE, iproc, world);
  MPI_Barrier(world);
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Ewald Xi_self = {:.18g} [Energ/Charge^2]\n", Xi_self);

  // restore the charge on GA electrode
  for (i = 0; i < localGA_num; i++) {
    iseq    = localGA_seq_arr[i];
    q[iseq] = q_store_tmp[iseq];
  }

  // self_Xi_flag = true;
  // self_Xi_flag = false;

  std::copy(q_store_tmp, q_store_tmp + nlocal, q);

  /*
  for (size_t i = 0; i < nlocal; i++)
  {
    printf("%d, %f %f\n", i, q[i], q_store_tmp[i]);
  }
  */
}
/* ------------------------------------------------------------------------------
  Generate the interaction AAA matrix between electrode atoms directly
------------------------------------------------------------------------------
*/
void FixConpGA::setup_AAA_Direct()
{
  FUNC_MACRO(0)
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Build AAA matrix from the direct pairwise interaction.\n");
  int *tag = atom->tag, *type = atom->type;
  // int nlocal = atom->nlocal;

  int i, j, jj, iseq, jseq, jtag, itype, jtype, jGA_index;
  int iproc, AAA_disp = localGA_EachProc_Dipl[me];
  int totalneigh_count = localGA_num; // Inital with a list N(localGA_num) self
                                      // to self interaction;
  int localGAPair_num   = localGA_firstneigh.capacity(), totalGAPair_num;
  int localGA_index_min = localGA_EachProc_Dipl[me] - 1, localGA_index_max = localGA_EachProc_Dipl[me] + localGA_num;
  double factor_coul = 1, factor_lj = -1, fforce;
  char info[512];
  map<int, int> tag2GA_index;
  MAP_RESERVE(tag2GA_index, totalGA_num);

  const double* const localGA_GreenCoef = &localGA_GreenCoef_vec.front();
  const double* const firstneigh_coeff  = &localGA_firstneigh_coeff.front();
  const int* const firstneigh_seq       = &localGA_firstneigh.front();

  // MPI_Allreduce(&localGAPair_num, &totalGAPair_num, 1, MPI_INT, MPI_SUM,
  // world); printf("ME:%d, %d %d\n", me, localGA_index_min, localGA_index_max);
  if (localGAPair_num < ((localGA_num - 2) * totalGA_num) / 2) {
    sprintf(info,
            "[ConpGA] setup_AAA_Direct: CPU[%d]: localGA_num %d, "
            "localGAPair_num %d, expect %d. To increase pair cutoff!\n",
            me, localGA_num, localGAPair_num, localGA_num * totalGA_num);
    error->all(FLERR, info);
  }

  memory->create(AAA, totalGA_num, totalGA_num, "FixConp/GA: AAA_tmp");

  double** AAA_tmp = AAA; // + AAA_disp;
  // printf("me %d, AAA_disp = %d\n", me, AAA_disp);
  for (i = 0; i < totalGA_num; i++) {
    tag2GA_index[totalGA_tag[i]] = i;
  }

  int* const localGA_seq_arr = &localGA_seq.front();
  for (i = 0; i < localGA_num; i++) {
    // itype = type[localGA_seq_arr[i]];
    // i<->i Self-interaction; For point charge is zero, For Gaussian charge is
    // 2/sqrt(PI) * eta Where eta is defined as Phi = erfc(eta * R)
    AAA_tmp[i][i + AAA_disp] = localGA_GreenCoef[i]; // self_GreenCoef[itype];
                                                     // force->pair->single(i, i, itype, itype, 0,
                                                     // factor_coul, factor_lj, fforce);
    // printf("%d %f\n", i, AAA_tmp[i][i]);
    /*
    for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] +
    localGA_numneigh[i]; jj++) { jseq = GAPair_localSeq[jj]; jtag = tag[jseq];
      jGA_index = tag2GA_index[jtag];
      AAA_tmp[i][jGA_index] = GAPair_coeff[jj];
      AAA_tmp_paircount++;
      // add j<->i from i<->j interaction
      if (jGA_index > localGA_index_min && jGA_index < localGA_index_max) {
        AAA[jGA_index][i + AAA_disp] = GAPair_coeff[jj];
        AAA_tmp_paircount++;
      }
    }
    */
    totalneigh_count += localGA_numlocalneigh[i] + localGA_numneigh[i];
    // printf("%d %d\n", i, localGA_numneigh[i]);
    for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj++) {
      jseq = firstneigh_seq[jj];
      // jtag = tag[jseq];
      jtag                  = jseq;
      jGA_index             = tag2GA_index[jtag];
      AAA_tmp[i][jGA_index] = AAA[jGA_index][i + AAA_disp] = firstneigh_coeff[jj];
    }
    for (jj = localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
      jseq                  = firstneigh_seq[jj];
      jtag                  = jseq;
      jGA_index             = tag2GA_index[jtag];
      AAA_tmp[i][jGA_index] = firstneigh_coeff[jj];
      if (jGA_index > localGA_index_min && jGA_index < localGA_index_max) {
        AAA[jGA_index][i + AAA_disp] = firstneigh_coeff[jj];
        totalneigh_count++;
      }
    }
  }
  // printf("me %d, AAA_tmp_paircount = %d \n", me, AAA_tmp_paircount);

  if (totalneigh_count != localGA_num * totalGA_num) {
    sprintf(info,
            "[ConpGA] setup_AAA_Direct: CPU%d: localGA_num: %d, Obtained_Pair "
            "%d, expect: %d. To increase pair cutoff!\n",
            me, localGA_num, totalneigh_count, localGA_num * totalGA_num);
    error->one(FLERR, info);
  }
  // printf("me %d, AAA_tmp_paircount = %d\n", me, AAA_tmp_paircount);

  MPI_Barrier(world);
  for (iproc = 0; iproc < nprocs; ++iproc) {
    // printf("%d %d %d %d\n", me, iproc, localGA_EachProc_Dipl[iproc],
    // localGA_EachProc[iproc]*totalGA_num);
    AAA_disp = localGA_EachProc_Dipl[iproc];
    MPI_Bcast(AAA, localGA_EachProc[iproc] * totalGA_num, MPI_DOUBLE, iproc, world);
  }
  if (me == work_me && DEBUG_LOG_LEVEL > 0) {
    printf("CPU-%d for AAA =\n", me);
    for (i = 0; i < totalGA_num; ++i) {
      printf("Atom #%2d ", totalGA_tag[i]);
      for (j = 0; j < totalGA_num; ++j) {
        printf("%12.8g ", AAA[i][j]);
      }
      printf("\n");
      // printf("  MeanSum = %12.10f\n", AAA_sum1D[i]);
    }
  }

  inv_AAA_LAPACK(AAA[0], totalGA_num);

  meanSum_AAA();

  // printf("Complete inv_AAA\n");

  // printf("\n inv(A) = ,  %d\n", itotalGA);
  if (me == work_me and DEBUG_LOG_LEVEL > 0) {
    printf("CPU-%d for inv(AAA)=\n", me);
    for (i = 0; i < totalGA_num; ++i) {
      printf("Atom #%2d ", totalGA_tag[i]);
      for (j = 0; j < totalGA_num; ++j) {
        printf("%12.8g ", AAA[i][j]);
      }
      printf("  MeanSum = %12.10g\n", AAA_sum1D[i]);
    }
  }
}

#ifdef SCALAPACK

int FixConpGA::inv_AAA_SCALAPACK(double* A, MKL_INT n)
{
  FUNC_MACRO(0);
  double total_time = 0, add_time = 0, inv_time = 0;
  time_t t_begin_A = clock();

  MKL_INT iam, blas_nprocs;
  MKL_INT nprow = 1, npcol = 1;
  const MKL_INT izero = 0, ione = 1, inegone = -1;
  double zero = 0.0E+0, one = 1.0E+0;
  MKL_INT ictxt, myrow, mycol, m = n;

  for (MKL_INT i = (MKL_INT)(sqrt(nprocs)) + 1; i > 0; i--) {
    if (nprocs % i == 0) {
      nprow = i;
      npcol = nprocs / i;
      break;
    }
  }
  char order = 'R'; // Block cyclic, Row major processor mapping
  char buf[256];

  // ictxt = sys2blacs_handle_(world);
  ictxt = Csys2blacs_handle(world);

  // blacs_get_(&zero, &zero, &ictxt ); // -> Create context

  Cblacs_gridinit(&ictxt, &order, nprow, npcol);

  if (ictxt >= 0) Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

  MKL_INT descA[9], descA_distr[9];
  MKL_INT info;

  MKL_INT nb, mb;

  int np_max = nprow > npcol ? nprow : npcol;
  if (scalapack_nb * np_max > n) {
    for (nb = scalapack_nb; nb > 0; nb--) {
      if (nb * npcol <= n) {
        break;
      }
    }
    for (mb = scalapack_nb; mb > 0; mb--) {
      if (mb * nprow <= m) {
        break;
      }
    }
    nb = mb = std::min(mb, nb);
  } else
    nb = mb = scalapack_nb;

  if (me == 0) utils::logmesg(lmp, "[ConpGA] SCALAPACK nprow x npcol = {} x {} with block size = {} x {}\n", nprow, npcol, mb, nb);

  MKL_INT mloc = numroc_(&m, &mb, &myrow, &izero, &nprow); // My proc -> row of local A
  MKL_INT nloc = numroc_(&n, &nb, &mycol, &izero, &npcol); // My proc -> col of local A
  double* A_distr;
  memory->create(A_distr, mloc * nloc, "SCALAPACK::A_distr");
  MKL_INT* ipiv = new MKL_INT[mloc + mb]; // mloc + nb

  MKL_INT lwork = -1, liwork = -1;
  MKL_INT iwkopt[4];
  double wkopt[4];

  MKL_INT lld = numroc_(&n, &n, &myrow, &izero, &nprow);
  lld         = lld > 1 ? lld : 1;
  descinit_(descA, &m, &n, &m, &n, &izero, &izero, &ictxt, &lld, &info);
  if (me == 0 && info != 0) {
    sprintf(buf, "[ConpGA] SCALAPACK Error in descinit_ for descA, info = %d\n", info);
    error->one(FLERR, buf);
  }

  // printf("me = %d, %dx%d, nb = %d/%d\n", me, mloc, nloc, nb, n);

  MKL_INT ldd_distr = mloc > 1 ? mloc : 1;
  descinit_(descA_distr, &m, &n, &mb, &nb, &izero, &izero, &ictxt, &ldd_distr, &info);
  if (me == 0 && info != 0) {
    sprintf(buf, "[ConpGA] SCALAPACK Error in descinit_, info = %d\n", info);
    error->one(FLERR, buf);
  }

  time_t t_begin0 = clock();
  pdgeadd_("N", &m, &n, &one, A, &ione, &ione, descA, &zero, A_distr, &ione, &ione, descA_distr);
  add_time += double(clock() - t_begin0) / CLOCKS_PER_SEC; // printf("[%d]descA = %d %d %d %d %d %d %d %d %d\n", me, descA[0], descA[1],
  // descA[2], descA[3], descA[4],
  //     descA[5], descA[6], descA[7], descA[8]);

  pdgetrf_(&n, &n, A_distr, &ione, &ione, descA_distr, ipiv, &info);
  if (me == 0 && info != 0) {
    sprintf(buf, "[ConpGA] SCALAPACK Error in pdgetrf_, info = %d\n", info);
    error->one(FLERR, buf);
  }

  pdgetri_(&n, A_distr, &ione, &ione, descA_distr, ipiv, wkopt, &lwork, iwkopt, &liwork, &info);
  if (me == 0 && info != 0) {
    sprintf(buf, "[ConpGA] SCALAPACK Error in pdgetri_ with lwork = -1, info = %d\n", info);
    error->one(FLERR, buf);
  }

  lwork  = (MKL_INT)(wkopt[0]) + 1;
  liwork = iwkopt[0];
  if (me == 0) utils::logmesg(lmp, "[ConpGA] SCALAPACK with lwork = {}, liwork = {}\n", lwork, liwork);
  if (lwork <= 0 || lwork > max_lwork) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] SCALAPACK with reseting lwork as {}\n", max_lwork);
    lwork = max_lwork;
  }
  if (liwork <= 0 || lwork > max_liwork) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] SCALAPACK with reseting liwork as {}\n", max_liwork);
    liwork = max_liwork;
  }

  double* work   = new double[lwork];
  MKL_INT* iwork = new MKL_INT[liwork];

  t_begin0 = clock();
  pdgetri_(&n, A_distr, &ione, &ione, descA_distr, ipiv, work, &lwork, iwork, &liwork, &info);
  inv_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;

  if (me == 0 && info != 0) {
    sprintf(buf, "[ConpGA] SCALAPACK Error in pdgetri_, info = %d\n", info);
    error->one(FLERR, buf);
  }

  t_begin0 = clock();
  pdgeadd_("N", &m, &n, &one, A_distr, &ione, &ione, descA_distr, &zero, A, &ione, &ione, descA);
  add_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;

  delete[] work;
  delete[] iwork;
  delete[] ipiv;

  Cblacs_gridexit(ictxt);

  total_time += double(clock() - t_begin_A) / CLOCKS_PER_SEC;
  if (me == 0 && DEBUG_LOG_LEVEL > -1) utils::logmesg(lmp, "add_time = {:f}, inv_time = {:f}, total_time = {:f}\n", add_time, inv_time, total_time);
  return nb;
}
#endif

void FixConpGA::coul_compute(int clear_flag)
{
  /*
  FUNC_MACRO(0);
  time_t t_begin0 = clock();
  force_clear();
  fix_forward_comm(0);
  force->pair->compute(3, 0);
  if(newton_pair)
    fix_reverse_comm(REV_COUL);
  update_P_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;

  t_begin0 = clock();
  force->kspace->qsum_qsq();
  force->kspace->compute(3,0); // eflag = 1001, and vflag = 0 to skip force
  calculation in kspace; update_K_time += double(clock() - t_begin0) /
  CLOCKS_PER_SEC;
  */
}

void FixConpGA::classic_compute_AAA()
{
  t_begin = clock();
  int i, j, ii, *i_ptr, *iend_ptr, *istart_ptr, ilocalseq;
  int nlocal = atom->nlocal, nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int* type  = atom->type;
  int iproc, procs = comm->nprocs, itestGA_tag, iseq_comm;

  double* prd           = domain->prd;
  double volume         = prd[0] * prd[1] * prd[2];
  double g_ewald        = force->kspace->g_ewald;
  double slab_volfactor = force->kspace->slab_volfactor;

  double* q          = atom->q;
  double test_charge = totalGA_num - 1;              // totalGA_num*1.1;
  double qsum        = double(totalGA_num) / 10 + 1; // double(totalGA_num)/10+1;
  double CCC_b       = 1 / (totalGA_num * qsum + qsum * qsum);
  double CCC_a = (qsum)*CCC_b, self_potential = 0, value, value_i, value_j;
  const double self_g_ewald_CONST = force->qqrd2e * MY_PI2 / (g_ewald * g_ewald * volume * slab_volfactor);

  double* AAA_sum;
  test_charge += qsum;
  if (me == 0) {
    utils::logmesg(lmp, "[ConpGA] \\frac{{\\pi}}{{\\eta^2 V}} = {:f} [energy unit]\n", self_g_ewald_CONST);
    utils::logmesg(lmp, "[ConpGA] \\tidle{{Q}}    Mat: q_ii = {:f} = {:f} + {:f}, q_ij = {:f}\n", test_charge, qsum, (double)(totalGA_num - 1), -1.0);
    utils::logmesg(lmp, "[ConpGA] \\tidle{{Q}}^-1 Mat: q_ii^-1 = {:.16f}, q_ij^-1 = {:.16f}\n", CCC_a + CCC_b, CCC_b);
  }

  istart_ptr = &localGA_seq.front();
  iend_ptr   = istart_ptr + localGA_num;

  ilocalseq = *istart_ptr;

  q_store.reserve(ntotal);
  double* const q_store_arr       = &q_store.front();
  double* const localGA_GreenCoef = &localGA_GreenCoef_vec.front();
  for (size_t i = 0; i < ntotal; ++i) {
    q_store_arr[i] = q[i];
    q[i]           = 1;
  }

  double* const cl_atom          = electro->cl_atom();
  double* const ek_atom          = electro->ek_atom();
  const int* const size_EachProc = &localGA_EachProc.front();
  const int* const size_Dipl     = &localGA_EachProc_Dipl.front();
  int *sizeSOL_EachProc = nullptr, *sizeSOL_Dipl = nullptr;
  double **Umat = nullptr, *Umat_1D = nullptr, **Umat_Global;
  double* pe_RML;

  /*
  int localSOL_num = nlocal - localGA_num, iSOL, totalSOL_num;
  memory->create(pe_RML, 10*nlocal, "FixConp/GA: PE_store");
  */
  // to derive env_alpha
  // Sol + GA

  // if (env_alpha_flag) {
  if (0) {
    /*
    MPI_Allreduce(&localSOL_num, &totalSOL_num, 1, MPI_INT, MPI_SUM, world);
    memory->create(Umat, localSOL_num, totalGA_num, "FixConp/GA: Umat");
    memory->create(Umat_1D, localSOL_num, "FixConp/GA: Umat");

    memory->create(sizeSOL_EachProc, nprocs, "FixConp/GA: sizeSOL_EachProc");
    memory->create(sizeSOL_Dipl, nprocs, "FixConp/GA: sizeSOL_EachProc");
    MPI_Allgather(&localSOL_num, 1, MPI_INT, sizeSOL_EachProc, 1, MPI_INT, world);

    sizeSOL_Dipl[0] = 0;
    for (i = 1; i < nprocs; ++i) {
      sizeSOL_Dipl[i] = sizeSOL_Dipl[i - 1] + sizeSOL_EachProc[i - 1];
    }

    if (me == 0) {
      memory->create(Umat_Global, totalSOL_num, totalGA_num, "FixConp/GA: Umat");
    } else {
      Umat_Global = Umat;
    }
    electro->qsum_qsq();
    electro->compute(3, 0);
    for (i = 0; i < nlocal; ++i) {
      if (GA_Flags[type[i]]) {
        pe_RML[i] = ek_atom[i]; // + cl_atom[i];
        // q[i] = 1;
      } else {
        q[i] = 0; // set q_SOL as 0 with q_GA = 1
      }
    }

    // GA only
    electro->qsum_qsq();
    electro->compute(3, 0);
    for (i = 0; i < nlocal; ++i) {
      if (GA_Flags[type[i]] != 0) {
        pe_RML[i] -= ek_atom[i]; // + cl_atom[i];
        q[i] = 0;                // set q_GA = 0;
      } else {
        q[i] = 1; // set q_SOL = 1;
      }
    }

    electro->qsum_qsq();
    electro->compute(3, 0);
    for (i = 0; i < nlocal; ++i) {
      if (GA_Flags[type[i]] != 0) {
        q[i] = -1;
      } else {
        pe_RML[i]  = -ek_atom[i]; // + cl_atom[i];
        q[i]       = 1;
        cl_atom[i] = pe_RML[i];
      }
    }
    */
  } else {
    for (i = 0; i < nlocal; ++i) {
      if (GA_Flags[type[i]] != 0) {
        // pe_RML[i] = 0;
        q[i] = -1;
      } else {
        q[i] = 0;
      }
    }
  } // env_alpha_flag;

  // solv_e_LS(Umat[0], pe_RML, localSOL_num, totalGA_num);

  for (i = nlocal; i < ntotal; ++i)
    cl_atom[i] = 0;

  for (iproc = 0; iproc < nprocs; ++iproc) {
    // The total charge is set to be zero, so the effect is re-added as
    // AAA_sum[i] -= self_g_ewald_CONST;
    if (localGA_EachProc[iproc] > 0) {
      if (me == iproc) q[localGA_seq[0]] = test_charge;
      electro->qsum_qsq();
      if (me == iproc) q[localGA_seq[0]] = -1;
      break;
    }
  }

  int AAA_disp = localGA_EachProc_Dipl[me];
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Generate AAA matrix:\n");
  int itotalGA = 0;
  int* tag     = atom->tag;

  comm_forward = comm_reverse = 1;
  int printf_flag = 0, printf_loops = 50;

  for (iproc = 0; iproc < nprocs; ++iproc)
  // if(0) // for inv_test
  {
    for (i = 0; i < localGA_EachProc[iproc]; ++i, ++itotalGA) {
      itestGA_tag = totalGA_tag[itotalGA];
      if (printf_flag % printf_loops == 0) {
        t_end = clock();
        if (me == 0)
          utils::logmesg(lmp, "progress = {:.2f}%% - {}/{}, time elapsed: {:8.2f}[s]\n", (float)(itotalGA + 1) / totalGA_num * 100, itotalGA + 1, totalGA_num,
                         double(t_end - t_begin) / CLOCKS_PER_SEC);
      }
      printf_flag++;

      if (iproc == me) {
        q[istart_ptr[i]] = test_charge;
        // local_Self_Energy[i] = localGA_GreenCoef[i];
        // self_GreenCoef[type[istart_ptr[i]]];
        // if(i != 0) q[istart_ptr[i-1]] = -1;
      } else {
        // if (tag2seq.count(itestGA_tag)) {
        iseq_comm = atom->map(itestGA_tag);
        if (iseq_comm != -1) {
          q[iseq_comm] = test_charge;
          // tag2seq_count = true;
        }
        // printf("Iter%d CPU:%d Change Seq %d[%d] change to %f\n", itotalGA, me
        // , tag2seq[itestGA_tag], itestGA_tag, q[tag2seq[itestGA_tag]]);
      }

      // force_clear();

      fix_forward_comm(0);
      fix_pair_compute(); // classic_compute_AAA
      // if (env_alpha_flag) {
      if (0) {
        fix_localGA_pairenergy_flag(0, false);
      } else {
        if (newton_pair) fix_reverse_comm(REV_COUL);
      }
      // For calulcation with newton = on
      // Take a reverse communication to send cl_atom energy;

      // if (newton_pair) {
      //  fix_reverse_comm(REV_COUL);
      //}
      // printf("force->pair->compute() itotalGA= %d/%d\n",itotalGA,
      // totalGA_num);
      electro->compute(3, 0);
      // printf("qsqsum = %g\n", force->kspace->qsqsum);
      /*
      force->kspace->eatom[0] = 100;
      force->kspace->eatom[1] = 101;
      force->kspace->eatom[2] = 102;
      force->kspace->qsum_qsq();
      force->kspace->compute(3, 0);
      */
      // printf("force->kspace->compute() itotalGA= %d/%d\n",itotalGA,
      // totalGA_num);

      for (ii = 0; ii < localGA_num; ++ii) {
        ilocalseq         = istart_ptr[ii];
        AAA[ii][itotalGA] = ek_atom[ilocalseq] + cl_atom[ilocalseq]; // +  +   // + pe_RML[ilocalseq];
#ifndef CONP_NO_DEBUG
        if (iproc == work_me && DEBUG_LOG_LEVEL > 0) {
          printf("Iter%d CPU:%d ilocalseq: %d[tag:%d] Q: %f,%f,%f eatom: %f, "
                 "cl_atom: %f, cl_atom: %f, self_U: %f\n",
                 itotalGA, me, ilocalseq, tag[ilocalseq], q[0], q[1], q[2], force->kspace->eatom[ilocalseq], electro->cl_atom()[ilocalseq],
                 force->pair->eatom[ilocalseq], self_potential * q[ilocalseq]);
        }
#endif
      }
      /*
      if (env_alpha_flag) {
        iSOL = 0;
        for (ii = 0; ii < nlocal; ++ii) {
          if (GA_Flags[type[ii]]) {
            ;
          } else {
            Umat[iSOL][itotalGA] = cl_atom[ii]
            + kspace_eatom[ii];
            iSOL++;
            cl_atom[ii] = pe_RML[ii];
          }
        }
      }
      */
      // printf("AAA_tmp itotalGA= %d/%d\n",itotalGA, totalGA_num);
      if (iproc == me) {
        // Get the electric potential on each test electrode atoms;
        // AAA_disp = starting point of sequence for my proc and i is the index
        // of local number AAA[i][itotalGA] -= (test_charge+1) *
        // pe_RML[istart_ptr[i]];
        AAA[i][itotalGA] /= -test_charge;
      }

      if (iproc == me) {
        q[istart_ptr[i]] = -1;
      } else if (iseq_comm != -1) { // The atom in the ghost list
        q[iseq_comm] = -1;
        // tag2seq_count = false;
      }
    }
  }

  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] End of AAA Matrix calulation, Total elapsed: {:8.6f}[s]\n", double(t_end - t_begin) / CLOCKS_PER_SEC);

  map<int, int> tag2GA_index;
  MAP_RESERVE(tag2GA_index, totalGA_num);
  for (i = 0; i < totalGA_num; i++) {
    tag2GA_index[totalGA_tag[i]] = i;
  }

  if (me == 0) utils::logmesg(lmp, "[ConpGA] Start of AAA Matrix calulation by AAA = inv(CCC)*BBB\n");

  memory->create(AAA_sum, localGA_num, "FixConp/GA: AAA_sum");

  t_begin = clock();
  for (i = 0; i < localGA_num; ++i) {
    AAA_sum[i] = 0;
    for (j = 0; j < totalGA_num; ++j) {
      AAA_sum[i] -= AAA[i][j];
    }
    AAA_sum[i] *= CCC_b;
    // AAA_sum[i] -= self_g_ewald_CONST;
  }

  for (i = 0; i < localGA_num; ++i) {
    // value_i = total_Self_Energy[i] - AAA_sum[i] + CCC_a * AAA_tmp[i][i];
    for (j = 0; j < totalGA_num; ++j) {
      // AAA[AAA_disp + i][j] = AAA_sum[i] - CCC_a * AAA_tmp[i][j]; // +
      // value_i;
      AAA[i][j] = AAA_sum[i] - CCC_a * AAA[i][j]; // + value_i;
    }
    // AAA[i][i] = total_Self_Energy[i];
    // printf("AAA[%d][%d] = %.16f\n", i, i, AAA[i][i]);
    // printf("\n");
  }
  /*
  if (env_alpha_flag) {
    // write_mat(Umat[0], localSOL_num*totalGA_num, "Umat0.mat");
    for (i = 0; i < localSOL_num; ++i) {
      Umat_1D[i] = 0;
      for (j = 0; j < totalGA_num; ++j) {
        Umat_1D[i] += Umat[i][j];
      }
      Umat_1D[i] *= CCC_b;
      // AAA_sum[i] -= self_g_ewald_CONST;
    }

    for (i = 0; i < localSOL_num; ++i) {
      for (j = 0; j < totalGA_num; ++j) {
        Umat[i][j] = Umat_1D[i] + CCC_a * Umat[i][j]; // + value_i;
      }
      Umat_1D[i] = 1;
      // AAA_sum[i] -= self_g_ewald_CONST;
    }
    // write_mat(Umat[0], localSOL_num*totalGA_num, "Umat.mat");



    env_alpha.reserve(localGA_num);

    double * const env_alpha_arr = &env_alpha.front();
    //for (i = 0; i < localSOL_num; ++i) {
    //  env_alpha_arr[i] = 1.0;
    //}


    MPI_Gatherv(Umat[0], localSOL_num, MPI_totalGA,
      Umat_Global[0], sizeSOL_EachProc, sizeSOL_Dipl, MPI_totalGA, 0, world
    );
    if (me == 0) {
      memory->grow(Umat_1D, totalSOL_num, "Fix:Conp growth to totalSOL_num >
  totalGA_num Umat_1D for me == 0"); for (i = 0; i < totalSOL_num; ++i)
  Umat_1D[i] = 1;

      double *Unew; int jj = 0;
      memory->create(Unew, localSOL_num * totalGA_num,
  "fix::conpGA::z_localGA_prv"); for (j = 0; j < totalGA_num; ++j) { for (i = 0;
  i < localSOL_num; ++i) { Unew[jj] = Umat[i][j]; jj++;
        }
      }

      write_mat(Umat_Global[0], totalSOL_num * totalGA_num, "Umat.mat");
      solve_LS(Unew, Umat_1D, 'N', totalSOL_num, totalGA_num);
      write_mat(Umat_1D, totalGA_num, "Umat_1D.mat");

      memory->destroy(Umat_Global);
    } else {
      memory->grow(Umat_1D, totalGA_num, "Fix:Conp grow to totalGA_num Umat_1D
  for me != 0");
    }

    MPI_Bcast(Umat_1D, 1, MPI_totalGA, 0, world);
    for (i = 0; i < localGA_num; ++i) {
      value = 0;
      for (j = 0; j < totalGA_num; ++j) {
        value += AAA[i][j] * Umat_1D[j];
      }
      env_alpha_arr[i] = (1 - value) / Umat_1D[size_Dipl[me] + i];
    }
    if(me == 0)
    write_mat(env_alpha_arr, localGA_num, "env_alpha.mat");

    Umat_Global = nullptr;
  }
  */

  memory->destroy(AAA_sum);
  // memory->destroy(Umat);
  // memory->destroy(Umat_1D);

  // Gather to root AAA_Global
  std::copy(q_store_arr, q_store_arr + ntotal, q);
  electro->qsum_qsq();
  force_clear();
  fix_forward_comm(0);
}

void FixConpGA::inverse(double** mat_all, int N, double**& mat_local, int localN)
{
#ifdef SCALAPACK
  if (N > scalapack_num && nprocs > 0) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] Inverse AAA via SCALAPACK routine, {} > {} !\n", N, scalapack_num);
    memory->destroy(mat_local);
    inv_AAA_SCALAPACK(mat_all[0], N);
    if (localN > 0) memory->create(mat_local, localN, N, "inverse:mat_local");
#else
  if (0) {
    ;
#endif
  } else {
    /* -------- DO LAPACK ------- */
    if (me == 0) utils::logmesg(lmp, "[ConpGA] Inverse AAA via LAPACK routine, {} < {} !\n", N, scalapack_num);
    if (me == 0) inv_AAA_LAPACK(mat_all[0], N);
  }
}

/* ----------------------------------------------------------------------------
  Generate the interaction AAA matrix between electrode atoms
---------------------------------------------------------------------------- */
void FixConpGA::setup_AAA()
{
  FUNC_MACRO(0);
  time_t t_begin_AAA = clock();
#ifdef SCALAPACK
  if (totalGA_num > scalapack_num) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] inverse the Ewald interaction matrix using SCALAPACK\n");
  } else {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] inverse the Ewald interaction matrix using LAPACK\n");
  }
#endif
  /* ---------- allocate the AAA matrix -----*/
  int AAA_len                    = localGA_num, i, j;
  const int* const size_EachProc = &localGA_EachProc.front();
  const int* const size_Dipl     = &localGA_EachProc_Dipl.front();
  if (AAA_len == 0)
    memory->create(AAA, 1, 1, "FixConp/GA: AAA_tmp");
  else
    memory->create(AAA, AAA_len, totalGA_num, "FixConp/GA: AAA_tmp");

  if (me == 0) utils::logmesg(lmp, "[ConpGA] allocate AAA_Global strucute for {:8.6f}[s]!\n", double(clock() - t_begin_AAA) / CLOCKS_PER_SEC);

  // for electro compute using the intermediate Green's function matrix
  if (pair_flag && electro->is_compute_AAA()) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] Compute the Ewald matrix via {}\n", electro->type_name());
    electro->compute_AAA(AAA);
  } else {
    if (me == 0)
      utils::logmesg(lmp, "[ConpGA] Compute the Ewald matrix via the default N "
                          "times Ewald summation.\n");
    classic_compute_AAA();
  }
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Compute Ewald matrix for {:8.6f}!\n", double(clock() - t_begin_AAA) / CLOCKS_PER_SEC);

  double *q = atom->q, **AAA_Global;
  MPI_Datatype MPI_totalGA;
  MPI_Type_contiguous(totalGA_num, MPI_DOUBLE, &MPI_totalGA);
  MPI_Type_commit(&MPI_totalGA);

  if (me == 0) {
    memory->create(AAA_Global, totalGA_num, totalGA_num, "FixConp/GA: AAA_Global");
  } else
    memory->create(AAA_Global, 1, 1, "FixConp/GA: AAA_Global");
  // AAA_Global = AAA;

  // Gather to root AAA_Global
  MPI_Gatherv(AAA[0], localGA_num, MPI_totalGA, AAA_Global[0], size_EachProc, size_Dipl, MPI_totalGA, 0, world);

  // final all data is gather back to AAA_Global at (me == 0)

  inverse(AAA_Global, totalGA_num, AAA, localGA_num);

  t_end = clock();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Matrix Inversion for elapsed: {:8.6f}[s]\n", double(t_end - t_begin_AAA) / CLOCKS_PER_SEC);

  if (me == 0) make_symmetric_matrix(symmetry_flag, totalGA_num, AAA_Global);
  MPI_Barrier(world);

  matrix_scatter(AAA_Global, AAA, totalGA_num, size_EachProc, size_Dipl);
  memory->destroy(AAA_Global);

  t_end = clock();
  if (me == 0)
    utils::logmesg(lmp,
                   "[ConpGA] MPI communication to distribute matrix, Total elapsed: "
                   "{:8.6f}[s]\n",
                   double(t_end - t_begin_AAA) / CLOCKS_PER_SEC);

  // Gather to root AAA_Global
  meanSum_AAA();

  MPI_Type_free(&MPI_totalGA);
  // memory->destroy(total_Self_Energy);
  // memory->destroy(local_Self_Energy);
  force_clear();
  if (me == 0) utils::logmesg(lmp, "[ConpGA] End of Ewald Matrix calulation, Total elapsed: {:8.6f}[s]\n", double(t_end - t_begin_AAA) / CLOCKS_PER_SEC);
}

void FixConpGA::group_check()
{
  int nlocal = atom->nlocal, icount = 0;
  int *type = atom->type, *tag = atom->tag, *mask = atom->mask;
  for (int iseq = 0; iseq < nlocal; ++iseq) {
    if (mask[iseq] & groupbit) {
      icount++;
      if (GA_Flags[type[iseq]] == 0) {
        error->one(FLERR, "Fix::Conp inconsistent fix group and electrodes setup!");
      }
    } else {
    }
  }
  if (icount != localGA_num) {
    error->one(FLERR, "Fix::Conp inconsistent fix group and electrodes setup!!");
  }
}

void FixConpGA::make_symmetric_matrix(SYM flag, int n, double** mat)
{
  FUNC_MACRO(0);
  int i, j;
  double value, value_i, value_j;
  if (flag == SYM::MEAN) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < i; ++j) {
        value_i = mat[i][j];
        value_j = mat[j][i];
        value   = 0.5 * (value_i + value_j);
        // if (fabs(value_i-value_j)/value > 1e-2){
        //  error->all(FLERR, "Fix::Conp AAA matrix is not a symmetrix one!");
        //}
        mat[i][j] = mat[j][i] = value;
      }
    }
  } else if (flag == SYM::SQRT) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < i; ++j) {
        value_i = mat[i][j];
        value_j = mat[j][i];
        value   = sqrt(value_i * value_j);
        if (value_i < 0) value = -value;
        // if (fabs(value_i-value_j)/value > 1e-2){
        //  error->all(FLERR, "Fix::Conp AAA matrix is not a symmetrix one!");
        //}
        mat[i][j] = mat[j][i] = value;
      }
    }
  } else if (flag == SYM::TRAN) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < i; ++j) {
        value_i   = mat[i][j];
        mat[i][j] = mat[j][i];
        mat[j][i] = value_i;
      }
    }
  }
}

/* -------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------
*/
void FixConpGA::update_charge_ppcg()
{
  FUNC_MACRO(0)
  int i, j, ii, iter;
  int nlocal = atom->nlocal, *type = atom->type, *tag = atom->tag;
  int* GA_seq_arr = &localGA_seq.front();
  int* GA_seq_end = &localGA_seq.front() + localGA_num;
  int iseq, i_nbr, iBlock;
  double value, qtmp, alpha, beta, tmp, *q = atom->q;
  double q_max_local, q_max_total = 0;
  double q_std_local, q_std_total = 0;
  double q_sqr_local, q_sqr_total = 0;
  double PE_ij_local, PE_ij_total = 0, dPE_ij_total;

  // double qGA_incr_sqr, qGA_incr_sqr_total, tmp;
  double resnorm_local, resnorm_total, next_resnorm_local, next_resnorm_total;
  // double next_Polak_resnorm_local, next_Polak_resnorm_total;
  double ptap_local, ptap_total, Pshift_by_res = 0;
  int neighbor_ago_store   = neighbor->ago;
  double *cl_atom          = electro->cl_atom(), *ek_atom;
  double qlocalGA_incr_sum = 0, qsum_store, qGA_incr_sum, qGA_incr, local_res_sum, total_res_sum;
  std::vector<double> virtual_kspace_eatom;
  t_begin = clock();

  ek_atom = electro->ek_atom();
  if (neutrality_flag == true) {
    if (neutrality_state == false) {
      qsum_store = electro->calc_qsum();
    } else {
#ifndef CONP_NO_DEBUG
      qsum_store = electro->calc_qsum();
      if (fabs(qsum_store) > 1e-11)
        if (me == 0) utils::logmesg(lmp, "invalid neutrality state of qsum = %.10f\n", qsum_store);
#endif
    }
  }

  q_store.reserve(nlocal);
  double* const q_store_arr = &q_store.front();
  double* const p_localGA   = &p_localGA_vec.front();
  double* const ap_localGA  = &ap_localGA_vec.front();
  double* const res_totalGA = &res_localGA_vec.front();
#ifndef INTEL_MPI
  double* const res_localGA = res_totalGA + localGA_EachProc_Dipl[me];
#else
  double* const res_localGA = res_totalGA + totalGA_num;
#endif

  double* z_localGA     = &z_localGA_vec.front();
  double* z_localGA_prv = &z_localGA_prv_vec.front();

  double* z_localGA_tmp; // * z_localGA_tmp;

  for (int i = 0; i < localGA_num; ++i) {
    z_localGA_prv[i] = 0.0;
  }

  for (i = 0; i < nlocal; ++i) {
    q_store_arr[i] = q[i];
    q[i]           = 0;
  }
  if (!GA2GA_flag) {
    for (i = 0; i < localGA_num; i++) {
      iseq = GA_seq_arr[i];
      qlocalGA_incr_sum -= q_store_arr[iseq];
      q_store_arr[iseq] = 0;
      B0_prev[i]        = 0;
    }
  }

  local_res_sum = 0;

  if (tol_style == TOL::REL_B) {
    PE_ij_local = 0;
    for (i = 0; i < localGA_num; i++) {
      iseq  = GA_seq_arr[i];
      value = ek_atom[iseq];
      // Copy the previous potential and shift it by the neutrality constraint
      // B0_next[i] = B0_prev[i] - Ushift;
      // the potential by the all electrolyte molecule \sum_J e_{iJ}
      B0_next[i] = value - B0_prev[i];

      // as the electric energy \sum_J e_{iJ}q_I * q_i
      PE_ij_local += qtmp * B0_next[i];

      res_localGA[i] = localGA_Vext[i] - value;
      local_res_sum += res_localGA[i];
      // if (me == 0) utils::logmesg(lmp, "Seq: %d, Charge %f, U_ext
      // %f\n", iseq, qtmp, res_localGA[i]);
    }
    MPI_Allreduce(&PE_ij_local, &PE_ij_total, 1, MPI_DOUBLE, MPI_SUM, world);
  } else {
    for (i = 0; i < localGA_num; i++) {
      iseq           = GA_seq_arr[i];
      value          = ek_atom[iseq];
      res_localGA[i] = localGA_Vext[i] - value;
      local_res_sum += res_localGA[i];
    }
  }

  if (bisect_level == 0) {
    MPI_Allreduce(&local_res_sum, &total_res_sum, 1, MPI_DOUBLE, MPI_SUM, world);
    Pshift_by_res = total_res_sum * pcg_shift;
  }
  //  To communicate residual on the ghost atom
  MPI_Allgatherv(res_localGA, localGA_num, MPI_DOUBLE, res_totalGA, &localGA_EachProc.front(), &localGA_EachProc_Dipl.front(), MPI_DOUBLE, world);

  neighbor->ago      = 0; // Set to re-check the atom charge quantity
  resnorm_local      = 0;
  next_resnorm_local = 0;
  ptap_local         = 0;

  for (i = 0; i < localGA_num; ++i) {
    value = Pshift_by_res;
    i_nbr = ppcg_block[i][0];
    for (j = 0; j < i_nbr; ++j) {
      value += P_PPCG[i][j] * res_totalGA[ppcg_GAindex[i][j]];
    }
    // if bisect_level == 0 give iBlock = 1;
    for (iBlock = (bisect_level == 0); iBlock < bisect_num; ++iBlock) {
      qtmp = 0;
      for (j = ppcg_block[i][iBlock]; j < ppcg_block[i][iBlock + 1]; ++j) {
        qtmp += res_totalGA[ppcg_GAindex[i][j]];
      }
      value += qtmp * block_ave_value[iBlock];
    }

    p_localGA[i] = z_localGA[i] = value;
    resnorm_local += z_localGA[i] * res_localGA[i];
    next_resnorm_local += res_localGA[i] * res_localGA[i];
    ptap_local += z_localGA[i] * z_localGA[i];
  }

  z_localGA_tmp = z_localGA;
  z_localGA     = z_localGA_prv;
  z_localGA_prv = z_localGA_tmp; // swap(k and k+1);

  MPI_Allreduce(&resnorm_local, &resnorm_total, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&next_resnorm_local, &next_resnorm_total, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&ptap_local, &ptap_total, 1, MPI_DOUBLE, MPI_SUM, world);

#ifndef CONP_NO_DEBUG
  if (me == work_me) {
    t_end = clock();
    utils::logmesg(lmp, "Iteration 0: z*r = {:20.18g}, r*r = {:20.18g}, z*z = {:20.18g}\n", resnorm_total, next_resnorm_total, ptap_total);
  }
#endif

  for (iter = 1; iter < maxiter + 1; iter++) {
    // ----------- Calculate ap = AAA*p update_ch_arge_PPCG[] -----------------
    for (i = 0; i < localGA_num; i++) {
      iseq    = GA_seq_arr[i];
      q[iseq] = p_localGA[i];
    }

    time_t t_begin = clock();
    fix_forward_comm(0);
    electro->pair_GAtoGA();
    update_P_time += double(clock() - t_begin) / CLOCKS_PER_SEC;

    t_begin = clock();
    electro->compute_GAtoGA();
    electro->calc_GA_potential(STAT::E, STAT::E, SUM_OP::BOTH);
    update_K_time += double(clock() - t_begin) / CLOCKS_PER_SEC;

    ptap_local = 0;
    for (i = 0; i < localGA_num; i++) {
      iseq = GA_seq_arr[i];
      // value = pair_ecatom[iseq] + kspace_eatom[iseq];
      ptap_local += cl_atom[iseq];
      // qtmp = q[iseq];
      // value /= qtmp;
      // value += self_ECpotential;
      ap_localGA[i] = ek_atom[iseq];
      // ptap_local += value * qtmp;
    }
    MPI_Allreduce(&ptap_local, &ptap_total, 1, MPI_DOUBLE, MPI_SUM, world);
    // ----------- End of Calculate ap = AAA*p -----------------

    // ----------- update r and q by r' = r - d*ap; q' = q + d*p;
    // -----------------
    alpha = resnorm_total / ptap_total;
    // printf("alpha = %f, resnorm_total = %f, ptap_total = %f\n", alpha,
    // resnorm_total, ptap_total);
    local_res_sum = 0;
    if (tol_style == TOL::REL_B) {
      PE_ij_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
        PE_ij_local += qGA_incr * B0_next[i];

        qlocalGA_incr_sum += qGA_incr;
        tmp = alpha * ap_localGA[i];
        B0_prev[i] += tmp;
        res_localGA[i] -= tmp;
        // B0_next[i] += tmp;
        local_res_sum += res_localGA[i];
      }

      MPI_Allreduce(&PE_ij_local, &dPE_ij_total, 1, MPI_DOUBLE, MPI_SUM, world);
      PE_ij_total += dPE_ij_total;

      if (fabs(dPE_ij_total / PE_ij_total) < tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} rel_B0 * q = {:g} netcharge = "
                         "{:g} Total Elps. {:8.2g}[s]\n",
                         iter, dPE_ij_total / PE_ij_total, qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      } // if
    } else if (tol_style == TOL::RES) {
      printf("start res\n");
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
        qlocalGA_incr_sum += qGA_incr;
        res_localGA[i] -= alpha * ap_localGA[i];
        local_res_sum += res_localGA[i];
      }
      printf("end res\n");

    } else if (tol_style == TOL::MAX_Q) {
      q_max_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
        qlocalGA_incr_sum += qGA_incr;
        res_localGA[i] -= alpha * ap_localGA[i];
        local_res_sum += res_localGA[i];

        tmp = fabs(qGA_incr / q_store_arr[iseq]);
        if (tmp > q_max_local) {
          q_max_local = tmp;
        }
      }
      MPI_Allreduce(&q_max_local, &q_max_total, 1, MPI_DOUBLE, MPI_MAX, world);

      if (q_max_total < tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} max_Q = {:g} netcharge = {:g} "
                         "Total Elps. {:8.2f}[s]\n",
                         iter, q_max_total, qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      } // if
    }
    /*
    for (i = 0; i < localGA_num; i++)
    {
      iseq = GA_seq_arr[i];
      qGA_incr = alpha * p_localGA[i];
      q_store_arr[iseq] += qGA_incr;
      qlocalGA_incr_sum += qGA_incr;
      res_localGA[i] -= alpha * ap_localGA[i];
      local_res_sum += res_localGA[i];

      // tmp           = qGA_incr / q_store_arr[iseq];
      // qGA_incr_sqr += tmp * tmp;
    }
    */
    // MPI_Allreduce(&qGA_incr_sqr, &qGA_incr_sqr_total, 1, MPI_DOUBLE, MPI_SUM,
    // world);

    /* for PE criterion
    if (ptap_total < tolerance * totalGA_num) {
      qGA_incr = qlocalGA_incr_sum;
      MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);

      #ifndef CONP_NO_DEBUG
      if (me == work_me) {
        t_end = clock();
        utils::logmesg(lmp,"***** Converged at iteration %d delta_PE = {:g} netcharge
    = {:g} Total Elps. %8.2f[s]\n", iter, ptap_total/totalGA_num, qGA_incr_sum +
    qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
      }
      #endif

      break;
    }
    */

    //  To communicate residual on the ghost atom
    MPI_Allgatherv(res_localGA, localGA_num, MPI_DOUBLE, res_totalGA, &localGA_EachProc.front(), &localGA_EachProc_Dipl.front(), MPI_DOUBLE, world);

    if (bisect_level == 0) {
      MPI_Allreduce(&local_res_sum, &total_res_sum, 1, MPI_DOUBLE, MPI_SUM, world);
      Pshift_by_res = total_res_sum * pcg_shift;
    }
    // ----------- End of update r and q; -----------------

    // ----------- update z using z' = M*r -----------------

    next_resnorm_local = 0; // next_Polak_resnorm_local
    if (bisect_level == 0) {
      for (i = 0; i < localGA_num; i++) {
        value = Pshift_by_res;
        i_nbr = ppcg_block[i][0];
        for (j = 0; j < i_nbr; ++j) {
          value += P_PPCG[i][j] * res_totalGA[ppcg_GAindex[i][j]];
        }
        z_localGA[i] = value;
        next_resnorm_local += res_localGA[i] * z_localGA[i];
        // next_Polak_resnorm_local += res_localGA[i]*z_localGA_prv[i];
      }
    } else {
      for (i = 0; i < localGA_num; i++) {
        value = 0;
        i_nbr = ppcg_block[i][0];
        for (j = 0; j < i_nbr; ++j) {
          value += P_PPCG[i][j] * res_totalGA[ppcg_GAindex[i][j]];
        }

        for (iBlock = 0; iBlock < bisect_num; ++iBlock) {
          qtmp = 0;
          for (j = ppcg_block[i][iBlock]; j < ppcg_block[i][iBlock + 1]; ++j) {
            qtmp += res_totalGA[ppcg_GAindex[i][j]];
          }
          value += qtmp * block_ave_value[iBlock];
        }
        z_localGA[i] = value;
        next_resnorm_local += res_localGA[i] * z_localGA[i];
        // next_Polak_resnorm_local += res_localGA[i]*z_localGA_prv[i];
      }
    } // else bisect_level != 0;

    MPI_Allreduce(&next_resnorm_local, &next_resnorm_total, 1, MPI_DOUBLE, MPI_SUM, world);
    // MPI_Allreduce(&next_Polak_resnorm_local, &next_Polak_resnorm_total, 1,
    // MPI_DOUBLE, MPI_SUM, world);

    beta = (next_resnorm_total) / resnorm_total;

    if (tol_style == TOL::RES) {
      if (fabs(next_resnorm_total) < totalGA_num * tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} res = {:g} netcharge = {:g} "
                         "Total Elps. {:8.2f}[s]\n",
                         iter, sqrt(next_resnorm_total / totalGA_num), qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      }
    }
    // ----------- update p using p_{k} = r_{k} + rho_{k}/rho_{k-1} * p_{k-1} ;
    // -----------------
    for (i = 0; i < localGA_num; i++) {
      // p_localGA[i] = accelerate_factor*res_localGA[i] + beta * p_localGA[i];
      p_localGA[i] = z_localGA[i] + beta * p_localGA[i];
    }

    resnorm_total = next_resnorm_total;
#ifndef CONP_NO_DEBUG
    if (me == work_me) {
      t_end = clock();
      utils::logmesg(lmp,
                     "Iteration {} delta_PE = {:g} res = {:g} q_max = {:g} q_std = {:g} rq_std = "
                     "{:g} rel_B0 * q = {:g} alpha = {:g} beta = {:g} Elps. {:8.2f}[s]\n",
                     iter, ptap_total / totalGA_num, sqrt(fabs(resnorm_total / totalGA_num)), q_max_total, sqrt(q_std_total / totalGA_num),
                     sqrt(q_std_total / q_sqr_total), dPE_ij_total / PE_ij_total, alpha, beta, double(t_end - t_begin) / CLOCKS_PER_SEC);
    }
#endif
    neighbor->ago++;
    z_localGA_tmp = z_localGA;
    z_localGA     = z_localGA_prv;
    z_localGA_prv = z_localGA_tmp; // swap(k and k+1);

  } // iter
  curriter = iter;

  neighbor->ago = neighbor_ago_store; // 0; // neighbor_ago_store;

  if (postreat_flag) {
    MPI_Allreduce(&qlocalGA_incr_sum, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, world);
    // PPCG
    double* Ele_qsum_ptr = Ele_qsum;
    if (calc_EleQ_flag) {
      set_zeros(Ele_qsum, 0, Ele_num);
      for (int i = 0; i < localGA_num; i++) {
        int iEle = localGA_eleIndex[i];
        int iseq = GA_seq_arr[i];
        Ele_qsum[iEle] += q_store_arr[iseq];
      }
    } else {
      Ele_qsum_ptr = nullptr;
    }
    POST2ND_charge(qGA_incr_sum + qsum_store, Ele_qsum_ptr, q_store_arr);
    // localGA_charge_neutrality(qGA_incr_sum + qsum_store, q_store_arr);
  }
  // double *B_tmp; B_tmp = B0_next; B0_next = B0_prev; B0_prev = B_tmp;
  std::copy(q_store_arr, q_store_arr + nlocal, q);

  if (neutrality_state == true) {
    electro->set_qsqsum(0.0, 0.0);
  } else {
    electro->qsum_qsq();
  }
  electro->update_qstat();

  fix_forward_comm(0);

  time_t t_end  = clock();
  update_Q_time = double(t_end - t_begin) / CLOCKS_PER_SEC;
  update_Q_totaltime += update_Q_time;
} // update_ch_arge_ppcg

/* --------------------------------------------------------------------------------------------------------
  Update the atomic charge on each eletrode atom based on the precondition
conjugate gradient (PCG) Using Jacobi preconditioner
---------------------------------------------------------------------------------------------------------*/
void FixConpGA::update_charge_pcg()
{
  FUNC_MACRO(0)
  int i, j, ii, iter;
  int nlocal = atom->nlocal, *type = atom->type, *tag = atom->tag;
  int* GA_seq_arr = &localGA_seq.front();
  int* GA_seq_end = &localGA_seq.front() + localGA_num;
  int iseq;
  double value, qtmp, alpha, beta, tmp, *q = atom->q;
  double q_max_local, q_max_total = 0;
  double q_std_local, q_std_total = 0;
  double q_sqr_local, q_sqr_total = 0;
  double PE_ij_local, PE_ij_total = 0, dPE_ij_total;

  double resnorm_local, resnorm_total, next_resnorm_local, next_resnorm_total;
  // double next_Polak_resnorm_local, next_Polak_resnorm_total;
  double ptap_local, ptap_total;
  double Pshift_by_res, local_res_sum, total_res_sum;
  ;
  int neighbor_ago_store   = neighbor->ago;
  double *cl_atom          = electro->cl_atom(), *ek_atom;
  double qlocalGA_incr_sum = 0, qsum_store = 0, qGA_incr_sum, qGA_incr;
  std::vector<double> virtual_kspace_eatom;
  t_begin = clock();

  ek_atom = electro->ek_atom();
  if (neutrality_flag == true) {
    if (neutrality_state == false) {
      qsum_store = electro->calc_qsum();
    } else {
#ifndef CONP_NO_DEBUG
      qsum_store = electro->calc_qsum();
      if (fabs(qsum_store) > 1e-11)
        if (me == 0) utils::logmesg(lmp, "invalid neutrality state of qsum = %.10f\n", qsum_store);
#endif
    }
  }

  /*
  if ( tol_style == TOL::REL_B && q_store.capacity()>= nlocal) {
    #ifndef CONP_NO_DEBUG
    for (i = 0; i < localGA_num; i++) {
      iseq = GA_seq_arr[i];
      if (q[iseq] != q_store[iseq]) {
        error->one(FLERR, "Fix::Conp inconsistent charge for PCG method with the
  criteria of rel_B");
      }
    }
    #else
      i = rand() % localGA_num;
      iseq = GA_seq_arr[i];
      if (q[iseq] != q_store[iseq]) {
        error->one(FLERR, "Fix::Conp inconsistent charge for PCG method with the
  criteria of rel_B");
      }
    #endif
  }
  */

  q_store.reserve(nlocal);

  double* const q_store_arr = &q_store.front();
  double* const p_localGA   = &p_localGA_vec.front();
  double* const ap_localGA  = &ap_localGA_vec.front();
  double* const res_localGA = &res_localGA_vec.front();
  double* z_localGA         = &z_localGA_vec.front();
  double* z_localGA_prv     = &z_localGA_prv_vec.front();

  double* const firstneigh_inva_self     = &localGA_firstneigh_inva.front();
  double* const firstneigh_inva          = firstneigh_inva_self + localGA_num;
  const int* const firstneigh_seq        = &localGA_firstneigh.front();
  const int* const firstneigh_localIndex = &localGA_firstneigh_localIndex.front();
  // if (me == work_me && force->kspace != nullptr) utils::logmesg(lmp,
  // "self_potential = %6e\n", self_ECpotential);

  double* z_localGA_tmp; // * z_localGA_tmp;
  // memory->create(z_localGA_prv, z_localGA_vec.capacity(),
  // "fix::conpGA::z_localGA_prv");
  for (int i = 0; i < localGA_num; ++i) {
    z_localGA_prv[i] = 0.0;
  }

  for (i = 0; i < nlocal; ++i) {
    q_store_arr[i] = q[i];
    q[i]           = 0;
  }
  if (!GA2GA_flag) {
    for (i = 0; i < localGA_num; i++) {
      iseq = GA_seq_arr[i];
      qlocalGA_incr_sum -= q_store_arr[iseq];
      q_store_arr[iseq] = 0;
      B0_prev[i]        = 0;
    }
  }

  local_res_sum = 0;

  if (tol_style == TOL::REL_B) {
    PE_ij_local = 0;
    for (i = 0; i < localGA_num; i++) {

      iseq  = GA_seq_arr[i];
      value = ek_atom[iseq];

      q[iseq] = q_store_arr[iseq];

      // Copy the previous potential and shift it by the neutrality constraint
      // B0_next[i] = B0_prev[i] - Ushift;
      // the potential by the all electrolyte molecule \sum_J e_{iJ}
      B0_next[i] = value - B0_prev[i];

      // as the electric energy \sum_J e_{iJ}q_I * q_i
      PE_ij_local += qtmp * B0_next[i];

      cl_atom[iseq] = res_localGA[i] = localGA_Vext[i] - value;
      local_res_sum += res_localGA[i];
#ifndef CONP_NO_DEBUG
      if (pcg_out) {
        q_iters[0][i] = qtmp;
      }
#endif
      // if (me == 0) utils::logmesg(lmp, "Seq: %d, Charge %f, U_ext
      // %f\n", iseq, qtmp, res_localGA[i]);
    }
    MPI_Allreduce(&PE_ij_local, &PE_ij_total, 1, MPI_DOUBLE, MPI_SUM, world);
  } else {
    for (i = 0; i < localGA_num; i++) {

      iseq          = GA_seq_arr[i];
      value         = ek_atom[iseq];
      cl_atom[iseq] = res_localGA[i] = localGA_Vext[i] - value;
      local_res_sum += res_localGA[i];
#ifndef CONP_NO_DEBUG
      if (pcg_out) {
        q_iters[0][i] = qtmp;
      }
#endif
      // if (me == 0) utils::logmesg(lmp, "Seq: %d, Charge %f, U_ext
      // %f\n", iseq, qtmp, res_localGA[i]);
    }
  }
  MPI_Allreduce(&local_res_sum, &total_res_sum, 1, MPI_DOUBLE, MPI_SUM, world);
  Pshift_by_res = total_res_sum * pcg_shift;

  //  To communicate residual on the ghost atom
  fix_forward_comm(1);
  for (i = localGA_num; i < usedGA_num; i++) {
    iseq           = GA_seq_arr[i];
    res_localGA[i] = cl_atom[iseq];
  }

  neighbor->ago      = 0; // Set to re-check the atom charge quantity
  resnorm_local      = 0;
  next_resnorm_local = 0;
  ptap_local         = 0;
  pcg_Matrix_Residual(Pshift_by_res, z_localGA, firstneigh_inva_self, firstneigh_inva, firstneigh_localIndex, res_localGA);
  for (i = 0; i < localGA_num; i++) {
    p_localGA[i] = z_localGA[i];
    resnorm_local += z_localGA[i] * res_localGA[i];
    next_resnorm_local += res_localGA[i] * res_localGA[i];
    ptap_local += z_localGA[i] * z_localGA[i];
  }
  z_localGA_tmp = z_localGA;
  z_localGA     = z_localGA_prv;
  z_localGA_prv = z_localGA_tmp; // swap(k and k+1);

  MPI_Allreduce(&resnorm_local, &resnorm_total, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&next_resnorm_local, &next_resnorm_total, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&ptap_local, &ptap_total, 1, MPI_DOUBLE, MPI_SUM, world);

#ifndef CONP_NO_DEBUG
  if (me == work_me) {
    t_end = clock();
    utils::logmesg(lmp, "Iteration 0: z*r = {:20.18g}, r*r = {:20.18g}, z*z = {:20.18g}\n", resnorm_total, next_resnorm_total, ptap_total);
  }
#endif
/***************************************************************
                pcg_res_norm structure
[0] resnorm_total       = z * res
[1] next_resnorm_total  = res * res
[2] ptap_total          = z * z
[3] alpha               = resnorm_total/ptap_total
[4] beta                = next_resnorm_total/resnorm_total;
************************************************************* */
#ifndef CONP_NO_DEBUG
  if (pcg_out != nullptr) {
    pcg_res_norm[0][0] = resnorm_total;
    pcg_res_norm[0][1] = next_resnorm_total;
    pcg_res_norm[0][2] = ptap_total;
    pcg_res_norm[0][3] = resnorm_total / ptap_total;
    pcg_res_norm[0][4] = next_resnorm_total / resnorm_total;
  }
#endif

  for (iter = 1; iter < maxiter; iter++) {
    // ----------- Calculate ap = AAA*p update_charge_PCG[] -----------------
    for (i = 0; i < localGA_num; i++) {
      iseq    = GA_seq_arr[i];
      q[iseq] = p_localGA[i];
    }

    time_t t_begin = clock();
    fix_forward_comm(0);
    electro->pair_GAtoGA();
    update_P_time += double(clock() - t_begin) / CLOCKS_PER_SEC;

    time_t t_begin0 = clock();
    electro->compute_GAtoGA();
    electro->calc_GA_potential(STAT::E, STAT::E, SUM_OP::BOTH);
    update_K_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;

    ptap_local = 0;
    for (i = 0; i < localGA_num; i++) {
      iseq = GA_seq_arr[i];
      // value = pair_ecatom[iseq] + kspace_eatom[iseq];
      ptap_local += cl_atom[iseq];
      // qtmp = q[iseq];
      // value /= qtmp;

      ap_localGA[i] = ek_atom[iseq];
      // ptap_local += value * qtmp;
    }
    MPI_Allreduce(&ptap_local, &ptap_total, 1, MPI_DOUBLE, MPI_SUM, world);

    // ----------- End of Calculate ap = AAA*p -----------------

    // ----------- update r and q by r' = r - d*ap; q' = q + d*p;
    // -----------------
    alpha = resnorm_total / ptap_total;
    // printf("alpha = %f, resnorm_total = %f, ptap_total = %f\n", alpha,
    // resnorm_total, ptap_total);
    local_res_sum = 0;
    if (tol_style == TOL::REL_B) {
      PE_ij_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
        PE_ij_local += qGA_incr * B0_next[i];

#ifndef CONP_NO_DEBUG
        if (pcg_out) {
          q_iters[iter][i] = q_store_arr[iseq];
        }
#endif
        qlocalGA_incr_sum += qGA_incr;
        tmp           = alpha * ap_localGA[i];
        cl_atom[iseq] = res_localGA[i] -= tmp;
        B0_prev[i] += tmp;
        local_res_sum += res_localGA[i];
      }

      MPI_Allreduce(&PE_ij_local, &dPE_ij_total, 1, MPI_DOUBLE, MPI_SUM, world);
      PE_ij_total += dPE_ij_total;

      if (fabs(dPE_ij_total / PE_ij_total) < tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} rel_B0 * q = {:g} netcharge = "
                         "{:g} Total Elps. {:8.2f}[s]\n",
                         iter, dPE_ij_total / PE_ij_total, qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      } // if
    } else if (tol_style == TOL::RES || tol_style == TOL::PE) {
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
#ifndef CONP_NO_DEBUG
        if (pcg_out) {
          q_iters[iter][i] = q_store_arr[iseq];
        }
#endif
        qlocalGA_incr_sum += qGA_incr;
        cl_atom[iseq] = res_localGA[i] -= alpha * ap_localGA[i];
        local_res_sum += res_localGA[i];
      }
    } else if (tol_style == TOL::MAX_Q) {
      q_max_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
#ifndef CONP_NO_DEBUG
        if (pcg_out) {
          q_iters[iter][i] = q_store_arr[iseq];
        }
#endif
        qlocalGA_incr_sum += qGA_incr;
        cl_atom[iseq] = res_localGA[i] -= alpha * ap_localGA[i];
        local_res_sum += res_localGA[i];

        tmp = fabs(qGA_incr / q_store_arr[iseq]);
        if (tmp > q_max_local) {
          q_max_local = tmp;
        }
        // qGA_incr_sqr += tmp * tmp;
      }
      MPI_Allreduce(&q_max_local, &q_max_total, 1, MPI_DOUBLE, MPI_MAX, world);

      if (q_max_total < tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} max_Q = {:g} netcharge = {:g} "
                         "Total Elps. {:8.2f}[s]\n",
                         iter, q_max_total, qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      } // if
    } else if (tol_style == TOL::STD_Q) {
      q_std_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
#ifndef CONP_NO_DEBUG
        if (pcg_out) {
          q_iters[iter][i] = q_store_arr[iseq];
        }
#endif
        qlocalGA_incr_sum += qGA_incr;
        cl_atom[iseq] = res_localGA[i] -= alpha * ap_localGA[i];
        local_res_sum += res_localGA[i];

        tmp = qGA_incr / q_store_arr[iseq];
        q_std_local += tmp * tmp;
      }
      MPI_Allreduce(&q_std_local, &q_std_total, 1, MPI_DOUBLE, MPI_SUM, world);
      if (q_std_total < totalGA_num * tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} std_Q = {:g} netcharge = {:g} "
                         "Total Elps. {:8.2f}[s]\n",
                         iter, sqrt(q_std_total / totalGA_num), qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      } // if
      // TOL::MAX_Q
    } else if (tol_style == TOL::STD_RQ) {
      q_std_local = q_sqr_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
#ifndef CONP_NO_DEBUG
        if (pcg_out) {
          q_iters[iter][i] = q_store_arr[iseq];
        }
#endif
        qlocalGA_incr_sum += qGA_incr;
        cl_atom[iseq] = res_localGA[i] -= alpha * ap_localGA[i];
        local_res_sum += res_localGA[i];

        tmp = q_store_arr[iseq];
        q_std_local += qGA_incr * qGA_incr;
        q_sqr_local += tmp * tmp;
      }

      MPI_Allreduce(&q_std_local, &q_std_total, 1, MPI_DOUBLE, MPI_SUM, world);
      MPI_Allreduce(&q_sqr_local, &q_sqr_total, 1, MPI_DOUBLE, MPI_SUM, world);

      if (q_std_total < q_sqr_total * tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} std_RQ = {:g} netcharge = {:g}\n"
                         "Total Elps. {:8.2f}[s]\n",
                         iter, sqrt(q_std_total / q_sqr_total), qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      }
    } else if (tol_style == TOL::ABS_Q) {
      q_std_local = q_sqr_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
#ifndef CONP_NO_DEBUG
        if (pcg_out) {
          q_iters[iter][i] = q_store_arr[iseq];
        }
#endif
        qlocalGA_incr_sum += qGA_incr;
        cl_atom[iseq] = res_localGA[i] -= alpha * ap_localGA[i];
        local_res_sum += res_localGA[i];

        q_std_local += fabs(qGA_incr);
        q_sqr_local += fabs(q_store_arr[iseq]);
      }

      MPI_Allreduce(&q_std_local, &q_std_total, 1, MPI_DOUBLE, MPI_SUM, world);
      MPI_Allreduce(&q_sqr_local, &q_sqr_total, 1, MPI_DOUBLE, MPI_SUM, world);

      if (q_std_total < q_sqr_total * tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} abs_RQ = {:g} netcharge = {:g} "
                         "Total Elps. {:8.2}[s]\n",
                         iter, q_std_total / q_sqr_total, qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      } // if
    }

    if (tol_style == TOL::PE) {
      if (fabs(ptap_total) < totalGA_num * tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} delta_PE = {:g} netcharge = "
                         "{:g} Total Elps. {:8.2f}[s]\n",
                         iter, ptap_total / totalGA_num, qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      }
    }

    MPI_Allreduce(&local_res_sum, &total_res_sum, 1, MPI_DOUBLE, MPI_SUM, world);
    Pshift_by_res = total_res_sum * pcg_shift;

    //  To communicate residual on the ghost atom
    fix_forward_comm(1);
    for (i = localGA_num; i < usedGA_num; i++) {
      iseq           = GA_seq_arr[i];
      res_localGA[i] = cl_atom[iseq];
#ifndef CONP_NO_DEBUG
      if (me == work_me && DEBUG_LOG_LEVEL > 1) utils::logmesg(lmp, "res_localGA: {} {} {} {:f}\n", i, iseq, tag[iseq], cl_atom[iseq]);
#endif
    }
    // ----------- End of update r and q; -----------------

    // ----------- update z using z' = M*r -----------------
    pcg_Matrix_Residual(Pshift_by_res, z_localGA, firstneigh_inva_self, firstneigh_inva, firstneigh_localIndex, res_localGA);
    next_resnorm_local = 0; // next_Polak_resnorm_local
    for (i = 0; i < localGA_num; i++) {
      next_resnorm_local += res_localGA[i] * z_localGA[i];
      // next_Polak_resnorm_local += res_localGA[i]*z_localGA_prv[i];
    }
    MPI_Allreduce(&next_resnorm_local, &next_resnorm_total, 1, MPI_DOUBLE, MPI_SUM, world);
    // MPI_Allreduce(&next_Polak_resnorm_local, &next_Polak_resnorm_total, 1,
    // MPI_DOUBLE, MPI_SUM, world);

    if (tol_style == TOL::RES) {
      if (fabs(next_resnorm_total) < totalGA_num * tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} res = {:g} netcharge = {:g} "
                         "Total Elps. {:8.2f}[s]\n",
                         iter, sqrt(next_resnorm_total / totalGA_num), qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      }
    }

    beta = (next_resnorm_total) / resnorm_total;

#ifndef CONP_NO_DEBUG
    if (pcg_out != nullptr) {
      pcg_res_norm[iter][0] = resnorm_total;
      pcg_res_norm[iter][1] = next_resnorm_total;
      pcg_res_norm[iter][2] = ptap_total;
      pcg_res_norm[iter][3] = alpha;
      pcg_res_norm[iter][4] = beta;
    }
#endif

    // ----------- update p using p_{k} = r_{k} + rho_{k}/rho_{k-1} * p_{k-1} ;
    // -----------------
    for (i = 0; i < localGA_num; i++) {
      // p_localGA[i] = accelerate_factor*res_localGA[i] + beta * p_localGA[i];
      p_localGA[i] = z_localGA[i] + beta * p_localGA[i];
    }

    resnorm_total = next_resnorm_total;
#ifndef CONP_NO_DEBUG
    if (me == work_me) {
      t_end = clock();
      utils::logmesg(lmp,
                     "Iteration {} delta_PE = {:g} res = {:g} q_max = {:g} q_std = {:g} rq_std = "
                     "{:g} rel_B0 * q = {:g} alpha = {:g} beta = {:g} Elps. {:8.2f}[s]\n",
                     iter, ptap_total / totalGA_num, sqrt(fabs(resnorm_total / totalGA_num)), q_max_total, sqrt(q_std_total / totalGA_num),
                     sqrt(q_std_total / q_sqr_total), dPE_ij_total / PE_ij_total, alpha, beta, double(t_end - t_begin) / CLOCKS_PER_SEC);
    }
#endif
    neighbor->ago++;

    z_localGA_tmp = z_localGA;
    z_localGA     = z_localGA_prv;
    z_localGA_prv = z_localGA_tmp; // swap(k and k+1);

  } // iter
  curriter = iter;

#ifndef CONP_NO_DEBUG
  if (pcg_out != nullptr) {
    pcg_analysis();
  }
#endif

  neighbor->ago = neighbor_ago_store; // 0; // neighbor_ago_store;
  if (postreat_flag) {
    // PCG
    MPI_Allreduce(&qlocalGA_incr_sum, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, world);
    // localGA_charge_neutrality(qGA_incr_sum + qsum_store, q_store_arr);
    double* Ele_qsum_ptr = Ele_qsum;
    if (calc_EleQ_flag) {
      set_zeros(Ele_qsum, 0, Ele_num);
      for (int i = 0; i < localGA_num; i++) {
        int iEle = localGA_eleIndex[i];
        int iseq = GA_seq_arr[i];
        Ele_qsum[iEle] += q_store_arr[iseq];
      }
    } else {
      Ele_qsum_ptr = nullptr;
    }
    POST2ND_charge(qGA_incr_sum + qsum_store, Ele_qsum_ptr, q_store_arr);
    /*
    if (tol_style == TOL::REL_B) {
      for (i = 0; i < localGA_num; i++) {
          iseq = GA_seq_arr[i];
          B0_next[i] -= value;
      }
    }
    */
  }
  std::copy(q_store_arr, q_store_arr + nlocal, q);
  // double *B_tmp; B_tmp = B0_next; B0_next = B0_prev; B0_prev = B_tmp;

  reset_nonzero();
  // For the update of pppm/cg by 3001
  if (neutrality_state == true) {
    electro->set_qsqsum(0.0, 0.0);
  } else {
    electro->qsum_qsq();
  }
  electro->update_qstat();

  fix_forward_comm(0);

  time_t t_end  = clock();
  update_Q_time = double(t_end - t_begin) / CLOCKS_PER_SEC;
  update_Q_totaltime += update_Q_time;
} // update_ch_arge_pcg

/* -------------------------------------------------------------------------------
  Analysis the convergent rate on PCG progress
 -------------------------------------------------------------------------------
 */
void FixConpGA::pcg_analysis()
{
  int i, j, k, iter;
  double resnorm_total, next_resnorm_total, ptap_total, alpha, beta;
  double qd, qd_scale, *q_ref = q_iters[curriter], qd_total, qd_scale_total, tmp, q_sqr, q_sqr_total;

  if (me == 0 && pcg_out) fprintf(pcg_out, "# MD iter at %lld\n", update->ntimestep);

  for (iter = 0; iter < curriter; ++iter) {
    resnorm_total      = pcg_res_norm[iter][0];
    next_resnorm_total = pcg_res_norm[iter][1];
    ptap_total         = pcg_res_norm[iter][2];
    alpha              = pcg_res_norm[iter][3];
    beta               = pcg_res_norm[iter][4];

    resnorm_total      = sqrt(resnorm_total / totalGA_num);
    next_resnorm_total = sqrt(next_resnorm_total / totalGA_num);
    ptap_total         = sqrt(ptap_total / totalGA_num);

    qd = qd_scale = q_sqr = 0;
    for (i = 0; i < localGA_num; ++i) {
      tmp = q_iters[iter][i] - q_ref[i];
      qd += tmp * tmp;

      tmp = 1 - q_iters[iter][i] / q_ref[i];
      qd_scale += tmp * tmp;
      q_sqr += q_iters[iter][i] * q_iters[iter][i];
    }
    MPI_Allreduce(&qd, &qd_total, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&qd_scale, &qd_scale_total, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&q_sqr, &q_sqr_total, 1, MPI_DOUBLE, MPI_SUM, world);

    qd_total       = sqrt(qd_total / totalGA_num);
    qd_scale_total = sqrt(qd_scale_total / totalGA_num);
    q_sqr_total    = sqrt(q_sqr / totalGA_num);

    if (me == 0 && pcg_out) {
      fprintf(pcg_out, "%3d %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n", iter, resnorm_total, next_resnorm_total, ptap_total, alpha, beta, qd_total,
              qd_scale_total, q_sqr_total);
    }
  }
}

/* -------------------------------------------------------------------------------
  To calculate the residual by the mulitplication with precondition matrix
 -------------------------------------------------------------------------------
 */
void FixConpGA::pcg_Matrix_Residual(double z_shift, double* const z, const double* const Mdiag, const double* const M, const int* const firstneigh_localIndex,
                                    const double* const r)
{
  int iseq, i_index, jj, j_index;
  int* GA_seq_arr = &localGA_seq.front();
  int ntotal      = atom->nlocal + atom->nghost;
  double* buf     = electro->cl_atom();
  double value;
  for (i_index = 0; i_index < localGA_num; i_index++) {
    z[i_index] = z_shift;
  }
  for (i_index = localGA_num; i_index < usedGA_num; i_index++) {
    z[i_index] = 0;
  }

  // Be careful z[i] = M[jj]*r[j] and z[j] = M[jj]*r[i]
  for (i_index = 0; i_index < localGA_num; i_index++) {
    for (jj = localGA_numneigh_disp[i_index]; jj < localGA_numneigh_disp[i_index] + localGA_numlocalneigh[i_index]; jj++) {
      j_index = firstneigh_localIndex[jj];
      z[i_index] += M[jj] * r[j_index];
      z[j_index] += M[jj] * r[i_index];
    }
    for (jj = localGA_numneigh_disp[i_index] + localGA_numlocalneigh[i_index]; jj < localGA_numneigh_disp[i_index] + localGA_numneigh[i_index]; jj++) {
      // jtag = firstneigh_seq[jj];
      // jseq = atom->map(jtag);
      j_index = firstneigh_localIndex[jj];
      z[i_index] += M[jj] * r[j_index];
      // for newton_pair == on, we have j_index > localGA_num and the value of
      // z[j_index] value of z[j_index] should be send to others;
      if (newton_pair) // Update the ghost atoms
        z[j_index] += M[jj] * r[i_index];
    }
    z[i_index] += Mdiag[i_index] * r[i_index];
  }
  // send_pack the z values
  if (newton_pair) {
    clear_atomvec(buf, extGA_num);
    /*
    for (iseq = 0; iseq < ntotal; iseq++) {
      buf[iseq] = 0;
    }
    */
    for (i_index = localGA_num; i_index < usedGA_num; i_index++) {
      iseq      = GA_seq_arr[i_index];
      buf[iseq] = z[i_index];
    }
    fix_reverse_comm(REV_COUL);
    for (i_index = 0; i_index < localGA_num; i_index++) {
      iseq = GA_seq_arr[i_index];
      z[i_index] += buf[iseq];
    }
  }

  if (me == work_me && DEBUG_LOG_LEVEL > 1) {
    printf("Mdiag = ");
    for (i_index = 0; i_index < localGA_num; i_index++) {
      printf("%20.18g ", Mdiag[i_index]);
    }
    printf("\nM = ");
    for (i_index = 0; i_index < localGA_firstneigh.capacity(); i_index++) {
      printf("%20.18g ", M[i_index]);
    }
    printf("\n");
    printf("r = ");
    for (i_index = 0; i_index < usedGA_num; i_index++) {
      printf("%20.18f ", r[i_index]);
    }
    printf("\n");
    printf("z = ");
    for (i_index = 0; i_index < localGA_num; i_index++) {
      printf("%20.18f ", z[i_index]);
    }
    printf("\n\n");
  }
}
/* ----------------------------------------------------------------------------------------
Update the charge on each eletrode atom only accouting the near field
---------------------------------------------------------------------------------------- */
void FixConpGA::update_charge_near()
{
  FUNC_MACRO(0);
  update_charge_cg(false);
}

/* ----------------------------------------------------------------------------------------
  Update the atomic charge on each eletrode atom based on the conjugate gradient
method (CG)
----------------------------------------------------------------------------------------
*/
void FixConpGA::update_charge_cg(bool far_flag)
{
  // work_me = 0;
  FUNC_MACRO(0);
  int i, j, ii, iter;
  int nlocal = atom->nlocal, *type = atom->type, *tag = atom->tag;
  int* GA_seq_arr = &localGA_seq.front();
  int* GA_seq_end = &localGA_seq.front() + localGA_num;
  int iseq;
  double value, qtmp, alpha, beta, *q = atom->q, tmp;
  double q_max_local, q_max_total = 0;
  double q_std_local, q_std_total = 0;
  double q_sqr_local, q_sqr_total = 0;
  double PE_ij_local, PE_ij_total = 0, dPE_ij_total;

  double resnorm_local, resnorm_total, next_resnorm_local, next_resnorm_total, ptap_local, ptap_total;
  int neighbor_ago_store = neighbor->ago;
  double qsum_store, qlocalGA_incr_sum = 0, qGA_incr_sum, qGA_incr,
                     qGA_incr_sqr; // qGA_incr_sqr_total;
  double* const cl_atom = electro->cl_atom();
  double* const ek_atom = electro->ek_atom();
  std::vector<double> virtual_kspace_eatom;

  t_begin = clock();

  if (neutrality_flag == true) {
    if (neutrality_state == false) {
      qsum_store = electro->calc_qsum();
    } else {
#ifndef CONP_NO_DEBUG
      qsum_store = electro->calc_qsum();
      if (fabs(qsum_store) > 1e-11)
        if (me == 0) utils::logmesg(lmp, "invalid neutrality state of qsum = %.10f\n", qsum_store);
#endif
    }
  }

  q_store.reserve(nlocal);
  double* const q_store_arr = &q_store.front();
  double* const p_localGA   = &p_localGA_vec.front();
  double* const ap_localGA  = &ap_localGA_vec.front();
  double* const res_localGA = &res_localGA_vec.front();
  // double accelerate_factor = 1/self_GreenCoef[1];
  // printf("accelerate_factor: %f\n", accelerate_factor);

  // if (me == work_me && force->kspace != nullptr) utils::logmesg(lmp,
  // "self_potential = %6e\n", self_ECpotential);

  // double qsum_store = 0;
  for (i = 0; i < nlocal; ++i) {
    q_store_arr[i] = q[i];
    // qsum_store += q[i];
    q[i] = 0;
  }
  // No GA2GA interaction starts from zero;
  if (!GA2GA_flag) {
    for (i = 0; i < localGA_num; i++) {
      iseq = GA_seq_arr[i];
      qlocalGA_incr_sum -= q_store_arr[iseq];
      q_store_arr[iseq] = 0;
      B0_prev[i]        = 0;
    }
  }

  resnorm_local = 0;
  if (tol_style == TOL::REL_B) {
    PE_ij_local = 0;
    for (i = 0; i < localGA_num; i++) {
      iseq  = GA_seq_arr[i];
      value = ek_atom[iseq];

      // q[iseq] = q_store_arr[iseq];
      // qtmp = q[iseq];
      // value /= qtmp; // as potential by both electrolyte and electrode

      // Copy the previous potential and shift it by the neutrality constraint
      // B0_prev[i] - Ushift;
      // the potential by the all electrolyte molecule \sum_J e_{iJ}
      B0_next[i] = value - B0_prev[i];
      // printf("Q[%d] = %.12g; B0 = %.12g\n", i, q_store_arr[i], B0_prev[i]);
      // B0_prev[i] = value;
      // as the electric energy \sum_J e_{iJ}q_I * q_i
      PE_ij_local += qtmp * B0_next[i];

      res_localGA[i] = localGA_Vext[i] - value;
      if (res_localGA[i] == 0) res_localGA[i] = std::numeric_limits<double>::epsilon();

      p_localGA[i] = res_localGA[i];
      resnorm_local += res_localGA[i] * res_localGA[i];
    }
    MPI_Allreduce(&PE_ij_local, &PE_ij_total, 1, MPI_DOUBLE, MPI_SUM, world);
  } else {
    for (i = 0; i < localGA_num; i++) {
      iseq           = GA_seq_arr[i];
      value          = ek_atom[iseq];
      res_localGA[i] = localGA_Vext[i] - value;
      if (res_localGA[i] == 0) res_localGA[i] = std::numeric_limits<double>::epsilon();
      p_localGA[i] = res_localGA[i];
      resnorm_local += res_localGA[i] * res_localGA[i];
    }
  }

  neighbor->ago = 0; // Set to re-check the atom charge quantity
  /*
  fix_pair_compute();
  if (force->kspace) {
    force->kspace->qsum_qsq();
    force->kspace->compute(999,0);
  }

  // Calculate the electric potential interaction between electrode atoms
  qsum_qsq();
  self_potential = Xi_self * qsum;
  resnorm_local = 0;
  for (i = 0; i < localGA_num; i++)
  {
    ilocalseq = GA_seq_arr[i];
    value = pair_ecatom[ilocalseq] + kspace_eatom[ilocalseq];
    qtmp = q[ilocalseq] + 1e-20;
    value /= qtmp;
    value += self_potential;
    q[i] = q_store_arr[i];
    res_localGA[i] = p_localGA[i] -= value;
    resnorm_local += res_localGA[i] * res_localGA[i];
    if (me == 0) utils::logmesg(lmp, "Seq: %d, Charge %f, U_ext %f",
  ilocalseq, qtmp, res_localGA[i]);
  }
  */
  MPI_Allreduce(&resnorm_local, &resnorm_total, 1, MPI_DOUBLE, MPI_SUM, world);
  if (DEBUG_LOG_LEVEL > 0) {
    utils::logmesg(lmp, "Iteration 0: r*r = {:20.18g}\n", resnorm_total);
  }
  for (iter = 0; iter < maxiter; iter++) {
    // ----------- Calculate ap = AAA*p from update_charge_cg[]
    // -----------------
    for (i = 0; i < localGA_num; i++) {
      iseq    = GA_seq_arr[i];
      q[iseq] = p_localGA[i];
    }

    time_t t_begin = clock();
    fix_forward_comm(0);
    electro->pair_GAtoGA();
    update_P_time += double(clock() - t_begin) / CLOCKS_PER_SEC;

    if (1 || !far_flag) {
      t_begin = clock();
      // calculation in kspace;
      electro->compute_GAtoGA();
      // set_zeros(&ek_atom[0], 0, localGA_num);
      electro->calc_GA_potential(STAT::E, STAT::E, SUM_OP::BOTH);
      update_K_time += double(clock() - t_begin) / CLOCKS_PER_SEC;
    } else {
      set_zeros(&ek_atom[0], 0, localGA_num);
      electro->calc_GA_potential(STAT::E, STAT::E, SUM_OP::BOTH);
      update_K_time += double(clock() - t_begin) / CLOCKS_PER_SEC;
    }

    ptap_local = 0;
    for (i = 0; i < localGA_num; i++) {
      iseq = GA_seq_arr[i];
      // value = pair_ecatom[iseq] + kspace_eatom[iseq];
      // qtmp = q[iseq];
      ptap_local += cl_atom[iseq];
      // value /= qtmp;
      // value += self_ECpotential;
      ap_localGA[i] = ek_atom[iseq];
      // ptap_local += value * qtmp;
    }
    MPI_Allreduce(&ptap_local, &ptap_total, 1, MPI_DOUBLE, MPI_SUM, world);
    // ----------- End of Calculate ap = AAA*p -----------------

    // ----------- update r and q by r' = r - d*ap; q' = q + d*p;
    // -----------------
    alpha              = resnorm_total / ptap_total;
    next_resnorm_local = 0;

    // qGA_incr_sqr = 0;
    if (tol_style == TOL::REL_B) {
      PE_ij_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
        qlocalGA_incr_sum += qGA_incr;
        PE_ij_local += qGA_incr * B0_next[i];

        tmp = alpha * ap_localGA[i];
        B0_prev[i] += tmp;
        res_localGA[i] -= tmp;
        next_resnorm_local += res_localGA[i] * res_localGA[i];
      }

      MPI_Allreduce(&PE_ij_local, &dPE_ij_total, 1, MPI_DOUBLE, MPI_SUM, world);
      PE_ij_total += dPE_ij_total;

      if (fabs(dPE_ij_total / PE_ij_total) < tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} rel_B0 * q = {:g} netcharge = "
                         "{:g} Total Elps. {:8.2f}[s]\n",
                         iter, dPE_ij_total / PE_ij_total, qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      } // if

    } else if (tol_style == TOL::RES) {
      printf("good\n");
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;
        qlocalGA_incr_sum += qGA_incr;
        res_localGA[i] -= alpha * ap_localGA[i];
        next_resnorm_local += res_localGA[i] * res_localGA[i];
      }
      printf("good2\n");
    } else if (tol_style == TOL::MAX_Q) {
      q_max_local = 0;
      for (i = 0; i < localGA_num; i++) {
        iseq     = GA_seq_arr[i];
        qGA_incr = alpha * p_localGA[i];
        q_store_arr[iseq] += qGA_incr;

        qlocalGA_incr_sum += qGA_incr;
        res_localGA[i] -= alpha * ap_localGA[i];
        next_resnorm_local += res_localGA[i] * res_localGA[i];

        tmp = fabs(qGA_incr / q_store_arr[iseq]);
        if (tmp > q_max_local) {
          q_max_local = tmp;
        }
      }

      MPI_Allreduce(&q_max_local, &q_max_total, 1, MPI_DOUBLE, MPI_MAX, world);

      if (q_max_total < tolerance) {
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == 0 && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "***** Converged at iteration {} max_Q = {:g} netcharge = {:g} "
                         "Total Elps. {:8.2f}[s]\n",
                         iter, q_max_total, qGA_incr_sum + qsum_store, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      } // if
    }

    // next_resnorm_local *= accelerate_factor;
    MPI_Allreduce(&next_resnorm_local, &next_resnorm_total, 1, MPI_DOUBLE, MPI_SUM, world);
    // MPI_Allreduce(&qGA_incr_sqr,       &qGA_incr_sqr_total, 1, MPI_DOUBLE,
    // MPI_SUM, world);

#ifndef CONP_NO_DEBUG
    if (me == work_me && DEBUG_LOG_LEVEL > 0) {
      t_end = clock();
      utils::logmesg(lmp,
                     "iter {} delta_PE = {:g} res = {:g} rel_B0 * q = {:g} netcharge = {:g} "
                     "Total Elps. {:8.2f}[s]\n",
                     iter, ptap_total / totalGA_num, sqrt(next_resnorm_total / totalGA_num), dPE_ij_total / PE_ij_total, qGA_incr_sum + qsum_store,
                     double(t_end - t_begin) / CLOCKS_PER_SEC);
    }
#endif

    // if (fabs(next_resnorm_total/totalGA_num) < tolerance) {
    if (tol_style == TOL::RES) {
      if (next_resnorm_total < totalGA_num * tolerance) {
        // qsum_qsq(q_store_arr);
        qGA_incr = qlocalGA_incr_sum;
        MPI_Reduce(&qGA_incr, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
#ifndef CONP_NO_DEBUG
        if (me == work_me && DEBUG_LOG_LEVEL > 0) {
          t_end = clock();
          utils::logmesg(lmp,
                         "Iteration {} delta_PE = {:g} res = {:g} q_max = {:g} q_std = {:g} "
                         "rq_std = {:g} rel_B0 * q = {:g} alpha = {:g} beta = {:g} Elps. "
                         "{:8.2f}[s]\n",
                         iter, ptap_total / totalGA_num, sqrt(fabs(resnorm_total / totalGA_num)), q_max_total, sqrt(q_std_total / totalGA_num),
                         sqrt(q_std_total / q_sqr_total), dPE_ij_total / PE_ij_total, alpha, beta, double(t_end - t_begin) / CLOCKS_PER_SEC);
        }
#endif
        break;
      }
    }
    // ----------- End of update r and q; -----------------

    // ----------- update p using p_{k} = r_{k} + rho_{k}/rho_{k-1} * p_{k-1} ;
    // -----------------
    beta = next_resnorm_total / resnorm_total;
    for (i = 0; i < localGA_num; i++) {
      // p_localGA[i] = accelerate_factor*res_localGA[i] + beta * p_localGA[i];
      p_localGA[i] = res_localGA[i] + beta * p_localGA[i];
    }
    resnorm_total = next_resnorm_total;
    /*
    #ifndef CONP_NO_DEBUG
    if (me == work_me && DEBUG_LOG_LEVEL > 0) {
      t_end = clock();
      utils::logmesg(lmp,"Iteration %d: res = {:g}, ptap = {:g}, alpha = {:g}, Elps.
    %8.2f[s]\n", iter, resnorm_total, ptap_total, alpha, double(t_end - t_begin)
    / CLOCKS_PER_SEC);
    }
    #endif
    */
    neighbor->ago++;
  } // iter
  curriter = iter;

  neighbor->ago = neighbor_ago_store; // 0; // neighbor_ago_store;
  if (postreat_flag) {
    MPI_Allreduce(&qlocalGA_incr_sum, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM, world);
    double* Ele_qsum_ptr = Ele_qsum;
    if (calc_EleQ_flag) {
      set_zeros(Ele_qsum, 0, Ele_num);
      for (int i = 0; i < localGA_num; i++) {
        int iEle = localGA_eleIndex[i];
        int iseq = GA_seq_arr[i];
        Ele_qsum[iEle] += q_store_arr[iseq];
      }
    } else {
      Ele_qsum_ptr = nullptr;
    }
    // localGA_charge_neutrality(qGA_incr_sum + qsum_store, q_store_arr);
    // printf("qGA_incr_sum = %.10g; qsum_store = %.10g\n", qGA_incr_sum, qsum_store);
    POST2ND_charge(qGA_incr_sum + qsum_store, Ele_qsum_ptr, q_store_arr);
  }
  std::copy(q_store_arr, q_store_arr + nlocal, q);

  reset_nonzero();
  if (neutrality_state == true) {
    electro->set_qsqsum(0.0, 0.0);
  } else {
    electro->qsum_qsq();
  }
  // electro->update_qstat();

  fix_forward_comm(0);

  time_t t_end  = clock();
  update_Q_time = double(t_end - t_begin) / CLOCKS_PER_SEC;
  update_Q_totaltime += update_Q_time;
}
/* ---------------------------------------------------------------------------------------
                reset all zero charge into epsilon value
----------------------------------------------------------------------------------------
*/
int FixConpGA::reset_nonzero()
{
  FUNC_MACRO(0);
  double* q                   = atom->q;
  const int* const GA_seq_arr = &localGA_seq.front();
  const double pos_eps        = std::numeric_limits<double>::epsilon();
  const double neg_eps        = -pos_eps;

  for (size_t i = 0; i < localGA_num; i++) {
    int iseq = GA_seq_arr[i];
    q[iseq] += pos_eps;
    // q[GA_seq_arr[i]] = i%2 == 0 ? 1.0e-16 : -1.0e-16;
  }
  return 0;
}

/* ------------------------------------------------------------------------------------------
    Global switch of the 2nd post-treatment STEP
------------------------------------------------------------------------------------------
*/
double FixConpGA::POST2ND_charge(double value, double* Ele_qsum0, double* q)
{
  FUNC_MACRO(0);
  switch (Ele_Cell_Model) {
    case CELL::POT:
      if (Ele_qsum0 != nullptr) {
        double qlocal_charge = 0;
        for (int i = 0; i < Ele_num; i++) {
          qlocal_charge += Ele_qsum0[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &qlocal_charge, 1, MPI_DOUBLE, MPI_SUM, world);
#ifndef CONP_NO_DEBUG
        if (DEBUG_LOG_LEVEL > 0) printf("me = %d; value = %.10g, qlocal = %.10g\n", me, value, qlocal_charge);
#endif
        return POST2ND_localGA_neutrality(value + qlocal_charge, q);
      } else {
#ifndef CONP_NO_DEBUG
        if (DEBUG_LOG_LEVEL > 0) printf("me = %d; value = %.10g\n", me, value);
#endif
        return POST2ND_localGA_neutrality(value, q);
      }
      break;
    case CELL::Q:
      return POST2ND_solve_ele_matrix(Ele_qsum0, q);
    default:
      error->all(FLERR, "There is not a valid post treatment step for current "
                        "CELL configuration\n");
      break;
  }
}
/* ------------------------------------------------------------------------------------------
    Update the charge on the electrode under the charge conservation
------------------------------------------------------------------------------------------
*/
double FixConpGA::POST2ND_solve_ele_matrix(double* Ele_qtotal, double* q)
{
  FUNC_MACRO(0);
  // printf("POST2ND_solve_ele_matrix start = \n");
  const int* const GA_seq_arr = &localGA_seq.front();
  int n                       = atom->ntypes;
  MPI_Allreduce(MPI_IN_PLACE, Ele_qtotal, Ele_num, MPI_DOUBLE, MPI_SUM, world);
  double dq[Ele_num], dU[Ele_num];
  set_zeros(dU, 0, Ele_num);
  for (int i = 0; i < Ele_num; i++) {
    int iType = Ele_To_aType[i];
    double value;
    // printf("i = %d, iType = %d, GA_Vext_var = %f\n", i, iType,
    // GA_Vext_var[iType]);
    value = dq[i] = GA_Vext_var[iType] - Ele_qtotal[i];
    for (int j = 0; j < Ele_num; j++) {
      dU[j] += Ele_matrix[i][j] * value;
    }
  }
  /*
  print_vec(Ele_matrix[0], Ele_num*Ele_num, "qd ");
  print_vec(dq, Ele_num, "dq ");
  print_vec(dU, Ele_num, "dU ");
  print_vec(Ele_qtotal, Ele_num, "Ele_qtotal ");
  */
  double dU0, dU1, dU2, dU3, dU4;
  double* ptr = &Ele_D_matrix[0][0];
  for (int i = 0; i < Ele_num; i++) {
    Ele_Uchem[i] += dU[i];
  }
  switch (Ele_num) {
    case 0:
      error->all(FLERR, "Cannot implement charge conservation calculation with zero electrode");
      break;
    case 1:
      dU0 = dU[0];
      if (B0_prev) {
        for (int i = 0; i < localGA_num; i++) {
          double delta_q;
          delta_q  = *(ptr++) * dU0;
          int iseq = GA_seq_arr[i];
          q[iseq] += delta_q;
          double delta_U = dU[localGA_eleIndex[i]];
          localGA_Vext[i] += delta_U;
          B0_prev[i] += delta_U;
        }
      } else {
        for (int i = 0; i < localGA_num; i++) {
          double delta_q;
          delta_q  = *(ptr++) * dU0;
          int iseq = GA_seq_arr[i];
          q[iseq] += delta_q;
          localGA_Vext[i] += dU[localGA_eleIndex[i]];
        }
      }
      break;
    case 2:
      dU0 = dU[0];
      dU1 = dU[1];
      if (B0_prev) {
        for (int i = 0; i < localGA_num; i++) {
          double delta_q;
          delta_q = *(ptr++) * dU0;
          delta_q += *(ptr++) * dU1;
          int iseq = GA_seq_arr[i];
          q[iseq] += delta_q;
          double delta_U = dU[localGA_eleIndex[i]];
          localGA_Vext[i] += delta_U;
          B0_prev[i] += delta_U;
        }
      } else {
        for (int i = 0; i < localGA_num; i++) {
          double delta_q;
          delta_q = *(ptr++) * dU0;
          delta_q += *(ptr++) * dU1;
          int iseq = GA_seq_arr[i];
          q[iseq] += delta_q;
          localGA_Vext[i] += dU[localGA_eleIndex[i]];
        }
      }
      break;
    default:
      error->all(FLERR, "charge conservation calculation more than two electrode is not implemented yet");
  }
  // printf("POST2ND_solve_ele_matrix end\n");
  return 0;
}

/* ------------------------------------------------------------------------------------------
    Update the charge on localGA to ensure a charge neutrality in
CELL_POT_CONTROL MODEL
------------------------------------------------------------------------------------------
*/

double FixConpGA::POST2ND_localGA_neutrality(double qsum_totalGA, double* q)
{
  FUNC_MACRO(0);
  // double *q = atom->q;
  const int* const GA_seq_arr = &localGA_seq.front();
  // rintf("qsum_totalGA = %.12g\n", qsum_totalGA);

  if (fabs(qsum_totalGA) < 1e-15) {
    // if (me == 0) utils::logmesg(lmp, "Reach the criteria without new
    // charge neutrality treatment\n");
    return 0;
  }
  // if (me == 0) utils::logmesg(lmp, "qsum_totalGA = {:g}\n",
  // qsum_totalGA);
  int n  = atom->ntypes;
  Ushift = qsum_totalGA * inv_AAA_sumsum;
  if (B0_prev) {
    for (size_t i = 0; i < localGA_num; i++) {
      q[GA_seq_arr[i]] -= AAA_sum1D[i] * qsum_totalGA;
      B0_prev[i] -= Ushift;
    }
  } else {
    for (size_t i = 0; i < localGA_num; i++) {
      q[GA_seq_arr[i]] -= AAA_sum1D[i] * qsum_totalGA;
    }
  }

  Uchem -= Ushift;

  /*
  if (me == 0) {
    if (unit_flag == 1) {
      utils::logmesg(lmp, "The Extra chemical potential shift is %f [V]\n", Uchem *
  inv_qe2f); } else { utils::logmesg(lmp, "The Extra chemical potential shift is %f
  [Energy/e]\n", Uchem);
    }
  }
  */
  for (int i = 0; i < localGA_num; i++) {
    localGA_Vext[i] -= Ushift;
  }
  neutrality_state = false; // TODO to make it true;
  return Ushift;
}

/* ----------------------------------------------------------------------------------------
    Update the atomic charge on each eletrode based on the inverse of matrix AAA
----------------------------------------------------------------------------------------
*/
void FixConpGA::update_charge_AAA()
{
  FUNC_MACRO(0)
  int i, j, ii;
  int nlocal                  = atom->nlocal;
  const int* const GA_seq_arr = &localGA_seq.front();
  const int* const GA_seq_end = &localGA_seq.front() + localGA_num;
  int iseq, *type = atom->type, *tag = atom->tag;
  // double qlocalGA_incr_sum(0), qGA_incr_sum(0);
  set_zeros(Ele_qsum, 0, Ele_num);
  double* q = atom->q; // qtmp, qsum_store;
  // double *BBB, *BBB_localGA;
  time_t t_begin = clock();
  // bool tag2seq_count = false;
  /*
  if (neutrality_flag == true) {
    if (neutrality_state == false) {
      qsum_store = electro->calc_qsum();
    } else {
      #ifndef CONP_NO_DEBUG
      qsum_store = electro->calc_qsum();
      if (fabs(qsum_store) > 1e-11)
        if(me == 0) utils::logmesg(lmp, "invalid neutrality state of qsum =
  %.10f\n", qsum_store); #endif
    }
  }
  */
  // if (me == 0 && DEBUG_LOG_LEVEL > 0) utils::logmesg(lmp, "qsum =
  // %.16f, Uchem = %f\n", qsum, Uchem);

  // double * qstore;
  // memory->create(q_store, localGA_num, "FixConpGA:q_store");

  // double self_potential = Xi_self * qsum;
  // vif (me == 0 && force->kspace != nullptr) utils::logmesg(lmp,
  // "self_potential = %6e\n", self_potential); double cl_hash = 0, ks_hash = 0;

  for (size_t i = 0; i < localGA_num; i++) {
    iseq         = GA_seq_arr[i];
    double value = force->kspace->eatom[iseq];
    // value = force->pair->cl_atom[iseq]/q[iseq] + force->kspace->eatom[iseq];

    BBB_localGA[i] = localGA_Vext[i] - value; // - electro->eatom[ilocalseq];
  }
  // printf("cl_hash = %.16f, ks_hash = %.16f\n", cl_hash, ks_hash);

  MPI_Allgatherv(BBB_localGA, localGA_num, MPI_DOUBLE, BBB, &localGA_EachProc.front(), &localGA_EachProc_Dipl.front(), MPI_DOUBLE, world);

  if (0 && me == 0) {
    for (size_t i = 0; i < totalGA_num; i++) {
      printf("CPU[%d]:%lu %d %20.18f\n", me, i, totalGA_tag[i], BBB[i]);
    }
  }

  // printf("start3 \n");

  double* iAAA = AAA[0]; // [localGA_EachProc_Dipl[me]];
  for (size_t i = 0; i < localGA_num; i++, iAAA += totalGA_num) {
    double value = 0;
    for (size_t j = 0; j < totalGA_num; j++) {
      value += iAAA[j] * BBB[j];
    }
    if (GA2GA_flag) {
      q[GA_seq_arr[i]] += value;
    } else {
      q[GA_seq_arr[i]] = value;
    }
    int iEle = localGA_eleIndex[i];
    Ele_qsum[iEle] += q[GA_seq_arr[i]];
    // printf("CPU%d: %lu[tag:%d] Q = %20.18g, Shift_Value = %g\n", me, i,
    // tag[GA_seq_arr[i]], q[GA_seq_arr[i]], value);
  }

  // printf("BBB_localGA[i] = %f, %f\n", BBB[0], BBB[1]);
  // printf("q1,q2= %.10f, %.10f, %.10f, %.10f\n", q[0], q[1], q[2], q[3]);
  // printf("qlocalGA_incr_sum = %f, %f\n", qlocalGA_incr_sum, qsum_store);

  // nality constrain
  if (postreat_flag) {
    // MPI_Allreduce(&qlocalGA_incr_sum, &qGA_incr_sum, 1, MPI_DOUBLE, MPI_SUM,
    // world); localGA_charge_neutrality(qsum_store + qGA_incr_sum, q);
    POST2ND_charge(qtotal_SOL, Ele_qsum, q);
    // if (me == 0 && DEBUG_LOG_LEVEL > 0) utils::logmesg(lmp, "qsum = {:.16}\n", qsum);
  }
  // printf("q1,q2= %.10f, %.10f, %.10f, %.10f\n", q[0], q[1], q[2], q[3]);

  reset_nonzero();
  electro->qsum_qsq();
  fix_forward_comm(0);

  time_t t_end  = clock();
  update_Q_time = double(t_end - t_begin) / CLOCKS_PER_SEC;
  update_Q_totaltime += update_Q_time;
  curriter = 1;
  /*
  double *prd = domain->prd;
  double volume = prd[0]*prd[1]*prd[2];
  double g_ewald = force->kspace->g_ewald;
  */
}

/* -----------------------------------------------------
    Solving the least squres problem
------------------------------------------------------ */
void FixConpGA::solve_LS(double* a, double* b, char T, MKL_INT M, MKL_INT N)
{
  // if T = 'T', swap m and n
  // MKL_INT m = N, n = M, nrhs = 1, lda = m, ldb = n, info, lwork, rank; // for
  // dgels_
  MKL_INT max = M > N ? M : N;
  MKL_INT m = M, n = N, nrhs = 1, lda = m, ldb = max, info, lwork,
          rank; // for dgels_
  // MKL_INT m = M, n = N, nrhs = 1, lda = m, ldb = m, info, lwork, rank; // for
  // dgels_

  double rcond = 0.001;
  char buf[256];
  double wkopt;
  double* work;
  double s[M];

  // char T = 'N';
  if (me == 0 && DEBUG_LOG_LEVEL > 0) {
    utils::logmesg(lmp, "solve_LS with T = {}\n", T);
    utils::logmesg(lmp, "  m = {},   n = {}\n", m, n);
    utils::logmesg(lmp, "lda = {}, ldb = {}\n", lda, ldb);
  }

  MKL_INT iwork[16 * max * 1 + 11 * max];
  lwork = -1;
  dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, &wkopt, &lwork, iwork, &info);
  if (info != 0) {
    sprintf(buf, "[ConpGA] Error in dgels_ with lwork = -1, info = %d\n", info);
    error->one(FLERR, buf);
  }
  lwork = (int)wkopt;
  work  = (double*)malloc(lwork * sizeof(double));
  dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, iwork, &info);
  if (info != 0) {
    sprintf(buf, "[ConpGA] Error in dgels_ with lwork = -1, info = %d\n", info);
    error->one(FLERR, buf);
  }

  // dgels_ not work try dgelss
  /*
  lwork = -1;
  dgels_( &T, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork, &info);
  if (info != 0) {
      sprintf(buf, "[ConpGA] Error in dgels_ with lwork = -1, info = %d\n",
  info); error->one(FLERR, buf);
  }
  lwork = (int)wkopt;
  work = (double*) malloc( lwork*sizeof(double) );
  dgels_( &T, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info );

  if( info > 0 ) {
    printf( "The diagonal element %i of the triangular factor ", info );
    printf( "of A is zero, so that A does not have full rank;\n" );
    printf( "the least squares solution could not be computed.\n" );
  }
  if (info != 0) {
    sprintf(buf, "[ConpGA] Error in dgels_ with lwork = -1, info = %d\n", info);
    error->one(FLERR, buf);
  }
  */

  free((void*)work);
}

/* -----------------------------------------------------
    Inverse the matrix of AAA in situ
------------------------------------------------------ */
void FixConpGA::inv_AAA_LAPACK(double* AAA, MKL_INT mat_size)
{
  FUNC_MACRO(0);
  MKL_INT i, j, k, idx1d;
  MKL_INT m     = mat_size;
  MKL_INT n     = mat_size;
  MKL_INT lda   = mat_size;
  MKL_INT* ipiv = new MKL_INT[mat_size + 1];
  MKL_INT lwork = -1; // mat_size*mat_size;
  MKL_INT info;
  double wkopt[4];
  char buf[256];

  if (me == 0) utils::logmesg(lmp, "[ConpGA] LAPACK with mat_size = {};\n", mat_size);
  dgetrf_(&m, &n, AAA, &lda, ipiv, &info);
  if (info != 0) {
    sprintf(buf, "[ConpGA] LAPACK Error in dgetrf_, info = %s\n", info);
    error->one(FLERR, buf);
  }

  dgetri_(&n, AAA, &lda, ipiv, wkopt, &lwork, &info); // save mem.
  if (info != 0) {
    sprintf(buf, "[ConpGA] LAPACK Error in dgetri_ with lwork = -1, info = %d\n", info);
    error->one(FLERR, buf);
  }

  lwork = (MKL_INT)wkopt[0] + 1; // save mem.
  if (me == 0) utils::logmesg(lmp, "[ConpGA] LAPACK with lwork = {}\n", lwork);
  if (lwork <= 0 || lwork > max_lwork) {
    if (me == 0) utils::logmesg(lmp, "[ConpGA] LAPACK with reseting lwork as {}\n", max_lwork);
    lwork = max_lwork;
  }

  double* work = new double[lwork];

  dgetri_(&n, AAA, &lda, ipiv, work, &lwork, &info);
  if (info != 0) {
    sprintf(buf, "[ConpGA] LAPACK Error in dgetri_, info = %d\n", info);
    error->one(FLERR, buf);
  }
  delete[] ipiv;
  ipiv = nullptr;
  delete[] work;
  work = nullptr;
}

/* ------------------------------------------------------------------------------------
    Calculate the average value of inv-AAA matrix for each column (atom) by CG
or PCG
--------------------------------------------------------------------------------------
*/
void FixConpGA::meanSum_AAA_cg()
{
  FUNC_MACRO(0)
  int i, iseq, n = atom->ntypes, *type = atom->type;
  int nlocal          = atom->nlocal;
  TOL tol_style_store = tol_style;
  double *q = atom->q, qtmp, AAA_sumsum_local = 0, tolerance_store = tolerance;
  const int* const localGA_seq_arr = &localGA_seq.front();

  double* q_store_arr;
  double* const cl_atom = electro->cl_atom();
  double* const ek_atom = electro->ek_atom();

  memory->create(AAA_sum1D, localGA_num, "FixConp/GA: AAA_sum1D");
  memory->create(q_store_arr, localGA_num, "FixConp/GA: q_store_arr");
  bool neutrality_flag_store = neutrality_flag;
  bool postreat_flag_store   = postreat_flag;

  for (i = 0; i < localGA_num; i++) {
    iseq           = localGA_seq_arr[i];
    q_store_arr[i] = q[iseq];
    q[iseq]        = 1e-30;
    // qtmp = q[iseq];
    cl_atom[iseq] = 0.0; // (-1 + localGA_Vext[i]); // * qtmp;
    // AAA_sum1D[i] = 0;
    ek_atom[iseq] = (-1 + localGA_Vext[i]); // The extra localGA_Vext will be cancelled in the
  }

  // if (minimizer == MIN::PCG) {
  tolerance = 1e-20;
  tol_style = TOL::MAX_Q;
  if (me == 0)
    utils::logmesg(lmp,
                   "[ConpGA] Switch to max_Q criterion to calculate inv_AAA_sumsum "
                   "with the tolerance of {:g}\n",
                   tolerance); //}

  neutrality_flag = postreat_flag = false;
  switch (minimizer) {
    case MIN::NEAR:
    case MIN::CG:
      update_charge_cg(false);
      break;
    case MIN::PCG:
      update_charge_pcg();
      break;
    case MIN::PCG_EX:
      update_charge_ppcg();
      break;
    default:
      error->all(FLERR, "meanSum_AAA_cg() only valid for MIN::CG, MIN::PCG or MIN::PPCG methods");
  }
  tolerance       = tolerance_store;
  tol_style       = tol_style_store;
  neutrality_flag = neutrality_flag_store;
  postreat_flag   = postreat_flag_store;

  for (i = 0; i < localGA_num; i++) {
    iseq = localGA_seq_arr[i];
    // update AAA_sum1D get the increase of q[iseq]
    // with a constant potential shift for all metal atoms
    AAA_sumsum_local += AAA_sum1D[i] = q[iseq]; // - AAA_sum1D[i];
    q[iseq]                          = q_store_arr[i];
    if (me == 0 && DEBUG_LOG_LEVEL > 1) {
      utils::logmesg(lmp, "[Proc. #{}] {} {:.16} {:.16} {:.16}\n", me, i, localGA_Vext[i], q[iseq], AAA_sum1D[i]);
    }
  }
  MPI_Allreduce(&AAA_sumsum_local, &AAA_sumsum, 1, MPI_DOUBLE, MPI_SUM, world);
  if (me == 0) utils::logmesg(lmp, "[ConpGA] inv_AAA_sumsum = {:20.16f} from iteration methods\n", 1 / AAA_sumsum);

  inv_AAA_sumsum = 1 / AAA_sumsum;
  for (size_t i = 0; i < localGA_num; i++) {
    // printf("%d %f\n", i, AAA_sum1D[i]);
    AAA_sum1D[i] *= inv_AAA_sumsum;
  }
  memory->destroy(q_store_arr);
  neutrality_flag = neutrality_flag_store;
}

/* ------------------------------------------------------------------------
    Calculate the average value of inv-AAA matrix for each column (atom)
    Q_i = \sum_{j} A^{-1}_{ij}U_j
    Q_{sum} = \sum_{i,j} A^{-1}_{ij}U_j
    Q_{sum} = -\sum_{i,j} A^{-1}_{ij}\dU_j with dU_j = du_0
    Q_{sum} = du_0 * (\sum_{i,j} A^{-1}_{ij}\dU_j)
    Set inv_AAA_sum = (\sum_{i,j} A^{-1}_{ij}\dU_j)^(-1)
    du_0 = -Q_{sum} * inv_AAA_sum
    dQ_i = du_0 * \sum_{j} A^{-1}_{ij} = -Q_{sum} (\sum_{j} A^{-1}_{ij} *
 inv_AAA_sum)
 ------------------------------------------------------------------------ */
void FixConpGA::meanSum_AAA()
{
  int AAA_disp = localGA_EachProc_Dipl[me];
  memory->create(AAA_sum1D, localGA_num, "FixConp/GA: AAA_sum1D");
  double AAA_sumsum_local = 0;
  AAA_sumsum              = 0;

  for (size_t i = 0; i < localGA_num; i++) {
    AAA_sum1D[i] = 0;
    for (size_t j = 0; j < totalGA_num; j++) {
      AAA_sum1D[i] += AAA[i][j];
    }
    AAA_sumsum_local += AAA_sum1D[i];
  }
  MPI_Allreduce(&AAA_sumsum_local, &AAA_sumsum, 1, MPI_DOUBLE, MPI_SUM, world);

  inv_AAA_sumsum = 1 / AAA_sumsum;

  if (DEBUG_LOG_LEVEL > 1 && me == work_me) utils::logmesg(lmp, "#atom  AAA_sum1D\n");
  if (Smatrix_flag == false) {
    for (size_t i = 0; i < localGA_num; i++) {
#ifndef CONP_NO_DEBUG
      if (DEBUG_LOG_LEVEL > 1 && me == work_me) utils::logmesg(lmp, "{} {:.16f}\n", i, AAA_sum1D[i]);
#endif
      AAA_sum1D[i] *= inv_AAA_sumsum;
    }
  }
  if (me == 0) utils::logmesg(lmp, "[ConpGA] inv_AAA_sumsum = {:.16f} from inverse method\n", inv_AAA_sumsum);
}

/* -------------------------------------------------------------------
    Load AAA matrix and totalGA_tag from a txt file
------------------------------------------------------------------- */
void FixConpGA::loadtxt_AAA()
{
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Load text AAA matrix from {}\n", AAA_save);
  int* totalGA_tag_arr = &totalGA_tag.front();
  int oldtotalGA_num, itag, i_old_seq;
  int* oldtotalGA_tag_arr;
  size_t i, j, i_new, j_new;
  map<int, int> oldtotalGA_tag2seq;
  MAP_RESERVE(oldtotalGA_tag2seq, totalGA_num);
  char INFO[500];
  double** AAA_Global;
  bool reshuffle_flag = false;

  double g_ewald = 0.0;
  int nn_pppm[3] = {0, 0, 0};

  // FILE* outa = fopen(AAA_save,"r");

  memory->create(AAA, localGA_num, totalGA_num, "FixConp/GA: AAA_tmp");
  if (me == 0) {
    memory->create(AAA_Global, totalGA_num, totalGA_num, "FixConp/GA: AAA_tmp");

    std::ifstream myfile(AAA_save);
    if (!myfile.is_open()) {
      sprintf(INFO, "[ConpGA] Cannot open A matrix file: %.400s", AAA_save);
      error->all(FLERR, INFO);
    }
    myfile >> oldtotalGA_num;
    if (oldtotalGA_num != totalGA_num) {
      sprintf(INFO, "[ConpGA] Inconsistent total GA atom number Old: %d vs New: %d", oldtotalGA_num, totalGA_num);
      error->all(FLERR, INFO);
    }
    myfile >> g_ewald;
    myfile >> nn_pppm[0];
    myfile >> nn_pppm[1];
    myfile >> nn_pppm[2];
    if (force->kspace) {
      if (fabs(1 - force->kspace->g_ewald / g_ewald) > 1e-12) {
        sprintf(INFO, "[ConpGA] Inconsistent kspace setup for %28.26f vs %f\n", g_ewald, force->kspace->g_ewald);
        error->all(FLERR, INFO);
      }
      if (nn_pppm[0] != force->kspace->nx_pppm || nn_pppm[1] != force->kspace->ny_pppm || nn_pppm[2] != force->kspace->nz_pppm) {
        sprintf(INFO, "[ConpGA] Inconsistent kspace setup for %dx%dx%d vs %dx%dx%d\n", nn_pppm[0], nn_pppm[1], nn_pppm[2], force->kspace->nx_pppm,
                force->kspace->ny_pppm, force->kspace->nz_pppm);
        error->all(FLERR, INFO);
      }
    }

    memory->create(oldtotalGA_tag_arr, totalGA_num, "FixConp/GA: oldtotalGA_tag_arr");

    // printf("Tag = ");
    for (i = 0; i < totalGA_num; i++) {
      if (!(myfile >> oldtotalGA_tag_arr[i])) error->all(FLERR, "[ConpGA] Invalid AAA matrix format, not enough GA tags");
      oldtotalGA_tag2seq[oldtotalGA_tag_arr[i]] = i;
      // printf("%d ", oldtotalGA_tag_arr[i]);
    }
    // printf("\n");
    for (i = 0; i < totalGA_num; i++) {
      itag = totalGA_tag_arr[i];
      if (oldtotalGA_tag2seq.count(itag) == 0) {
        sprintf(INFO, "[ConpGA] The GA atom with tag [%d] from current MD run is missing", itag);
        error->all(FLERR, INFO);
      }
      i_old_seq                     = oldtotalGA_tag2seq[itag];
      oldtotalGA_tag_arr[i_old_seq] = i;
      // printf("%d[%d] %d\n", i, itag, i_old_seq);
      if (i_old_seq != i) reshuffle_flag = true;
    }

    // reshuffle_flag = true;
    if (reshuffle_flag) {
      if (me == 0) utils::logmesg(lmp, "[ConpGA] Read from {} file with reshuffle mode!\n", AAA_save);
      for (i = 0; i < totalGA_num; i++) {
        i_new = oldtotalGA_tag_arr[i];
        for (j = 0; j < totalGA_num; j++) {
          j_new = oldtotalGA_tag_arr[j];
          if (!(myfile >> AAA_Global[i_new][j_new]))
            error->all(FLERR, "[ConpGA] Invalid AAA matrix format, not enough "
                              "AAA matrix elements");
        }
      }
    } else {
      double* AAA_start = &AAA_Global[0][0];
      if (me == 0) utils::logmesg(lmp, "[ConpGA] Read from {} without the reshuffle of matrix\n", AAA_save);
      for (size_t i = 0; i < totalGA_num * totalGA_num; i++) {
        if (!(myfile >> AAA_start[i]))
          error->all(FLERR, "[ConpGA] Invalid AAA matrix format, not enough "
                            "AAA matrix elements");
      }
    }
    memory->destroy(oldtotalGA_tag_arr);
    myfile.close();
  } else { // me == 0;
    AAA_Global = AAA;
  }
  MPI_Barrier(world);

  const int* const size_EachProc = &localGA_EachProc.front();
  const int* const size_Dipl     = &localGA_EachProc_Dipl.front();

  matrix_scatter(AAA_Global, AAA, totalGA_num, size_EachProc, size_Dipl);
  if (me == 0) memory->destroy(AAA_Global);

  if (me == work_me && DEBUG_LOG_LEVEL > 0) {
    printf("CPU-%d:\n", me);
    for (size_t i = 0; i < totalGA_num; ++i) {
      printf("Atom #%2d ", totalGA_tag_arr[i]);
      for (size_t j = 0; j < totalGA_num; ++j) {
        printf("%12.8f ", AAA_Global[i][j]);
      }
      // printf("  meanSum = %12.10f\n", AAA_sum1D[i]);
    }
  }
  meanSum_AAA();
} // loadtxt_AAA()

/* -------------------------------------------------------------------
    Load AAA matrix and totalGA_tag from a binary file
------------------------------------------------------------------- */
void FixConpGA::load_AAA()
{
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Load binary Ewald matrix from {}\n", AAA_save);
  int* totalGA_tag_arr = &totalGA_tag.front();
  int oldtotalGA_num, itag, i_old_seq;
  int* oldtotalGA_tag_arr;
  double g_ewald = 0.0;
  int nn_pppm[3] = {0, 0, 0};
  size_t i, j, i_new, j_new;
  map<int, int> oldtotalGA_tag2seq;
  MAP_RESERVE(oldtotalGA_tag2seq, totalGA_num);
  char INFO[500];
  double *AAA_tmp1D, **AAA_Global = nullptr;
  bool reshuffle_flag = false;

  memory->create(AAA, localGA_num, totalGA_num, "FixConp/GA: AAA");
  if (me == 0) {
    memory->create(AAA_Global, totalGA_num, totalGA_num, "FixConp/GA: AAA_Global");
    size_t result;
    /*
    if (access(AAA_save, F_OK) == -1) {
      sprintf(INFO, "[ConpGA] Cannot open A matrix file: %.400s", AAA_save);
      error->all(FLERR, INFO);
    }
    */
    FILE* infile = fopen(AAA_save, "rb");
    if (infile == nullptr) {
      sprintf(INFO, "[ConpGA] Cannot open A matrix file: %.400s", AAA_save);
      error->all(FLERR, INFO);
    }

    utils::sfread(FLERR, &oldtotalGA_num, sizeof(int), 1, infile, nullptr, error);
    if (oldtotalGA_num != totalGA_num) {
      sprintf(INFO, "[ConpGA] Inconsistent total GA atom number Old: %d vs New: %d", oldtotalGA_num, totalGA_num);
      error->all(FLERR, INFO);
    }

    utils::sfread(FLERR, &g_ewald, sizeof(double), 1, infile, nullptr, error);
    utils::sfread(FLERR, nn_pppm, sizeof(int), 3, infile, nullptr, error);
    if (force->kspace) {
      if (fabs(1 - force->kspace->g_ewald / g_ewald) > 1e-12) {
        sprintf(INFO, "[ConpGA] Inconsistent kspace setup for %28.26f vs %f\n", g_ewald, force->kspace->g_ewald);
        error->all(FLERR, INFO);
      }
      if (nn_pppm[0] != force->kspace->nx_pppm || nn_pppm[1] != force->kspace->ny_pppm || nn_pppm[2] != force->kspace->nz_pppm) {
        sprintf(INFO, "[ConpGA] Inconsistent kspace setup for %dx%dx%d vs %dx%dx%d\n", nn_pppm[0], nn_pppm[1], nn_pppm[2], force->kspace->nx_pppm,
                force->kspace->ny_pppm, force->kspace->nz_pppm);
        error->all(FLERR, INFO);
      }
    }

    memory->create(oldtotalGA_tag_arr, totalGA_num, "FixConp/GA: oldtotalGA_tag_arr");

    utils::sfread(FLERR, oldtotalGA_tag_arr, sizeof(int), totalGA_num, infile, nullptr, error);
    // if (result != totalGA_num) error->all(FLERR, "[ConpGA] Invalid AAA matrix format, not enough atoms");

    for (i = 0; i < totalGA_num; i++)
      oldtotalGA_tag2seq[oldtotalGA_tag_arr[i]] = i;

    for (i = 0; i < totalGA_num; i++) {
      itag = totalGA_tag_arr[i];
      if (oldtotalGA_tag2seq.count(itag) == 0) {
        sprintf(INFO,
                "FixConp/GA: the GA atom with tag [%d] from current MD run is "
                "missing",
                itag);
        error->all(FLERR, INFO);
      }
      i_old_seq                     = oldtotalGA_tag2seq[itag];
      oldtotalGA_tag_arr[i_old_seq] = i;
      // printf("%d[%d] %d\n", i, itag, i_old_seq);
      if (i_old_seq != i) reshuffle_flag = true;
    }

    // reshuffle_flag = true;
    // printf("reshuffle_flag %d\n", reshuffle_flag);
    if (reshuffle_flag) {
      if (me == 0) utils::logmesg(lmp, "[ConpGA] Read from {} file with reshuffle mode!\n", AAA_save);

      memory->create(AAA_tmp1D, totalGA_num, "FixConp/GA: AAA_tmp");
      for (i = 0; i < totalGA_num; i++) {
        i_new = oldtotalGA_tag_arr[i];

        utils::sfread(FLERR, &AAA_tmp1D[0], sizeof(double), totalGA_num, infile, nullptr, error);
        // if (result != totalGA_num)
        //  error->all(FLERR, "FixConp/GA: Invalid AAA matrix format, not enough "
        //                    "AAA matrix elements");

        for (j = 0; j < totalGA_num; j++) {
          j_new                    = oldtotalGA_tag_arr[j];
          AAA_Global[i_new][j_new] = AAA_tmp1D[j];
          // AAA[i][j] = AAA_tmp[i_new][j_new];
        }
      }
      memory->destroy(AAA_tmp1D);

    } else {
      if (me == 0) utils::logmesg(lmp, "[ConpGA] Read from {} without the reshuffle of matrix\n", AAA_save);
      utils::sfread(FLERR, &AAA_Global[0][0], sizeof(double), totalGA_num * totalGA_num, infile, nullptr, error);
    }
    memory->destroy(oldtotalGA_tag_arr);
    // infile->close();
  } else { // me == 0;
    AAA_Global = AAA;
  }
  MPI_Barrier(world);

  const int* const size_EachProc = &localGA_EachProc.front();
  const int* const size_Dipl     = &localGA_EachProc_Dipl.front();

  matrix_scatter(AAA_Global, AAA, totalGA_num, size_EachProc, size_Dipl);
  if (me == 0) memory->destroy(AAA_Global);
  meanSum_AAA();
}

/* ------------------------------------------------------------------
 Save AAA matrix and totalGA_tag into a bin format file (*.bin)
------------------------------------------------------------------- */
void FixConpGA::save_AAA()
{
  if (me == 0) {
    utils::logmesg(lmp, "[ConpGA] Save AAA matrix to binary {} file\n", AAA_save);
    FILE* outa                 = fopen(AAA_save, "wb");
    int* const totalGA_tag_arr = &totalGA_tag.front();
    double g_ewald             = 0.0;
    int nn_pppm[3]             = {0, 0, 0};
    if (force->kspace) {
      g_ewald    = force->kspace->g_ewald;
      nn_pppm[0] = force->kspace->nx_pppm;
      nn_pppm[1] = force->kspace->ny_pppm;
      nn_pppm[2] = force->kspace->nz_pppm;
    }

    if (outa == nullptr) {
      char INFO[500];
      sprintf(INFO, "[ConpGA] Cannot open A matrix file: %.400s", AAA_save);
      error->all(FLERR, INFO);
    }
    fwrite(&totalGA_num, sizeof(int), 1, outa);
    fwrite(&g_ewald, sizeof(double), 1, outa);
    fwrite(nn_pppm, sizeof(int), 3, outa);
    fwrite(totalGA_tag_arr, sizeof(int), totalGA_num, outa);
    fwrite(&AAA[0][0], sizeof(double), localGA_num * totalGA_num, outa);
    fclose(outa);
  }
  for (int rank = 1; rank < nprocs; ++rank) {
    MPI_Barrier(world);
    if (me == rank) {
      FILE* outa = fopen(AAA_save, "ab");
      fwrite(&AAA[0][0], sizeof(double), localGA_num * totalGA_num, outa);
      fclose(outa);
    }
  }
} // save_AAA()

/* ------------------------------------------------------------------
 Save AAA matrix and totalGA_tag into a txt format file (*.txtdat)
------------------------------------------------------------------- */
void FixConpGA::savetxt_AAA()
{
  if (me == 0) {
    utils::logmesg(lmp, "[ConpGA] Save AAA matrix to text {} file\n", AAA_save);
    FILE* outa                 = fopen(AAA_save, "w");
    int* const totalGA_tag_arr = &totalGA_tag.front();

    double g_ewald = 0.0;
    int nn_pppm[3] = {0, 0, 0};
    if (force->kspace) {
      g_ewald    = force->kspace->g_ewald;
      nn_pppm[0] = force->kspace->nx_pppm;
      nn_pppm[1] = force->kspace->ny_pppm;
      nn_pppm[2] = force->kspace->nz_pppm;
    }

    if (outa == nullptr) {
      char INFO[500];
      sprintf(INFO, "[ConpGA] Cannot open A matrix file: %.400s", AAA_save);
      error->all(FLERR, INFO);
    }
    fprintf(outa, "%d\n", totalGA_num);
    fprintf(outa, "%20.16g %d %d %d\n", g_ewald, nn_pppm[0], nn_pppm[1], nn_pppm[2]);

    for (size_t i = 0; i < totalGA_num; i++) {
      fprintf(outa, "%d ", totalGA_tag_arr[i]);
    }
    fprintf(outa, "\n");
    for (size_t i = 0; i < localGA_num; i++) {
      for (size_t j = 0; j < totalGA_num; j++) {
        fprintf(outa, "%20.16g ", AAA[i][j]);
      }
      fprintf(outa, "\n");
    }
    fclose(outa);
  }
  for (int rank = 1; rank < nprocs; ++rank) {
    MPI_Barrier(world);
    if (me == rank) {
      FILE* outa = fopen(AAA_save, "a");
      for (size_t i = 0; i < localGA_num; i++) {
        for (size_t j = 0; j < totalGA_num; j++) {
          fprintf(outa, "%20.16g ", AAA[i][j]);
        }
        fprintf(outa, "\n");
      }
      fclose(outa);
    }
  }
} // savetxt_AAA()

/* --------------------------------------------------------------------------
   Calulcate the real space Couloubic interaction between GA<->GA via:
   1) POT_Flag: true to output potential, false to output energy
   2) COMM_Flag: Switch to communicate ghost data
   3) default: false, true
-------------------------------------------------------------------------- */
void FixConpGA::fix_pair_compute(bool POT_Flag, bool COMM_Flag)
{
  FUNC_MACRO(1)
  int i, iseq, itype, *type           = atom->type;
  int jj, jseq, jtag, *firstneigh_seq = &localGA_firstneigh.front();
  double *const cl_atom = electro->cl_atom(), *const q = atom->q;
  double qtmp, value, *firstneigh_coeff = &localGA_firstneigh_coeff.front();
  const int* const localGA_seq_arr = &localGA_seq.front();
  double* const localGA_GreenCoef  = &localGA_GreenCoef_vec.front();
  int nlocal                       = atom->nlocal;
  if (newton_pair) {
    clear_atomvec(cl_atom, extGA_num);
  } else {
    clear_atomvec(cl_atom, localGA_num);
  }

  if (!POT_Flag) {
    if (pair_flag) {
      for (i = 0; i < localGA_num; i++) {
        iseq = localGA_seq_arr[i];
        // itype = type[iseq];
        qtmp = atom->q[iseq];
        // i<->i Self-interaction; For point charge is zero, For Gaussian charge
        // is 1/2 * [2/sqrt(PI) * eta] Half for the similar accounting of
        // capacitance
        cl_atom[iseq] += qtmp * qtmp * localGA_GreenCoef[i];
        for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj++) {
          // jseq = firstneigh_seq[jj];
          jtag  = firstneigh_seq[jj];
          jseq  = atom->map(jtag);
          value = firstneigh_coeff[jj] * qtmp * q[jseq];
          cl_atom[iseq] += value;
          cl_atom[jseq] += value;
        }
        for (jj = localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
          jtag = firstneigh_seq[jj];
          jseq = atom->map(jtag);
// TO DO: remove jseq check;
#ifndef CONP_NO_DEBUG
          if (jseq == -1) error->one(FLERR, "Illegal fix conp command for unknown minimization method");
#endif
          value = firstneigh_coeff[jj] * qtmp * q[jseq];
          cl_atom[iseq] += value;
          if (newton_pair) cl_atom[jseq] += value;
        }
      } // i-forloop
    } else {
      /*
      error->all(FLERR, "The pair compute in FixConpGA::fix_pair_compute has "
                        "been turned off!\n");
      */
      electro->pair->pair_GAtoGA_energy(groupbit, false); // output energy
      // force->pair->compute(3, 0);
    }
    // POT_Flag
  } else {
    if (pair_flag) {
      for (i = 0; i < localGA_num; i++) {
        iseq = localGA_seq_arr[i];
        qtmp = atom->q[iseq];
        cl_atom[iseq] += qtmp * localGA_GreenCoef[i];
        for (jj = localGA_numneigh_disp[i]; jj < localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj++) {
          jtag = firstneigh_seq[jj];
          jseq = atom->map(jtag);
          cl_atom[iseq] += firstneigh_coeff[jj] * q[jseq];
          cl_atom[jseq] += firstneigh_coeff[jj] * qtmp;
        }
        for (jj = localGA_numneigh_disp[i] + localGA_numlocalneigh[i]; jj < localGA_numneigh_disp[i] + localGA_numneigh[i]; jj++) {
          jtag = firstneigh_seq[jj];
          jseq = atom->map(jtag);
// TO DO: remove jseq check;
#ifndef CONP_NO_DEBUG
          if (jseq == -1) error->one(FLERR, "Illegal fix conp command for unknown minimization method");
#endif
          cl_atom[iseq] += firstneigh_coeff[jj] * q[jseq];
          if (newton_pair) cl_atom[jseq] += firstneigh_coeff[jj] * qtmp;
        }
      } // i-forloop
    } else {
      /*
      error->all(FLERR, "The pair compute in FixConpGA::fix_pair_compute has "
                        "been turned off!\n");
      */
      electro->pair->pair_GAtoGA_energy(groupbit, true); // output energy
      // force->pair->compute(3, 0);
    }
  }
  if (COMM_Flag && newton_pair) {
    fix_reverse_comm(REV_COUL_RM);
  }

  /*
  if(self_Xi_flag == true) {
    qsum_qsq();
    value = qsum * Xi_self;
    for (i = 0; i < localGA_num; i++)
    {
      iseq = localGA_seq_arr[i];
      cl_atom[iseq] += value * atom->q[iseq];
    }
  }
  */
}

void FixConpGA::zero_solution_check()
{
  FUNC_MACRO(0);
  printf("start of zero () from %d\n", me);
  int nlocal = atom->nlocal;
  int* mask  = atom->mask;
  double *q = atom->q, qFabs_local = 0.0, qFabs_sum;
  if (atom->natoms == totalGA_num) {
    error->all(FLERR, "There is no solution atom, so the selfGG cannot be turned on!\n");
  }

  for (int i = 0; i < nlocal; ++i) {
    if (!(mask[i] & groupbit)) {
      qFabs_local += fabs(q[i]);
    }
  }
  MPI_Allreduce(&qFabs_local, &qFabs_sum, 1, MPI_DOUBLE, MPI_SUM, world);
  if (qFabs_sum == 0) {
    error->all(FLERR, "There is not charge on the solution atoms, so the "
                      "selfGG cannot be turned on!");
  }
  printf("end of zero () from %d\n", me);
}

/* --------------------------------------------------------------------------------------
  Clear cl energy on the GA atoms
--------------------------------------------------------------------------------------
*/
void FixConpGA::clear_cl_atom()
{
  double* cl_atom = electro->cl_atom();
  if (newton_pair) {
    clear_atomvec(cl_atom, extGA_num);
  } else {
    clear_atomvec(cl_atom, localGA_num);
  }
}

/* --------------------------------------------------------------------------
  Clear the atomic property on the duplicated ghost atoms,
--------------------------------------------------------------------------------------
*/
double FixConpGA::check_atomvec(double* v, const char* name)
{
  char info[FIX_CONP_GA_WIDTH + 1];
  int i, iseq;
  double v_local                   = 0.0, v_sum;
  int n                            = localGA_num;
  const int* const localGA_seq_arr = &localGA_seq.front();
  for (i = 0; i < n; i++) {
    iseq = localGA_seq_arr[i];
    v_local += v[iseq] * v[iseq];
  }
  MPI_Allreduce(&v_local, &v_sum, 1, MPI_DOUBLE, MPI_SUM, world);
  v_sum = sqrt(v_sum);
  if (me == 0) printf("%s = %18.16f\n", name, v_sum);
  return v_sum;
}

/* --------------------------------------------------------------------------------------
  Clear the atomic property on the duplicated ghost atoms,
--------------------------------------------------------------------------------------
*/
void FixConpGA::clear_atomvec(double* q, int n)
{
  char info[FIX_CONP_GA_WIDTH + 1];
  int i, iseq;
  const int* const localGA_seq_arr = &localGA_seq.front();
  for (i = 0; i < n; i++) {
    iseq    = localGA_seq_arr[i];
    q[iseq] = 0;
  }
}

/* --------------------------------------------------------------------------------------
  Update the localGA sequence for each new atomsort();
--------------------------------------------------------------------------------------
*/
void FixConpGA::localGA_atomsort()
{
  FUNC_MACRO(0)
  char info[FIX_CONP_GA_WIDTH + 1];
  int i, iseq, newseq, i_index, nlocal = atom->nlocal, nghost = atom->nghost;
  int *tag = atom->tag, *type = atom->type;
  int ntotal                       = nlocal + nghost, iGA, iextGA, ishostGA;
  const int* const localGA_tag_arr = &localGA_tag.front();
  int* const GA_seq                = &localGA_seq.front();
  // int * const extlocalGA_seq_arr = localGA_seq_arr + usedGA_num;
  int* const extGA_seq = GA_seq + localGA_num;

  // resort the sequence of localGA for firstgroupFlag = off
  if (firstgroup_flag == false) {
    for (i = 0; i < localGA_num; i++) {
      newseq = atom->map(localGA_tag_arr[i]);
#ifndef CONP_NO_DEBUG
      if (newseq == -1 || newseq >= nlocal) {
        snprintf(info, FIX_CONP_GA_WIDTH, "Local atom tag: %d with old sequence %d missing with newseq: %d", localGA_tag_arr[i], localGA_seq[i], newseq);
        error->one(FLERR, info);
      } else {
        GA_seq[i] = newseq;
      }
#else
      GA_seq[i] = newseq;
#endif
    }
  } else {
// firstgroup == -1
#ifndef CONP_NO_DEBUG
    for (i = 0; i < localGA_num; i++) {
      newseq = atom->map(localGA_tag_arr[i]);
      if (newseq == -1 || newseq >= nlocal) {
        snprintf(info, FIX_CONP_GA_WIDTH, "Local atom tag: %d with old sequence %d missing with newseq: %d", localGA_tag_arr[i], localGA_seq[i], newseq);
        error->one(FLERR, info);
      }
      if (newseq != GA_seq[i]) {
        snprintf(info, FIX_CONP_GA_WIDTH,
                 "Invalid [%d] atom sequence from first on atom tag: %d with old sequence %d, now with tag %d missing "
                 "with newseq: %d with tag: %d\n",
                 i, localGA_tag_arr[i], localGA_seq[i], tag[localGA_seq[i]], newseq, tag[newseq]);
        error->one(FLERR, info);
      }
    }
#endif
  }

  for (i = 0; i < true_usedGA_num; ++i) {
    extraGA_num[i] = 0;
  }
  // usedGA_num = localGA_num + ghostGA_num;
  /*
  std::vector<bool> check_vec(usedGA_num, false);

  for (i = localGA_num; i < usedGA_num; ++i)
  {
    newseq = atom->map(localGA_tag_arr[i]);
    if (newseq == -1 || newseq < nlocal){
        snprintf(info, FIX_CONP_GA_WIDTH, "Local atom tag: %d with old sequence
  %d missing with newseq: %d", localGA_tag_arr[i], localGA_seq[i], newseq);
        error->one(FLERR, info);
    }
    localGA_seq_arr[i] = newseq;
    // check_vec[i] = true;
  }
  */
  /*
  for (i = localGA_num; i < usedGA_num; ++i)
  {
    if (check_vec[i] == false) {
      printf("wrong strucute 1\n!");
    }
    check_vec[i] = false;
  }
  */

  if (newton_pair) {
    const map<int, int>& tag_To_extraGAIndex = mul_tag_To_extraGAIndex;

    if (firstgroup_flag == true) {
      for (int iAll = 0, iseq = nlocal; iseq < ntotal; ++iseq) {
        if (GA_Flags[type[iseq]] != 0) {
          int itag = tag[iseq];
#ifdef CONP_NO_DEBUG
          iGA = mul_tag_To_extraGAIndex[itag];
#else
          iGA = tag_To_extraGAIndex.at(itag);
#endif
          // all_ghostGA_seq[iAll]                = iseq;
          comm_recv_arr[all_ghostGA_ind[iAll]] = iseq;
          // all_ghostGA_ind[iAll] = tag_to_ghostGA[itag];

          extGA_seq[extraGA_num[iGA] + extraGA_disp[iGA]] = iseq;
          extraGA_num[iGA] += 1;
          iAll++;
        }
      }
    } else {
      for (iseq = nlocal; iseq < ntotal; ++iseq) {
        if (GA_Flags[type[iseq]] != 0) {
          int itag = tag[iseq];
#ifdef CONP_NO_DEBUG
          iGA = mul_tag_To_extraGAIndex[itag];
#else
          iGA = tag_To_extraGAIndex.at(itag);
#endif
          extGA_seq[extraGA_num[iGA] + extraGA_disp[iGA]] = iseq;
          extraGA_num[iGA] += 1;
        }
      }
    }

    /*
    iextGA = ishostGA = 0;
    for (iseq = nlocal; iseq < ntotal; ++iseq) {
      if(GA_Flags[type[iseq]]) {
        iGA = tag_To_localIndex.at(tag[iseq]);
        if (iGA < localGA_num) {
          extlocalGA_seq_arr[extraGA_disp[iGA] + extraGA_num[iGA]] = iseq;
          extraGA_num[iGA]++;
          iextGA ++;
        } else {
          if (check_vec[iGA] == true) {
            if (me == 0) printf("iGA = %d, tag = %d, %d\n", iGA, tag[iseq],
    iseq);
            // error->one(FLERR, "duplciate save");
          }
          check_vec[iGA] = true;
          localGA_seq_arr[iGA] = iseq;
          ishostGA ++;
        }

        iseq = atom->map(tag[i]);
        #ifndef CONP_NO_DEBUG
        if (newseq == -1){
          snprintf(info, FIX_CONP_GA_WIDTH, "Local atom tag: %d with old
    sequence %d missing with newseq: %d", localGA_tag_arr[i], localGA_seq[i],
    newseq); error->one(FLERR, info);
        }
        #endif
          //printf("CPU[%d]: %d %d %d\n", me, i, iseq, tag[i]);
        if (iseq >= nlocal) {
          localGA_seq_arr[iGA] = i;
          iGA++;
        } else {
          extlocalGA_seq_arr[iextGA] = i;
          iextGA++;
        }

      // }
      */
    // } // nlocal <= i < ntotal
    /*
    for (i = localGA_num; i < usedGA_num; ++i)
    {
      if (check_vec[i] == false) {
        printf("wrong strucute2\n!");
      }
      check_vec[i] = false;
    }
    */
    /*
    if (iextGA != dupl_ghostGA_num || ishostGA != ghostGA_num) {
      snprintf(info, FIX_CONP_GA_WIDTH, "Wrongggggggg iextGA %d neq
    dupl_ghostGA_num %d", iextGA, dupl_ghostGA_num); error->one(FLERR, info);
    }
    */
  } else {
    for (i = localGA_num; i < usedGA_num; i++) {
      newseq = atom->map(localGA_tag_arr[i]);
#ifndef CONP_NO_DEBUG
      if (newseq == -1 || newseq < nlocal) {
        sprintf(info, "Ghost atom tag: %d with old sequence %d missing with newseq: %d", localGA_tag_arr[i], localGA_seq[i], newseq);
        error->one(FLERR, info);
      } else {
        GA_seq[i] = newseq;
      }
#else
      GA_seq[i] = newseq;
#endif
    } // i in [localGA_num, usedGA_num]
    if (firstgroup_flag == true) {
      for (int iAll = 0, iseq = nlocal; iseq < ntotal; ++iseq) {
        if (GA_Flags[type[iseq]] != 0) {
          // int itag = tag[iseq];
          // all_ghostGA_seq[iAll]                = iseq;
          comm_recv_arr[all_ghostGA_ind[iAll]] = iseq;
          // all_ghostGA_ind[iAll] = tag_to_ghostGA[itag];
          iAll++;
        }
      }
    }
  } // end of newton_pair
  /*
  double hash = hash_vec(all_ghostGA_ind, all_ghostGA_num);
  printf("[me = %d] Hash of all_ghostGA_ind = %.18f\n", me, hash);
  hash = hash_vec(all_ghostGA_seq, all_ghostGA_num);
  printf("[me = %d] Hash of all_ghostGA_seq = %.18f\n", me, hash);
  */
}

/*
void FixConpGA::qsum_qsq(){
  qsum_qsq(atom->q);
}
*/

/* -------------------------------------------
  Clear the force and energy in all local atoms
------------------------------------------- */
void FixConpGA::force_clear()
{
  FUNC_MACRO(0)
  int nall = atom->nlocal;
  if (force->newton) nall += atom->nghost;
  if (force->kspace) force->kspace->energy = 0;
  force->pair->eng_coul = 0;
  double **f = atom->f, *cl_atom = electro->cl_atom(), *eatom = force->pair->eatom;
  double* ek_atom = electro->ek_atom();
  // size_t nbytes = sizeof(double) * nall;
  set_zeros(&cl_atom[0], 0, nall);
  set_zeros(&eatom[0], 0, nall);
  set_zeros(&f[0][0], 0, 3 * nall);
  /*
  memset(&cl_atom[0], 0, nbytes);
  memset(&eatom[0], 0, nbytes);
  // memset(&ek_atom[0],  0,   nbytes);
  memset(&f[0][0], 0, 3 * nbytes);
  */
}
/* -------------------------------------------------
     Function to determine whether the
      GA<->GA interaction is accounted
------------------------------------------------- */
void FixConpGA::setup_GG_Flag()
{
  FUNC_MACRO(0)
  GA2GA_flag = true;
  if (fast_bvec_flag == true && electro->is_kspace_conp_GA() && selfGG_flag == false && DEBUG_SOLtoGA != false) {
    // with the capability to skip GA <-> GA interaction
    GA2GA_flag = false;
  }
  if (GA2GA_flag == false) {
    zero_solution_check();
  }

  if (GA2GA_flag == true && tol_style == TOL::REL_B) {
    ; // error->all(FLERR, "relative B0 model is only suitable for GA2GA_flag == False.");
  }

  if (minimizer != MIN::INV) {

    // GA2GA_flag = false;
  }
  if (me == 0)
    utils::logmesg(lmp,
                   "[ConpGA] Solving Eq=b with  GA<->GA interaction, "
                   "selfGG_Flag = {}.\n",
                   GA2GA_flag);
}

/* --------------------------------------
  Clear the energy in all local atoms
-------------------------------------- */
void FixConpGA::energy_clear()
{
  FUNC_MACRO(1);
  int nall = atom->nlocal;
  // nall += nghost = atom->nghost;
  // int nghost = atom->nghost;
  if (force->kspace) force->kspace->energy = 0;
  force->pair->eng_coul = 0;
  double **f = atom->f, *cl_atom = electro->cl_atom(), *eatom = force->pair->eatom;
  for (int i = 0; i < nall; ++i) {
    cl_atom[i] = 0;
    eatom[i]   = 0;
    // f[i][0] = 0; f[i][1] = 0; f[i][2] = 0;
  }
}

/* calculate both GA<->GA and GA<->Solution energy */
void FixConpGA::fix_localGA_pairenergy(bool GG_Flag)
{
  FUNC_MACRO(1)
  if (fast_bvec_flag == false || pair_flag == false) {
    force->pair->compute(3, 0);
    if (newton_pair) fix_reverse_comm(REV_COUL);
  } else {
    fix_localGA_pairenergy_flag(GG_Flag, true);
  }
}

/* -------------------------------------------------------------------
           Calculate both GA<->GA and GA<->Solution energy
           With fast_bvec_flag = true or pair_flag = true
------------------------------------------------------------------- */
void FixConpGA::fix_localGA_pairenergy_flag(bool GG_Flag, bool to_GA_Flag)
{
  // printf("CPU: %d XXX4\n", me);
  int ii, iseq, jj, jseq, inum, jnum, itype, jtype, i_neighGA;
  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  // factor_coul == 1 and factor_lj = -1 to ignore the LJ and force calculation
  double factor_coul = 1, factor_lj = 0;
  double qtmp, qtmp_half, xtmp, ytmp, ztmp, value;
  double delx, dely, delz, rsq, fforce, ecoul;
  double** x      = atom->x;
  double* q       = atom->q; // *ecoul_ptr;
  double* cl_atom = electro->cl_atom();
  bool iGA_Flags, jGA_Flags;
  int* type = atom->type;
  int* jlist;

  int *ilist, *numneigh, **firstneigh;
  int nlocal = atom->nlocal, nghost = atom->nghost, *tag = atom->tag;
  // loop over neighbors of my atoms
  inum       = list->inum;
  ilist      = list->ilist;
  numneigh   = list->numneigh;
  firstneigh = list->firstneigh;

  const int* const localGA_seq_arr      = &localGA_seq.front();
  const double* const localGA_GreenCoef = &localGA_GreenCoef_vec.front();
  const double* const firstneigh_coeff  = &localGA_firstneigh_coeff.front();
  /*
  for(ii = 0; ii < localGA_num; ++ii){
    iseq = localGA_seq_arr[ii];
    qtmp = q[iseq];
    cl_atom[iseq] = 0; qtmp * qtmp * localGA_GreenCoef[ii];
  }
  */

  // For GA<->GA interaction
  if (GG_Flag == true) {
    fix_pair_compute();
  } else {
    if (newton_pair) {
      clear_atomvec(cl_atom, extGA_num);
    } else {
      clear_atomvec(cl_atom, localGA_num);
    }
  }

  // For GA<->Solution and Solution<->GA
  for (ii = 0; ii < inum; ++ii) {
    iseq = ilist[ii];

    itype     = type[iseq];
    qtmp      = q[iseq];
    xtmp      = x[iseq][0];
    ytmp      = x[iseq][1];
    ztmp      = x[iseq][2];
    jlist     = firstneigh[iseq];
    jnum      = numneigh[iseq];
    iGA_Flags = GA_Flags[itype] != 0;

    // cl_atom[iseq] += qtmp * qtmp * localGA_GreenCoef[ii];

    for (jj = 0; jj < jnum; jj++) {
      jseq = jlist[jj];
      // factor_coul = special_coul[sbmask(j)];
      jseq &= NEIGHMASK;
      jtype = type[jseq];
      // jGA_Flags = GA_Flags[jtype];

      // only skip to the next loop  with both iGA_typeCheck == jGA_typeCheck ==
      // True
      //  or iGA_typeCheck == jGA_typeCheck == False
      // if (not (iGA_typeCheck || jGA_typeCheck) or (iGA_typeCheck &&
      // jGA_typeCheck)) {
      if (iGA_Flags == (GA_Flags[jtype] != 0)) {
        continue;
      }

      delx = xtmp - x[jseq][0];
      dely = ytmp - x[jseq][1];
      delz = ztmp - x[jseq][2];
      if (fabs(delx) > cut_coul || fabs(dely) > cut_coul || fabs(delz) > cut_coul) continue;

      rsq = delx * delx + dely * dely;
      if (rsq > cut_coulsq) continue;

      rsq += delz * delz;
      if (rsq > cut_coulsq) continue;

      value = 0.5 * force->pair->single(iseq, jseq, itype, jtype, rsq, factor_coul, factor_lj, fforce);

      if (!(iGA_Flags ^ to_GA_Flag))
        cl_atom[iseq] += value;
      else
        cl_atom[jseq] += value;
    } // jj
  }   // ii

  if (newton_pair)
    // Communicate cl_atom
    fix_reverse_comm(REV_COUL_RM);
}

void FixConpGA::localGA_neigh_build()
{
  FUNC_MACRO(0);
  if (list == nullptr) {
    list = force->pair->list;
    // printf("Set list to %x\n", list);
  }
  // printf("CPU: %d XXX4\n", me);
  int i, ii, j, jj, inum, jnum, itype, jtype, itag, jtag;
  double* special_coul = force->special_coul;
  double* special_lj   = force->special_lj;
  double factor_coul = 1, factor_lj = -1;
  double qtmp, xtmp, ytmp, ztmp, value;
  double delx, dely, delz, rsq, fforce, ecoul;
  double** x      = atom->x;
  double* q       = atom->q; // *ecoul_ptr;
  double* cl_atom = electro->cl_atom();
  double* ek_atom = nullptr;
  if (force->kspace) ek_atom = force->kspace->eatom;
  int ilocalGA = 0, numneigh_GA, numlocalneigh_GA, numghostneigh_GA, localGA_firstneigh_STLlist_size;

  int* type = atom->type;
  int* jlist;

  int *ilist, *numneigh, **firstneigh;
  int nlocal = atom->nlocal, nghost = atom->nghost, *tag = atom->tag;
  // loop over neighbors of my atoms
  inum       = list->inum;
  ilist      = list->ilist;
  numneigh   = list->numneigh;
  firstneigh = list->firstneigh;

  // for (ii = 0; ii < inum; ii++) {
  // localGA_build()
  mul_tag_To_extraGAIndex.empty();
  localGA_numneigh.reserve(localGA_num + 1);
  localGA_numlocalneigh.reserve(localGA_num + 1);
  localGA_numneigh_disp.reserve(localGA_num + 1);
  /*
  localGA_extraU.reserve(localGA_num+1);
  // Zero the initial potential.
  ecoul_ptr = &localGA_extraU.front();
  for (i = 0; i < localGA_num; ++i)
  {
    ecoul_ptr[i] = 0.0;
  }
  */
  // std::vector<int> localGA_neighor_num(localGA_num);
  // std::list<int> localGA_firstneigh_STLlist;
  std::list<std::pair<int, double>> localGA_firstneigh_STLlist;

  std::vector<std::pair<int, double>> localGA_localneigh_vec;
  std::vector<std::pair<int, double>> localGA_ghostneigh_vec;
  localGA_localneigh_vec.reserve(nlocal + 1);
  localGA_ghostneigh_vec.reserve(nghost + 1);
  std::pair<int, double>* localGA_localneigh_ptr = &localGA_localneigh_vec.front();
  std::pair<int, double>* localGA_ghostneigh_ptr = &localGA_ghostneigh_vec.front();
  const int* const localGA_seq_arr               = &localGA_seq.front();
  // map<std::pair<int, int>, int> pair_connect;
  int repeated_pair_num = 0;

  pcg_neighbor_size_local = 0;

  double pcg_cut = cut_coul + cut_coul_add, pcg_cutsq = pcg_cut * pcg_cut;
  // TODO comment the next line to do PCG radiu test
  pcg_cutsq = pcg_cutsq > cut_coulsq ? pcg_cutsq : cut_coulsq;
  // double pcg_cut = cut_coul;
  // double pcg_cutsq = cut_coul * cut_coul;
  // printf("CPU: %d nlocal: %d\n", me, nlocal);
  // for(i = 0; i < nlocal; ++i){
  // for(ii = 0; ii < inum; ++ii){
  if (me == 0)
    utils::logmesg(lmp,
                   "[ConpGA] localGA_neigh_build() pcg_cut = {:f} from [{:f}+{:f}] "
                   "pcg_cutsq = {:f}\n",
                   pcg_cut, cut_coul, cut_coul_add, pcg_cutsq);

  localGA_numneigh_max = 0;
  for (ii = 0; ii < localGA_num; ++ii) {
    /*
    i = ilist[ii];
    itype = type[i];
    if(GA_typeCheck[itype] == false) {
      continue;
    }
    */
    i     = localGA_seq_arr[ii];
    itype = type[i];

    // To half the value of localGA_localneigh_coeff
    qtmp             = q[i] * 2;
    xtmp             = x[i][0];
    ytmp             = x[i][1];
    ztmp             = x[i][2];
    jlist            = firstneigh[i];
    jnum             = numneigh[i];
    numlocalneigh_GA = numghostneigh_GA = 0;
    // printf("%d neighbor#:%d\n", i, jnum);

    itag = tag[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      // factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;
      jtype = type[j];
      if (GA_Flags[jtype] == 0) {
        continue;
      }

      jtag = tag[j];
      /*
      std::pair<int, int> a0 = std::make_pair(jtag, itag);
      std::pair<int, int> a1 = std::make_pair(itag, jtag);
      */

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq  = delx * delx + dely * dely + delz * delz;
      // skipp pair distance > cut_coulsq
      // if (rsq < pcg_cutsq)
      if (pcg_mat_sel != PCG_MAT::FULL && rsq > pcg_cutsq) continue;

      pcg_neighbor_size_local++;
      // printf("%d %d %f/%f\n", i, j, rsq, cut_coulsq);
      /*
      if (pair_connect.count(a0) == 1 || pair_connect.count(a1) == 1 ) {
        //printf("repeated map: i:%d, itag%d; j:%d, jtag:%d\n", i, itag, j,
      jtag); repeated_pair_num++;
      }
      else
      {
        pair_connect[a0] = 1;
      }
      */
      // ecoul = force->pair->single(i, j, itype, jtype, rsq, factor_coul,
      // factor_lj, fforce);
      ecoul = 0.5 * electro->single(itype, jtype, rsq);
      // printf("%d %d %d %d %f %f\n", i, j, itype, jtype, rsq, ecoul);
      if (j < nlocal) {
        // localGA_localneigh_ptr[numlocalneigh_GA].first=j; // Modfy
        localGA_localneigh_ptr[numlocalneigh_GA].first  = tag[j]; // Modify
        localGA_localneigh_ptr[numlocalneigh_GA].second = ecoul;  // / (qtmp * q[j]);
        numlocalneigh_GA++;
      } else {
        localGA_ghostneigh_ptr[numghostneigh_GA].first  = tag[j];
        localGA_ghostneigh_ptr[numghostneigh_GA].second = ecoul; //  / (qtmp * q[j]);
        numghostneigh_GA++;
      }
      // ecoul_ptr[i] += ecoul;
      // ecoul_ptr[j] += ecoul;

    } // jj
    // std::sort(localGA_localneigh_ptr, localGA_localneigh_ptr +
    // numlocalneigh_GA); std::sort(localGA_ghostneigh_ptr,
    // localGA_ghostneigh_ptr + numghostneigh_GA);
    localGA_firstneigh_STLlist.insert(localGA_firstneigh_STLlist.end(), localGA_localneigh_ptr, localGA_localneigh_ptr + numlocalneigh_GA);
    localGA_firstneigh_STLlist.insert(localGA_firstneigh_STLlist.end(), localGA_ghostneigh_ptr, localGA_ghostneigh_ptr + numghostneigh_GA);
    localGA_numlocalneigh[ilocalGA] = numlocalneigh_GA;
    localGA_numneigh[ilocalGA]      = numlocalneigh_GA + numghostneigh_GA;
    localGA_numneigh_max            = localGA_numneigh_max > localGA_numneigh[ilocalGA] ? localGA_numneigh_max : localGA_numneigh[ilocalGA];
    ilocalGA++;
    // if (me == work_me) printf("[i] %d %d \n", numlocalneigh_GA,
    // numghostneigh_GA); if (me == work_me) printf("%d %d %d|",
    // numlocalneigh_GA, numghostneigh_GA, numlocalneigh_GA + numghostneigh_GA);
  } // ii
  if (me == 0) utils::logmesg(lmp, "[ConpGA] The maximum number of neighbors is: {}\n", localGA_numneigh_max);

  // for(ii = 0; ii < inum; ++ii){
  /*
  for(ii = 0; ii < localGA_num; ++ii){
    i = localGA_seq_arr[ii];
    itype = type[i];

    // To half the value of localGA_localneigh_coeff
    qtmp = q[i] * 2;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    numlocalneigh_GA = numghostneigh_GA = 0;
    //printf("%d neighbor#:%d\n", i, jnum);
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      //factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;
      jtype = type[j];
      if(GA_Flags[jtype] == false) {
        continue;
      }
      #ifndef CONP_NO_DEBUG
      if (me == work_me && DEBUG_LOG_LEVEL > 1) printf("[i] %d %d, tag[%d] from
  %d ilist; [j] %d tag:%d\n", i, ii, tag[i], jnum, j, tag[j]); #endif } // jj }
  // ii
  */
  // printf("CPU: %d XXX2\n", me);
  std::list<std::pair<int, double>>::iterator it;
  localGA_firstneigh_STLlist_size = localGA_firstneigh_STLlist.size();
  localGA_firstneigh.reserve(localGA_firstneigh_STLlist_size);
  localGA_firstneigh_coeff.reserve(localGA_firstneigh_STLlist_size);
  for (i = 0, it = localGA_firstneigh_STLlist.begin(); i < localGA_firstneigh_STLlist_size; ++it, i++) {
    localGA_firstneigh[i]       = it->first;
    localGA_firstneigh_coeff[i] = it->second;
  }
  // std::copy(localGA_firstneigh_STLlist.begin(),
  // localGA_firstneigh_STLlist.end(), localGA_firstneigh.begin());

  // std::copy(localGA_firstneigh_coeff_STLlist.begin(),
  // localGA_firstneigh_coeff_STLlist.end(), localGA_firstneigh_coeff.begin());
  localGA_numneigh_disp[0] = 0;
  if (localGA_num > 0) {
    for (int i = 1; i < localGA_num; ++i) {
      localGA_numneigh_disp[i] = localGA_numneigh_disp[i - 1] + localGA_numneigh[i - 1];
    }
  }

  // printf("CPU: %d XXX1\n", me);
  for (i = 0; i < localGA_num; ++i) {
    // value = cl_atom[localGA_seq[i]] + cl_atom[localGA_seq[i]];
    // value += ek_atom[localGA_seq[i]] + ek_atom[localGA_seq[i]];
    // ecoul_ptr[i] -= value;
    // ecoul_ptr[i] /= -q[i];
    // printf("CPU: %d, Atom[%d/%d] potential:%f %fV\n", me, i, localGA_num,
    // ecoul_ptr[i], cl_atom[localGA_seq[i]]);
  }
  /*
  for (int i = 0; i < neighbor_ele_local_list.size(); ++i)
  {
    printf("%d %d %f\n", i, neighbor_ele_local[i], neighbor_ele_coeff[i]);
  }
  */
  /*for (int i = 0; i < localGA_num; ++i) {
    printf("%d %d %f\n", i, ilocalGA, localGA_extraU[i]);
  }
  */
  // neighbor_ele_local_list.clear();
  // neighbor_coeff_local_list.clear();
  // printf("CPU %d End of Pre_Force %d %d\n", me,
  // neighbor_ele_local_list.size(), neighbor_coeff_local_list.size());
  // printf("CPU[%d]: size %d, repeated: %d\n", me,
  // localGA_firstneigh_STLlist_size, repeated_pair_num);
  /*
  for(ii = 0; ii < inum; ++ii) {
    i = ilist[ii];
    if(i >= nlocal)
      printf("iseq = %d, ii = %d\n", i, ii);
  }
  exit(0);
  */
}

/* To obtain Umat at each step */
void FixConpGA::calc_Umat()
{
  FUNC_MACRO(0);
  int i, iseq, nlocal = atom->nlocal;
  double* q                   = atom->q;
  double* const cl_atom       = electro->cl_atom();
  double* const ek_atom       = force->kspace->eatom;
  int* type                   = atom->type;
  int* const localGA_seq_arr  = &localGA_seq.front();
  double* const q_store_arr   = &q_store.front();
  double* const env_alpha_arr = &env_alpha.front();
  for (i = 0; i < nlocal; ++i) {
    q_store_arr[i] = q[i];
    q[i]           = 1;
  }
  double* Umat_1D;
  memory->create(Umat_1D, localGA_num, "Fix:Conp Umat_1D");
  force_clear();
  fix_forward_comm(0);
  fix_localGA_pairenergy_flag(0, true);
  force->kspace->qsum_qsq();
  electro->compute(3, 0);

  for (i = 0; i < localGA_num; ++i) {
    iseq          = localGA_seq_arr[i];
    Umat_1D[i]    = ek_atom[iseq] + cl_atom[iseq] - env_alpha_arr[i];
    cl_atom[iseq] = 0.0;
  }
  env_Umat.push_back(Umat_1D);
  // write_mat(Umat_1D, localGA_num, "Usave.mat");
  std::copy(q_store_arr, q_store_arr + nlocal, q);
  fix_forward_comm(0);
  electro->qsum_qsq();
}

/* ---------------------------------------------------------------------- */
// The routine to calulcate the atom dependent alpha for each GA atom
// It was calculated via the Least Squared methods.
/* ---------------------------------------------------------------------- */

void FixConpGA::calc_env_alpha()
{
  int M                       = env_Umat.size(), i, j;
  double* const env_alpha_arr = &env_alpha.front();
  double *Umat = nullptr, *Umat_1D = nullptr, *Umat_1D_total = nullptr;
  const int* const size_EachProc = &localGA_EachProc.front();
  const int* const size_Dipl     = &localGA_EachProc_Dipl.front();

  if (me == 0) {
    memory->create(Umat, totalGA_num * (M + 1), "Fix:Conp Umat_1D");
    memory->create(Umat_1D_total, totalGA_num, "Fix:Conp Umat_1D");

  } else {
    Umat_1D_total = env_alpha_arr;
  }

  for (i = 0; i < M; ++i) {
    Umat_1D = env_Umat.front();
    MPI_Gatherv(Umat_1D, localGA_num, MPI_DOUBLE, Umat_1D_total, size_EachProc, size_Dipl, MPI_DOUBLE, 0, world);
    env_Umat.pop_front();
    memory->destroy(Umat_1D);
    for (j = 0; j < totalGA_num; ++j) {
      // printf("w = %d, %d, %d\n", i, j, i+j*M);
      Umat[i + j * M] = Umat_1D_total[j];
    }
  }
  Umat_1D = nullptr;

  if (me == 0) {
    int max_size = M > totalGA_num ? M : totalGA_num;
    memory->grow(Umat_1D_total, max_size, "Fix:Conp Umat_1D_total again");
    for (i = 0; i < M; ++i)
      Umat_1D_total[i] = double(totalSOL_num);
    // write_mat(Umat, totalGA_num * M, "Umat_solution.mat");
    solve_LS(Umat, Umat_1D_total, 'T', M, totalGA_num);
    // write_mat(Umat_1D_total, totalGA_num, "Umat_1D_solution.mat");
    memory->destroy(Umat);
  }
  Umat             = nullptr;
  __env_alpha_flag = false;
}

/* ---------------------------------------------------------------- */
// Function to obtain the extermum value
// two functions invoke it
//   from setup()
//   from pre_forece()
void FixConpGA::equalize_Q()
{
  double* cl_atom = electro->cl_atom();
  double* ek_atom = electro->ek_atom();

  // fix_localGA_pairenergy(GA2GA_flag);
  // norm_mat_p(cl_atom, localGA_num, "Classic");

#ifndef CONP_NO_DEBUG
  force_clear();
#else
  clear_cl_atom();
#endif

  time_t t_begin0 = clock();
  switch (minimizer) {
    // MIN::NEAR
    case MIN::NEAR:
      electro->pair_SOLtoGA(GA2GA_flag);
      electro->calc_GA_potential(STAT::POT, electro->get_ek_stat(), SUM_OP::OFF);
      update_P_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;
      break;

    // MIN::FAR
    case MIN::FAR:
      electro->compute_SOLtoGA(GA2GA_flag);
      update_K_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;

      t_begin0 = clock();
      electro->calc_GA_potential(electro->get_cl_stat(), STAT::POT, SUM_OP::OFF);
      update_P_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;
      break;

    // OTHER
    default:
      // electro->compute(3, 0);
      electro->compute_SOLtoGA(GA2GA_flag);
      // electro->compute_GAtoGA();
      // set_zeros(&ek_atom[0], 0, localGA_num);
      update_K_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;
      // for
      // norm_mat_p(cl_atom, localGA_num, "Before");
      // norm_mat_p(cl_atom, localGA_num, "Comm..");
      t_begin0 = clock();
      electro->pair_SOLtoGA(GA2GA_flag);
      electro->calc_GA_potential(STAT::POT, STAT::POT, SUM_OP::ON);
      update_P_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;
      break;
  }

  // Extra force calculation in real and kspace are necessary//
  // Because the position of

  // Convert cl_atom and ek_atom into POTential and sum together;

  /*
      \sum_{j}\Delta q_je_{ij} = \chi_i -\sum_{I}q_I e_{iI}-\sum_{j}q^0_je_{ij}
      Where $q_i^0$ is the equilibrium charge in the last MD step
      $\Delta q_i$ is the change of the charge quantity on $i$th electrode atom.
  */
  switch (minimizer) {
    case MIN::INV:
      update_charge_AAA();
      break;
    case MIN::CG:
      update_charge_cg();
      break;
    case MIN::PCG:
      update_charge_pcg();
      break;
    case MIN::PCG_EX:
      update_charge_ppcg();
      break;
    case MIN::NEAR:
      update_charge_near();
      break;
    case MIN::NONE:;
      break;
    default:
      error->all(FLERR, "Illegal fix conp command for unknown minimization method");
  }
}

/* ---------------------------------------------------------------------- */

void FixConpGA::pre_force(int vflag)
{
  if (__env_alpha_flag && update->ntimestep % env_alpha_stepsize == 0) {
    calc_Umat();
  }
  if (update->ntimestep % everynum != 0) {
    return;
  };

  FUNC_MACRO(0) double **f = atom->f, force_store[3] = {0, 0, 0};
  // bool clear_force = true;
  update_K_time = update_P_time = 0.0;
  /*
  double *epair = force->pair->cl_atom;
  double *eatom = force->kspace->eatom;
  int nlocal = atom->nlocal, nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  double *q = atom->q;
  */
  time_t t_begin0 = clock();
  localGA_atomsort();
  update_P_time += double(clock() - t_begin0) / CLOCKS_PER_SEC;

  // B0_refresh_time = 3;
  if (tol_style == TOL::REL_B && update->ntimestep > 0) {
#ifndef CONP_NO_DEBUG
    calc_B0(B0_next);
    force_clear();
    double B0_tol = check_tol * 30;
    for (int i = 0; i < localGA_num; ++i) {
      if (!isclose(B0_prev[i], B0_next[i], B0_tol)) {
        double B0_error = (B0_prev[i] - B0_next[i]) / B0_next[i];
        char info[512];
        sprintf(info,
                "Error: Inconsistent B0 by the electrode alone from the equilibrium charge in the "
                "last step for [%i] %.16f vs %.16f error: %g\n",
                i, B0_prev[i], B0_next[i], B0_error);
        error->warning(FLERR, info);
      }
    }
// double * B_tmp; B_tmp = B0_next; B0_next = B0_prev; B0_prev = B_tmp;
#endif
    if (update->ntimestep % B0_refresh_time == 0) {
      calc_B0(B0_prev);
      force_clear();
    }
  }

  if (neutrality_flag && update->ntimestep % qsqsum_refresh_time == 0) {
    electro->qsum_qsq();
  }

  if (update_Vext_flag) update_Vext();

  // force_clear();
  //  t_begin0        = clock();
  equalize_Q();
  if (clear_force_flag) force_clear();

  if (force_analysis_flag) {
    force_analysis(3, vflag);
  }

  // t_end = clock();
  // if (me == work_me) utils::logmesg(lmp, "Time Elps for energy min
  // {:8.5f}\n", double(t_end - t_begin) / CLOCKS_PER_SEC);
  // printf("[4]fz2001 = %g\n", f[2000][0]);

  /*for (size_t i = 0; i < ntotal; i++)
  {
    printf("Q2[%lu] = %f ", i, q[i]);
  }
  printf("\n");*/

  // memory->create(d, 1000, "FixConpGA:nmax");
  /*
  kspace_eatom_clear();
  //force->pair->compute(999, 0);
  printf("After charge update force->kspace energy = %f\n",
  force->kspace->energy); force->kspace->compute(999, 0); nlocal = atom->nlocal;
  double value(0.0), *kspace_eatom = force->kspace->eatom;
  for (int i = 0; i < nlocal; i++) {
    value += kspace_eatom[i];
  }
  printf("After charge update force->kspace energy = %f vs %f\n",
  force->kspace->energy, value);
  */

  /*
  printf("pair_compute\n");
  printf("epair: %f %f\n", epair[0], epair[1]);
  printf("epair: %f %f\n", eatom[0], eatom[1]);
  */
  // class Pair *pair = nullptr;
  // pair = force->pair_match("lj/cut/gauss/long",1);
  // double fforce, energy;
  /*
  energy = force->pair->single(0, 1, 1, 2, 1, 1, 0, fforce);
  printf("single:Test %f %f \n", epair[1], energy/2);
  energy = force->pair->single(0, 1, 3, 3, 1, 1, 0, fforce);
  printf("single:Test %f %f \n", epair[1], energy/2);
  */

  // GG_inter_coeff();
  // printf("CPU %d Out of Pre_Force\n", me);
  // force_clear();
  /*
  {
  qsum_qsq();
  force->pair->compute(999, 0);
  if (force->kspace) // Extra Kspace Check
    force->kspace->compute(999, 0);

  double value;
  int* const GA_seq_arr = &localGA_seq.front();
  int i, iseq, *type = atom->type;
  double *q = atom->q, qtmp, res = 0, tmp;
  for (i = 0; i < localGA_num; i++)
  {
    iseq = GA_seq_arr[i];
    value = force->pair->cl_atom[iseq] + force->kspace->eatom[iseq];
    qtmp = q[iseq] + 1e-20;
    value /= qtmp;
    value += self_ECpotential;
    //res_localGA[i] = p_localGA[i] = GA_typeUextra[type[iseq]] - value;
    tmp = GA_typeUextra[type[iseq]] - value;
    res += tmp*tmp;
  }
  printf("res_2 = %20.18g\n", res);
  }
  */
  // For the validation check with XXX

  // force_clear();
  // printf("end of pre_force()\n");
  if (vflag != -1) {
    Ele_iter++;
    switch (Ele_Cell_Model) {
      case CELL::POT:
        for (int i = 0; i < Ele_num; i++) {
          Ele_qave[i] += Ele_qsum[i];
          Ele_qsq[i] += Ele_qsum[i] * Ele_qsum[i];
        }
        break;
      case CELL::Q:
        double Uchem0 = Ele_Uchem[0];
        for (int i = 0; i < Ele_num; i++) {
          double dU = Ele_Uchem[i] - Uchem0;
          Ele_qave[i] += dU;
          Ele_qsq[i] += dU * dU;
        }
        break;
      default:;
    }
  }
}

/*
void FixConpGA::kspace_eatom_clear() {
    int i, n = atom->nlocal;
    double* kspace_eatom = force->kspace->eatom;
    if (force->kspace->tip4pflag) n += atom->nghost;
    for (i = 0; i < n; i++) kspace_eatom[i] = 0.0;
    force->kspace->energy = 0.0;
}

void FixConpGA::pair_eatom_clear() {
    int i, n = atom->nlocal;
    double* eatom = force->pair->eatom;
    double* cl_atom = force->pair->cl_atom;
    //if (force->kspace->tip4pflag) n += atom->nghost;
    for (i = 0; i < n; i++) {
      eatom[i] = 0.0;
      cl_atom[i] = 0.0;
    }
    force->pair->energy = 0.0;
}
*/

// Forward from Local-Atom to Ghost atom;
void FixConpGA::fix_forward_comm(int pflag)
{
  // int bordergroup_store = comm->bordergroup;
  int n;
  pack_flag = pflag;
  // return comm->forward_comm_fix(this);

  if (firstgroup_flag == false) {
    comm->forward_comm_fix(this);
  } else {
    // printf("electro->forward_comm\n");
    electro->forward_comm(pflag);
    /*
    double* const ghostGA_value = electro->ghost_buf;
    if (pflag == 0) {
      double* q = atom->q;
      for (int i = 0; i < all_ghostGA_num; i++) {
        int iseq = all_ghostGA_seq[i];
        // int iind = all_ghostGA_ind[i];
        // if (q[iseq] != ghostGA_value[iind])
        //   printf("q[%d] %f vs g[%d] %f\n", iseq, q[iseq], iind,
        //   ghostGA_value[iind]);
        q[iseq] = ghostGA_value[i];
      }
    } else if (pflag == 1) {
      // comm->forward_comm_fix(this);
      double* cl_atom = electro->cl_atom();
      for (int i = 0; i < all_ghostGA_num; i++) {
        int iseq = all_ghostGA_seq[i];
        // int iind      = all_ghostGA_ind[i];
        cl_atom[iseq] = ghostGA_value[i];
      }
    }
    */

    // bordergroup = comm->bordergroup;
    // comm->bordergroup = 1; // On-limit group comm
    // comm->forward_comm_fix(this);
    // comm->bordergroup = bordergroup; // Off-limit group comm
  }
}

// Reverse from Ghost atom to Local-Atom
void FixConpGA::fix_reverse_comm(int pflag)
{
  int n;
  pack_flag = pflag;
  // if(firstgroup == -1) {
  // return comm->reverse_comm_fix(this);

  if (firstgroup_flag == false) {
    comm->reverse_comm_fix(this);
  } else {
    electro->reverse_comm(pflag);
    /*
    double* const ghostGA_value = electro->ghost_buf;
    set_zeros(ghostGA_value, 0, used_ghostGA_num);
    if (pflag == REV_COUL_RM) {
      double* cl_atom = electro->cl_atom();
      for (int i = 0; i < all_ghostGA_num; i++) {
        int iseq = all_ghostGA_seq[i];
        // int iind = all_ghostGA_ind[i];
        ghostGA_value[i] += cl_atom[iseq];
        cl_atom[iseq] = 0;
      }
    } else if (pflag == REV_Q) {
      double* q = atom->q;
      for (int i = 0; i < all_ghostGA_num; i++) {
        int iseq = all_ghostGA_seq[i];
        // int iind = all_ghostGA_ind[iseq];
        ghostGA_value[i] += q[i];
      }
    } else if (pflag == REV_COUL) {
      double* cl_atom = electro->cl_atom();
      for (int i = 0; i < all_ghostGA_num; i++) {
        int iseq = all_ghostGA_seq[i];
        // int iind = all_ghostGA_ind[i];
        ghostGA_value[i] += cl_atom[iseq];
      }
    }
    */
  }
}

/* ----------------------------------------------------------------------
   Thermo_output of Uchem variables
------------------------------------------------------------------------- */

double FixConpGA::compute_scalar()
{
  if (unit_flag == 1)
    return Uchem * inv_qe2f;
  else
    return Uchem;
}

double FixConpGA::compute_vector(int n)
{
  // printf("n = %d\n", n);
  switch (n) {
    case 0:
      return (double)curriter;
      break;
    case 1:
      return update_Q_time;
      break;
    case 2:
      return update_Q_totaltime;
      break;
    case 3:
      return update_P_time;
      break;
    case 4:
      return update_P_totaltime;
      break;
    case 5:
      return update_K_time;
      break;
    case 6:
      return update_K_totaltime;
      break;
    case 7:
      return force_analysis_res;
    case 8:
      return force_analysis_max;
    case 9:
      return 0;
    default:
      error->all(FLERR, "Invalid Variable index for fix conp/GA!");
  }
  return 0;
}

void FixConpGA::comm_REV_COUL_RM()
{
  if (newton_pair) {
    fix_reverse_comm(REV_COUL_RM);
  }
}

int FixConpGA::pack_forward_comm(int n, int* list, double* buf, int pbc_flag, int* pbc)
{
  int m;
  if (pack_flag == 0) { // transfer charge quantity
    double* q = atom->q;
    for (m = 0; m < n; m++)
      buf[m] = q[list[m]];
  } else if (pack_flag == 1) {
    double* cl_atom = electro->cl_atom();
    for (m = 0; m < n; m++)
      buf[m] = cl_atom[list[m]];
  }
  // printf("[Cpu:%d] fix_forward_comm():: n= %d, pack_flag = %d\n", me, m,
  // pack_flag);
  return m;
}

void FixConpGA::unpack_forward_comm(int n, int first, double* buf)
{
  int i, m;
  if (pack_flag == 0) {
    double* q = atom->q;
    for (m = 0, i = first; m < n; m++, i++)
      q[i] = buf[m];
  } else if (pack_flag == 1) {
    double* cl_atom = electro->cl_atom();
    for (m = 0, i = first; m < n; m++, i++) {
      cl_atom[i] = buf[m]; // printf("%d %d %f\n", me, i, cl_atom[i]);
    };
  }
}

int FixConpGA::pack_reverse_comm(int n, int first, double* buf)
{
  int i, m;
  double* q       = atom->q;
  double* cl_atom = electro->cl_atom();

  switch (pack_flag) {
    case REV_Q:
      for (m = 0, i = first; m < n; m++, i++)
        buf[m] = q[i];
      break;
    case REV_COUL:
      for (m = 0, i = first; m < n; m++, i++)
        buf[m] = cl_atom[i];
      break;
    case REV_COUL_RM:
      for (m = 0, i = first; m < n; m++, i++) {
        buf[m]     = cl_atom[i];
        cl_atom[i] = 0;
      }

      break;
  }
  return m;
}

void FixConpGA::unpack_reverse_comm(int n, int* list, double* buf)
{
  int m;
  double* q       = atom->q;
  double* cl_atom = electro->cl_atom();
  switch (pack_flag) {
    case REV_Q:
      for (m = 0; m < n; m++)
        q[list[m]] += buf[m];
      break;
    case REV_COUL:
    case REV_COUL_RM:
      for (m = 0; m < n; m++)
        cl_atom[list[m]] += buf[m];
      break;
  }
}
