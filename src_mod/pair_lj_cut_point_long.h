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


   PairStyle("lj/cut/point/long", PairLJCutPointLong)
   A pair_style for Conp/GA between the Gaussian interaction

   Version: 1.00 Data: Nov/19/2020
   Kylin (WUST)
   A coulombic interaction for point charge with a virtual gaussian-like
   self energy. The reciprocal interaction was consistent with the Point
   charge version of Ewald summation or PPPM in Lammps

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST
   TODO: SOL_to_GA calculation not include the FACTOR_COUL
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

// clang-format off
PairStyle(lj/cut/point/long, PairLJCutPointLong)
// clang-format on

#else

#ifndef LMP_PAIR_LJ_CUT_POINT_LONG_H
#define LMP_PAIR_LJ_CUT_POINT_LONG_H

#include "pair.h"
#include "pair_conp_GA.h"

namespace LAMMPS_NS
{

class PairLJCutPointLong : public Pair, public Pair_CONP_GA {

public:
  PairLJCutPointLong(class LAMMPS*);
  virtual ~PairLJCutPointLong();
  virtual void compute(int, int);
  virtual void settings(int, char**);
  void coeff(int, char**);
  virtual void init_style();
  virtual double init_one(int, int);
  void write_restart(FILE*);
  void read_restart(FILE*);
  virtual void write_restart_settings(FILE*);
  virtual void read_restart_settings(FILE*);
  void write_data(FILE*);
  void write_data_all(FILE*);
  virtual double single(int, int, int, int, double, double, double, double&);

  void init_special_tables();

  // void compute_inner();
  // void compute_middle();
  // virtual void compute_outer(int, int);
  virtual void* extract(const char*, int&);
  double* eta;            // Guassian distribution width for type
  double** eta_mix_table; // Guassian interaction factor width between different type
  double* sigma_width;    // Inverse Guassian distribution width for mixing
                          // 1/sqrt(value*value*2);
  CHARGE* isGCharge;      // 0 for pure PCharge and 1 for Pcharge with self interaction

  int n_gauss, *n_gauss_type, n_plus;
  double **ftable_list, **dftable_list, **ctable_list;
  double **dctable_list, **etable_list, **detable_list;
  double **ftable_vec, **dftable_vec, **ctable_vec;
  double **dctable_vec, **etable_vec, **detable_vec;

  virtual double pair_SOLtoGA_energy(int, bool, bool);
  virtual double pair_GAtoGA_energy(int, bool);
  virtual double compute_single(int, int, double);
  virtual void compute_Group(int, int, int, bool);
  virtual void clear_saved();

protected:
  void compute_Gauss(int eflag, int vflag);
  void compute_Point(int eflag, int vflag);
  void compute_Point_Group(int eflag, int vflag, int groupbit, bool);
  void compute_Gauss_Group(int eflag, int vflag, int groupbit, bool);

  void add_saved_pair(int eflag, int vflag, int groupbit);

  double pair_SOLtoGA_energy_Point(int, bool, bool);
  double pair_GAtoGA_energy_Point(int, bool);

  double pair_SOLtoGA_energy_Gauss(int, bool, bool);
  double pair_GAtoGA_energy_Gauss(int, bool);

  double cut_lj_global;
  double **cut_lj, **cut_ljsq;
  double cut_coul, cut_coulsq;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double* cut_respa;
  double qdist; // TIP4P distance from O site to negative charge
  double g_ewald;
  double* self_Greenf;
  char self_flag;
  bool gauss_flag;
  /* for pair */
  int pair_num, pair_num_allocate;
  int* pair_ij;
  double *pair_ecoul, *pair_evdwl, *pair_fcoul, *pair_fvdwl, *pair_dxyz;
  virtual void allocate();
};

} // namespace LAMMPS_NS

#endif
#endif

    /* ERROR/WARNING messages:

    E: Illegal ... command

    Self-explanatory.  Check the input script syntax and compare to the
    documentation for the command.  You can use -echo screen as a
    command-line option when running LAMMPS to see the offending line.

    E: Incorrect args for pair coefficients

    Self-explanatory.  Check the input script or data file.

    E: Pair style lj/cut/coul/long requires atom attribute q

    The atom style defined does not have this attribute.

    E: Pair style requires a KSpace style

    No kspace style is defined.

    E: Pair cutoff < Respa interior cutoff

    One or more pairwise cutoffs are too short to use with the specified
    rRESPA cutoffs.

    */
