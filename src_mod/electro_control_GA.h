/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   Electro_Control_GA: An independent Electro Control Class for CONP fix
------------------------------------------------------------------------- */

#ifndef LMP_ELECTRO_CONTROL_H
#define LMP_ELECTRO_CONTROL_H

#include "fix.h"
#include "func_macro.h"
#include "kspace_conp_GA.h"
#include "lmptype.h"
#include "pair_conp_GA.h"
#include "pointers.h"

namespace LAMMPS_NS
{
class FixConpGA;
class Electro_Control_GA : protected Pointers {
  bool kspace_conp_Flag;
  int me;
  int compute_id;
  int FIX_groupbit;
  STAT cl_stat;
  STAT ek_stat;
  int& DEBUG_LOG_LEVEL;
  int& Ele_num;
  double*& Ele_qsum;
  int*& localGA_eleIndex;
  int nswap, iswap_me;
  int* sendproc;
  int* recvproc;
  int* sendnum;
  int* recvnum;
  double* buf_send;
  double* buf_recv;
  int** sendlist;
  int** recvlist;

public:
  double* ghost_buf;
  bool DEBUG_SOLtoGA, DEBUG_GAtoGA;
  FixConpGA* FIX;
  KSpaceConpGA* kspace;
  Pair_CONP_GA* pair;
  double qsum_SOL, qsqsum_SOL;
  Electro_Control_GA(class LAMMPS*, FixConpGA*);
  ~Electro_Control_GA();
  void setup();
  double single(int i, int j, double rsq) { return pair->compute_single(i, j, rsq); }

  int* setup_comm(int*);
  void forward_comm(int);
  void reverse_comm(int);
  int pack_forward_comm(int, int, int*, double*);
  void unpack_forward_comm(int, int, int*, double*);
  int pack_reverse_comm(int, int, int*, double*);
  void unpack_reverse_comm(int, int, int*, double*);
  void pre_reverse();
  double* ek_atom();
  double* cl_atom();
  bool is_kspace_conp_GA() { return kspace_conp_Flag; };
  bool is_compute_AAA()
  {
    if (kspace_conp_Flag) {
      bool result = kspace->is_compute_AAA();
      return result;
    } else
      return false;
  }
  void compute_addstep(bigint n);
  double** compute_AAA(double**);
  void compute(int, int);
  void set_qsqsum(double, double);
  void calc_qsqsum_SOL();
  double calc_qsum();
  double calc_qsqsum();
  void qsum_qsq();
  void compute_SOLtoGA(bool);
  void compute_GAtoGA();
  void calc_GA_potential(STAT, STAT, SUM_OP);
  void conv_GA_potential(int, int, bool);
  void conv_GA_potential_both(STAT, STAT);
  void update_qstat();
  void pair_SOLtoGA(bool);
  void pair_GAtoGA();
  const char* type_name()
  {
    if (!kspace_conp_Flag) {
      return "the default electrostatic solver";
    } else
      return kspace->type_name();
  }
  STAT get_ek_stat() { return ek_stat; }
  STAT get_cl_stat() { return cl_stat; }

  /*
  virtual void compute_vector(bigint *, double *) = 0;
  virtual void compute_vector_corr(bigint *, double *) = 0;
  virtual void compute_matrix(bigint *, double **) = 0;
  virtual void compute_matrix_corr(bigint *, double **) = 0;
  */
};
} // namespace LAMMPS_NS

#endif
