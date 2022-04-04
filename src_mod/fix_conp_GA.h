/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   FixStyle("conp/GA", FixConpGA)
   Fix to deform the SR-CPM simulation.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
  Version 0.0.1 15/05/2019
  by Kylin from WUST
  Constant electric potential method for metal atoms.
  1) Short-range interactions calculation via a sparse matrix multiplication for
with MPI; 2) Long-range calulcation by PPPM; 3) Update the atomic charge on
electrode atoms by Lagrange multiplyer constrain;

  FixStyle(conp/metal/GA,FixGAConp)  Fix_Name only avail for lowercase

  Version 0.0.2 06/11/2019
  by Kylin from WUST
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

// clang-format off
FixStyle(conp/GA, FixConpGA)
// clang-format on

#else

#ifndef LMP_FIX_CONP_GA_H
#define LMP_FIX_CONP_GA_H
#define FIX_CONP_GA_WIDTH 255

// #define CONP_NO_DEBUG
// #define DEBUG_LOG_LEVEL 1

#include "fix.h"

#include "func_macro.h"
#include "pair_conp_GA.h"
#include "spmat_class.h"

#include <ctime>
#include <list>
#include <map>
#include <vector>

#ifdef MKL_ILP64
#include <mkl_types.h>
#else
#define MKL_INT int
#endif

namespace LAMMPS_NS
{

// using map because it's faster;
class Electro_Control_GA;

class FixConpGA : public Fix {
public:
  FixConpGA(class LAMMPS*, int, char**);
  ~FixConpGA();
  int setmask();
  void init();
  void setup(int);
  void pre_force(int);
  // void post_force(int);
  // void end_of_step();
  void pre_reverse(int, int);
  void post_run();
  // void init_list(int, class NeighList *);
  void group_check();
  void localGA_build();
  void localGA_neigh_build();
  void localGA_atomsort();
  void setup_AAA();
  void equalize_Q();
  void setup_once();
  void setup_AAA_Direct();
  void update_charge_AAA();
  void update_charge_cg(bool far_flag = true);
  void update_charge_pcg();
  void update_charge_ppcg();
  void update_charge_near();
  bigint count_type(int itype);
  void pcg_Matrix_Residual(double, double* const, const double* const, const double* const, const int* const, const double* const);
  double compute_scalar();
  double compute_vector(int);
  double** pair_matrix(double**);

  double fit_erfc(double, double, double&);

  Electro_Control_GA* electro;
  int DEBUG_LOG_LEVEL;
  double **DEBUG_fxyz_store, *DEBUG_eatom_store;
  std::clock_t t_begin, t_end;

  std::vector<int> localGA_EachProc, localGA_EachProc_Dipl;

  // solving the constant charge method
  int Ele_num;         // Number of electrode
  CELL Ele_Cell_Model; // Model of CELL
  int Ele_iter;        // total iteration;
  int* Ele_GAnum;      // The number of atoms in each electrode
  double **Ele_matrix, **Ele_D_matrix;
  double *Ele_qsum, *Ele_Uchem, *Ele_qave, *Ele_qsq;
  double* localGA_Vext;      // The external potential on each local electrode atoms
  int* GA_Flags;             // The check if this types belong to an electrode, at value =
                             // 0, means the electrolyte.
  CONTROL* Ele_Control_Type; //
  int* Ele_To_aType;
  std::map<int, double*> VextID_to_ATOM_store;
  int* localGA_eleIndex;
  int *all_ghostGA_ind, *comm_recv_arr, all_ghostGA_num;
  double** GA_Vext_ATOM_data;
  std::vector<int> localGA_numneigh, localGA_numlocalneigh, localGA_numneigh_disp;
  std::vector<int> localGA_tag; // The tags      vector of GA assigned to local CPU
  std::vector<int> localGA_seq; // The sequences vector of GA assigned to local CPU
  std::vector<int> totalGA_tag; // all tags of GA atoms.
  std::vector<double> localGA_firstneigh_coeff, localGA_GreenCoef_vec,
      localGA_firstneigh_inva; // localGA_extraU;
  std::vector<int> localGA_firstneigh, localGA_firstneigh_localIndex;
  std::vector<double> res_localGA_vec, ap_localGA_vec, p_localGA_vec, z_localGA_vec, z_localGA_prv_vec;

  std::vector<int>*pcg_localGA_firstneigh, *pcg_localGA_firstneigh_localIndex;
  std::vector<double>* pcg_localGA_firstneigh_inva;
  std::vector<double> f_store_vec, env_alpha;
  std::list<double*> env_Umat;
  // shift_Ux: Langrage Multiplyer to neutralize the whole system Qsum == 0;
  // double self_Xi_const, self_ECpotential; // Extra interaction energy for
  // charges with its image charge in PBC boundary
  double Xi_self;
  // Ushift the magnitude of the potential shift by the neutraliziation
  // constraint in the last iteration.
  double Uchem, Ushift, gewald_tune_factor, qtotal_SOL,
      qsum_set; // qsum, qsqsum,
  double AAA_sumsum, inv_AAA_sumsum, *self_GreenCoef, cut_coulsq, cut_coul, cut_coul_add;
  bool neutrality_flag, neutrality_state, pair_flag,
      fast_bvec_flag; // self_Xi_flag, nonneutral_removel_flag;
  bool firstgroup_flag, Uchem_extract_flag, valid_check_flag, update_Vext_flag, postreat_flag;
  bool clear_force_flag, Smatrix_flag, force_analysis_flag, GA2GA_flag, selfGG_flag, INFO_flag;
  bool __env_alpha_flag, calc_EleQ_flag;
  bool DEBUG_SOLtoGA, DEBUG_GAtoGA;
  int qvalue_style;
  map<int, int> mul_tag_To_extraGAIndex;
  int *extraGA_num, *extraGA_disp, **ppcg_GAindex, **ppcg_block;
  int bisect_level, bisect_num;
  int env_alpha_stepsize, localSOL_num, totalSOL_num;

  int Xi_self_index;
  SYM symmetry_flag;       // Wether or not to make a symmetric matrix
  int valid_check_firstep; // the first step start valid check in each run;
  double check_tol;        // The absolute toleratance in valid check.
  int work_me;             // the index of proc to print info, default -1;
  PCG_MAT pcg_mat_sel;     // stype of precondition matrix: normal, full, pseudo
  int me, nprocs, runstage, unit_flag, pack_flag;
  int newton_pair, setup_run;

  int everynum;
  int maxiter, curriter;
  int localGA_num, totalGA_num, pcg_neighbor_size_local, localGA_numneigh_max;
  int ghostGA_num, dupl_ghostGA_num, used_ghostGA_num, extGA_num, usedGA_num;
  int true_usedGA_num; // The number of individual Ghost GA atoms.
  int sparse_pcg_num, scalapack_num, size_subAAA;
  size_t size_map;
  double update_Q_totaltime, update_Q_time;
  double update_K_totaltime, update_K_time;
  double update_P_totaltime, update_P_time;

  // int firstgroup; //  The group resorted onto the first group in atom Module
  double tolerance, inv_qe2f, pair_g_ewald, pseuo_tolerance;
  double kspace_g_ewald, force_analysis_res, force_analysis_max;
  int kspace_nn_pppm[3];
  char AAA_save[FIX_CONP_GA_WIDTH + 1];
  int AAA_save_format;
  MIN minimizer;
  TOL tol_style;
  int B0_refresh_time, qsqsum_refresh_time;
  MKL_INT max_lwork, max_liwork, scalapack_nb;

  char** GA_Vext_str;
  int* GA_Vext_info;
  double* GA_Vext_var;
  varTYPE* GA_Vext_type;

  void fix_pair_compute(bool PE_Flag = false, bool COMM_Flag = true);
  void comm_REV_COUL_RM();
  void clear_cl_atom();
  double check_atomvec(double* v, const char* info);
  void zero_solution_check();
  void force_clear();

private:
  void classic_compute_AAA();
  void matrix_scatter(double**& mat_all, double**& mat, int N, const int* S, const int* P);
  void Print_INFO();
  void setup_GG_Flag();
  void calc_B0(double*);
  void make_symmetric_matrix(SYM flag, int n, double** mat);
  void energy_clear();
  void fix_localGA_pairenergy(bool);
  void fix_localGA_pairenergy_flag(bool, bool);
  void fix_forward_comm(int);
  void fix_reverse_comm(int);
  void clear_atomvec(double*, int);
  void gen_DC_invAAA();
  void gen_DC_invAAA_sparse();
  void update_Vext();
  void setup_Vext();
  void setup_CELL();
  void make_ELE_matrix();
  void make_ELE_matrix_direct();
  void make_ELE_matrix_iterative();

  int optimal_split_1D(const double* const p0, const double* const p1);
  double quadratic_fit(double*, double*);
  double quadratic_fit_int(int*, double*);

  // double ppcg_split_STD(double, int&, int&, const int*, int*, const int*);
  // double ppcg_best_split(int index, int**range, int*num_arr, double*
  // split_arr); void check_ppcg_partition(int index, int** range);
  double POST2ND_charge(double, double*, double*);
  double POST2ND_localGA_neutrality(double, double*);
  double POST2ND_solve_ele_matrix(double*, double*);
  int reset_nonzero();
  void calc_Uchem_with_zero_Q_sol();
  void coul_compute(int);
  void solve_LS(double*, double*, char, MKL_INT, MKL_INT);
  // void env_alpha_calc();
  // void group_group_compute(bool, const double* const, double*, const double*
  // const);

  void kspace_eatom_clear();
  void uself_calc();
  void calc_Umat();
  void inverse(double**, int, double**&, int);

  void inv_AAA_LAPACK(double*, MKL_INT);
#ifdef SCALAPACK
  int inv_AAA_SCALAPACK(double*, MKL_INT);
#endif

  void gen_invAAA();
  void sparsify_invAAA();
  void sparsify_invAAA_EX();
  void savetxt_AAA();
  void loadtxt_AAA();
  void load_AAA();
  void save_AAA();
  void meanSum_AAA();
  void make_S_matrix();
  void meanSum_AAA_cg();
  void firstgroup_check();
  int solver_CG(double*, double*, int, double, int);
  double **AAA, *AAA_sum1D, **P_PPCG, **q_iters, **pcg_res_norm;
  double *BBB, *BBB_localGA, *B0_prev, *B0_next, *B0_store,
      *Q_GA_store; //, ,  BBB_by_I, the electric potential by the mobile
                   // electrolyte molecule.
  double pcg_shift, pcg_shift_scale, ppcg_invAAA_cut, *ppcg_shift, *block_ave_value;
  std::vector<double> q_store;
  int pack_forward_comm(int, int*, double*, int, int*);
  void unpack_forward_comm(int, int, double*);
  int pack_reverse_comm(int, int, double*);
  void unpack_reverse_comm(int, int*, double*);
  void pcg_P_shift();
  void pcg_analysis();
  void force_analysis(int, int);
  void energy_calc(int, int, int);
  void calc_env_alpha();

  FILE* pcg_out;
  class NeighList* list;

  /*
  void force_cal(int);
  void update_charge();
  int electrode_check(int);
  void cg();
  void inv();

 private:
  int me,runstage;

  double Btime,Btime1,Btime2;
  double Ctime,Ctime1,Ctime2;
  double Ktime,Ktime1,Ktime2;
  double Kn1time,Kn1time1,Kn1time2;
  double Kn2time,Kn2time1,Kn2time2;
  double Sintime,Sintime1,Sintime2;
  double Cpytime,Cpytime1,Cpytime2;
  double cgtime,cgtime1,cgtime2;
  double ComTime, ComTime1, ComTime2;
  FILE *outf,*outa,*a_matrix_fp;
  bool binaryFlag;
  int a_matrix_f;
  int minimizer, checktype;
  double vL,vR;
  int molidL,molidR, Lgroupbit, Rgroupbit;
  int maxiter;
  double tolarence;
  char prefix_str[FIX_GA_CONP_WIDTH], a_matrix_name[FIX_GA_CONP_WIDTH];

  double unitk[3];
  double *ug;
  double g_ewald,eta,gsqmx,volume,slab_volfactor;
  int *kxvecs,*kyvecs,*kzvecs;
  double ***cs,***sn,**csk,**snk;
  int kmax,kmax3d,kmax_created,kcount;
  int kxmax,kymax,kzmax;
  double *sfacrl,*sfacrl_all,*sfacim,*sfacim_all;
  int everynum;
  int elenum,elenum_old,elenum_all;
  double *eleallq;
  double *aaa_all,*bbb_all;
  int *tag2eleall,*eleall2tag,*curr_tag2eleall,*ele2tag;
*/
};

}; // namespace LAMMPS_NS

#endif
#endif
