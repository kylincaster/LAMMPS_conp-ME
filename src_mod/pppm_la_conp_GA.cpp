/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   pppm_la_conp/GA: PPPM class to perform pppm calculation for Conp/GA
   with the intermediate matrix implementation
------------------------------------------------------------------------- */

#include "pppm_la_conp_GA.h"
#include "debug_func.h"
#include "electro_control_GA.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "fix_conp_GA.h"
#include "force.h"
#include "gridcomm.h"
#include "math_const.h"
#include "memory.h"
#include "remap_wrap.h"
#include <cstring>

#define OFFSET 16384
#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF 1.0f
// #define MPI_FFT_SCALAR MPI_SINGLE
#else
#define ZEROF 0.0
#define ONEF 1.0
// #define MPI_FFT_SCALAR MPI_DOUBLE
#endif

using namespace LAMMPS_NS;
using namespace MathConst;

enum { REVERSE_RHO };
enum { FORWARD_IK, FORWARD_AD, FORWARD_IK_PERATOM, FORWARD_AD_PERATOM };

PPPM_LA_conp_GA::PPPM_LA_conp_GA(LAMMPS* lmp) : PPPM_conp_GA(lmp), gw(NULL)
{
  /*
  nxhi_conp = nxlo_conp = 0;
  nyhi_conp = nylo_conp = 0;
  nzhi_conp = nzlo_conp = 0;
  nx_conp = ny_conp = nz_conp = nxyz_conp = 0;
  */
  matrix_flag  = true;
  matrix_saved = false;
}

PPPM_LA_conp_GA::~PPPM_LA_conp_GA()
{
  ; // printf("~PPPM_LA_conp_GA()");
}

double** PPPM_LA_conp_GA::compute_AAA(double** matrix)
{
  // if(me == 0) printf("compute_AAA()\n");
  double t_begin = MPI_Wtime();
  FFT_SCALAR* gw_store;
  /*
  if(matrix_saved == false) {
    gw_used = compute_intermediate_matrix(true);
  } else {
    gw_used = gw;
  }
  */
  if (me == 0)
    utils::logmesg(lmp, "[ConpGA] compute Ewald matrix from the Green's function "
                        "via {pppm_la_conp/GA}\n");
  MPI_Barrier(world);
  double fft_time  = MPI_Wtime();
  size_t nxyz_pppm = nz_pppm * ny_pppm * nx_pppm;
  int k_disp, j_disp;
  FFT_SCALAR*** greens_real;
  memory->create(greens_real, nz_pppm, ny_pppm, nx_pppm, "pppm:greens_real");
  FFT_SCALAR* const greens_real_arr = &greens_real[0][0][0];
  memset(greens_real_arr, 0, nxyz_pppm * sizeof(FFT_SCALAR));
  memory->create(gw_store, nxyz_pppm, "pppm:gw_store");
  memset(gw_store, 0, 1 * nxyz_pppm * sizeof(FFT_SCALAR));
  // printf("0x\n");

  // fft green's funciton k -> r
  for (int i = 0, n = 0; i < nfft; i++) {
    work2[n++] = greensfn[i];
    work2[n++] = ZEROF; // *= greensfn[i];
  }
  fft2->compute(work2, work2, -1);
  // printf("1x\n");

  const double qscale            = qqrd2e * scale;
  const FFT_SCALAR greens_factor = 0.5 / nxyz_pppm;
  for (int k = nzlo_in, n = 0; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++) {
        greens_real_arr[ny_pppm * nx_pppm * k + nx_pppm * j + i] = greens_factor * work2[n];
        n += 2;
      }

  MPI_Allreduce(MPI_IN_PLACE, greens_real_arr, nxyz_pppm, MPI_FFT_SCALAR, MPI_SUM, world);
  // printf("2x\n");

  int nx_out                 = nxhi_out - nxlo_out + 1; // nx_pppm + order + 1;
  int ny_out                 = nyhi_out - nylo_out + 1; // ny_pppm + order + 1;
  int nz_out                 = nzhi_out - nzlo_out + 1; // nz_pppm + order + 1;
  int nxyz_out               = nx_out * ny_out * nz_out;
  int nxy_pppm               = nx_pppm * ny_pppm;
  FixConpGA* FIX             = electro->FIX;
  int localGA_num            = FIX->localGA_num;
  int totalGA_num            = FIX->totalGA_num;
  int* localGA_EachProc      = &FIX->localGA_EachProc.front();
  int* localGA_EachProc_Dipl = &FIX->localGA_EachProc_Dipl.front();
  int order_Cb               = order * order * order;

  int* grid_pos;
  FFT_SCALAR* rho_GA;
  memory->create(grid_pos, 3 * order_Cb, "pppm:grid_pos");
  memory->create(rho_GA, order_Cb, "pppm:rho_GA");
  // printf("3x\n");

  /*
  int* iptrx = grid_pos;
  for (int ni = nlower; ni <= nupper; ni++) {
    for (int mi = nlower; mi <= nupper; mi++) {
      for (int li = nlower; li <= nupper; li++) {
        iptrx[0] = ni; iptrx[1] = mi; iptrx[2] = li;
        iptrx += 3;
      }
    }
  }
  */
  // memset(gw[0],          0, totalGA_num*nxyz_out*sizeof(FFT_SCALAR));
  // memset(gw_global[0],   0, totalGA_num*nxyz_pppm*sizeof(FFT_SCALAR));

  /*
  printf("nx: %d, %d, nx_pppm: %d\n", nxlo_out, nxhi_out, nx_pppm);
  printf("ny: %d, %d, ny_pppm: %d\n", nylo_out, nyhi_out, ny_pppm);
  printf("nz: %d, %d, nz_pppm: %d\n", nzlo_out, nzhi_out, nz_pppm);
  */

  int* GA2grid_all;
  FFT_SCALAR** GA2rho_all;
  memory->create(GA2grid_all, totalGA_num * 3, "pppm:GA2rho_all");
// TODO replace 6 by 3;
#ifdef INTEL_MPI
  memory->create2d_offset(GA2rho_all, totalGA_num * 6, nlower, nupper, "pppm:GA2rho_all");
#else
  memory->create2d_offset(GA2rho_all, totalGA_num * 3, nlower, nupper, "pppm:GA2rho_all");
#endif

  MPI_Datatype GA2grid_Type;
  MPI_Type_contiguous(3, MPI_INT, &GA2grid_Type);
  MPI_Type_commit(&GA2grid_Type);
  MPI_Allgatherv(&GA2grid[0], localGA_EachProc[me], GA2grid_Type, &GA2grid_all[0], localGA_EachProc, localGA_EachProc_Dipl, GA2grid_Type, world);
  MPI_Type_free(&GA2grid_Type);

  MPI_Datatype GA2rho_Type;
  MPI_Type_contiguous(3 * order, MPI_DOUBLE, &GA2rho_Type);
  MPI_Type_commit(&GA2rho_Type);
  MPI_Allgatherv(&GA2rho[0][nlower], localGA_EachProc[me], GA2rho_Type, &GA2rho_all[0][nlower], localGA_EachProc, localGA_EachProc_Dipl, GA2rho_Type, world);
  MPI_Type_free(&GA2rho_Type);

  // printf("4x\n");

  char* occupied = NULL;
  memory->create(occupied, nxyz_pppm, "pppm:occupied");
  memset(occupied, 0, nxyz_pppm * sizeof(bool));
  // printf("4.1x\n");

  for (int iGA = 0; iGA < localGA_num; ++iGA) {
    int iGA3   = iGA * 3;
    int iGA_nx = GA2grid[iGA3];
    int iGA_ny = GA2grid[iGA3 + 1];
    int iGA_nz = GA2grid[iGA3 + 2]; // nix s
    for (int ni = nlower; ni <= nupper; ni++) {
      int miz = (nz_pppm + ni + iGA_nz) % nz_pppm;
      for (int mi = nlower; mi <= nupper; mi++) {
        int miy = (ny_pppm + mi + iGA_ny) % ny_pppm;
        for (int li = nlower; li <= nupper; li++) {
          int mix                                        = (nx_pppm + li + iGA_nx) % nx_pppm;
          occupied[miz * nxy_pppm + miy * nx_pppm + mix] = true;
        } // li
      }   // mi
    }     // ni
  }       // iGA

  MPI_Allreduce(MPI_IN_PLACE, occupied, nxyz_pppm, MPI_C_BOOL, MPI_LOR, world);
  int occupied_dots = 0;
  for (size_t i = 0; i < nxyz_pppm; ++i) {
    if (occupied[i]) occupied_dots++;
  }
  if (me == 0) {
    utils::logmesg(lmp, "[ConpGA] for the mesh points covered by GA atoms is {:.2f}%% for {{{}/{}}}\n", 100 * double(occupied_dots) / nxyz_pppm, occupied_dots,
                   nxyz_pppm);
  }
  const double matrix_factor = qscale * nxyz_pppm / volume;

  for (int iGA = 0; iGA < localGA_num; iGA++) {
    int iGA3          = iGA * 3;
    int iGA_nx        = GA2grid[iGA3];
    int iGA_ny        = GA2grid[iGA3 + 1];
    int iGA_nz        = GA2grid[iGA3 + 2];
    FFT_SCALAR* ptr_x = GA2rho[iGA3 + 0];
    FFT_SCALAR* ptr_y = GA2rho[iGA3 + 1];
    FFT_SCALAR* ptr_z = GA2rho[iGA3 + 2];

    int* grid_pos_ptr = grid_pos;
    ;
    for (int ni = nlower, ii = 0; ni <= nupper; ni++) {
      FFT_SCALAR iz0 = ptr_z[ni];
      int miz        = ni + iGA_nz;
      for (int mi = nlower; mi <= nupper; mi++) {
        FFT_SCALAR iy0 = iz0 * ptr_y[mi];
        int miy        = mi + iGA_ny;
        for (int li = nlower; li <= nupper; li++) {
          int mix         = li + iGA_nx;
          FFT_SCALAR ix0  = iy0 * ptr_x[li];
          rho_GA[ii]      = ix0;
          grid_pos_ptr[0] = miz;
          grid_pos_ptr[1] = miy;
          grid_pos_ptr[2] = mix;
          grid_pos_ptr += 3;
          ii++;
        }
      }
    }

    FFT_SCALAR* gw_ptr = gw_store;
    char* oPtr         = occupied;
    for (int mjz = 0; mjz < nz_pppm; mjz++) {
      for (int mjy = 0; mjy < ny_pppm; mjy++) {
        for (int mjx = 0; mjx < nx_pppm; mjx++) {
          if (!oPtr[0]) {
            gw_ptr[0] = 0.0;
            gw_ptr++;
            oPtr++;
            continue;
          }

          FFT_SCALAR tmp = 0.0;
          for (int ni = nlower, ii = 0; ni <= nupper; ni++) {
            FFT_SCALAR iz0 = ptr_z[ni];
            int miz        = ni + iGA_nz;
            for (int mi = nlower; mi <= nupper; mi++) {
              FFT_SCALAR iy0 = iz0 * ptr_y[mi];
              int miy        = mi + iGA_ny;
              for (int li = nlower; li <= nupper; li++) {
                int mix        = li + iGA_nx;
                FFT_SCALAR ix0 = iy0 * ptr_x[li];
                int mz         = abs(mjz - miz) % nz_pppm;
                int my         = abs(mjy - miy) % ny_pppm;
                int mx         = abs(mjx - mix) % nx_pppm;
                int ind        = mz * nxy_pppm + my * nx_pppm + mx;
                tmp += ix0 * greens_real_arr[ind];
              }
            }
          } // ni

          /*
          double tmp = 0.0; int *grid_pos_ptr = grid_pos;
          for (int ii = 0; ii < order_Cb; ++ii) {
            int mz = abs(mjz - grid_pos_ptr[0]) % nz_pppm;
            int my = abs(mjy - grid_pos_ptr[1]) % ny_pppm;
            int mx = abs(mjx - grid_pos_ptr[2]) % nx_pppm;
            int ind = mz * nxy_pppm + my * nx_pppm + mx;
            tmp += rho_GA[ii] * greens_real_arr[ind];
            grid_pos_ptr += 3;
          }
          */
          gw_ptr[0] = tmp;
          gw_ptr++;
          oPtr++;
        }
      }
    } // mjz

    /* Print Progress */
    if (me == 0 && iGA % 50 == 0) {
      double dtime = MPI_Wtime() - t_begin;
      utils::logmesg(lmp, "[ConpGA] progress = {:2.2f}%% for {{{}/{}}}, time elapsed: %8.2f[s]\n", (double)(iGA + 1) / localGA_num * 100, iGA + 1, localGA_num,
                     dtime);
    }

    for (int jGA = 0; jGA < totalGA_num; jGA++) {
      int jGA3       = jGA * 3;
      int jGA_nx     = GA2grid_all[jGA3];
      int jGA_ny     = GA2grid_all[jGA3 + 1];
      int jGA_nz     = GA2grid_all[jGA3 + 2]; // nix s
      FFT_SCALAR aij = 0.;

      for (int ni = nlower; ni <= nupper; ni++) {
        FFT_SCALAR iz0 = GA2rho_all[jGA3 + 2][ni];
        int mz         = (nz_pppm + ni + jGA_nz) % nz_pppm;
        for (int mi = nlower; mi <= nupper; mi++) {
          FFT_SCALAR iy0 = iz0 * GA2rho_all[jGA3 + 1][mi];
          int my         = (ny_pppm + mi + jGA_ny) % ny_pppm;
          for (int li = nlower; li <= nupper; li++) {
            int mx         = (nx_pppm + li + jGA_nx) % nx_pppm;
            FFT_SCALAR ix0 = iy0 * GA2rho_all[jGA3][li];
            int ind        = mz * nxy_pppm + my * nx_pppm + mx;
            aij += ix0 * gw_store[ind];
          }
        }
      } // ni
      matrix[iGA][jGA] += aij * matrix_factor;
    } // jGA
  }   // iGA;
  memory->destroy(occupied);
  const double diag_factor = qscale * g_ewald / MY_PIS;
  const double bg_factor   = qscale * MY_PI2 / (g_ewald * g_ewald * volume);
  const double efact       = qscale * MY_2PI / volume;
  const double zprd        = domain->prd[2];
  const double efact_half  = 0.5 * efact;
  const double zprd_const  = efact * zprd * zprd / 12.0 + bg_factor; // qsum_group;

  double *z_pos_local = NULL, *z_pos = NULL;
  double** x  = atom->x;
  int my_disp = localGA_EachProc_Dipl[me];
  if (slabflag == 1) {
    int* GA_seq_arr = &FIX->localGA_seq.front();
    memory->create(z_pos, totalGA_num, "pppm_la_conp/GA:z_pos");
    memory->create(z_pos_local, localGA_num, "pppm_la_conp/GA:localGA_num");
    for (int iGA = 0; iGA < localGA_num; ++iGA) {
      int iseq         = GA_seq_arr[iGA];
      double iz        = x[iseq][2];
      z_pos_local[iGA] = iz;
    }
    MPI_Allgatherv(z_pos_local, localGA_EachProc[me], MPI_DOUBLE, z_pos, localGA_EachProc, localGA_EachProc_Dipl, MPI_DOUBLE, world);

    for (int iGA = 0; iGA < localGA_num; ++iGA) {
      // int iseq = GA_seq_arr[iGA];
      if (me == 0 && iGA % 50 == 0) {
        double dtime = MPI_Wtime() - t_begin;
        utils::logmesg(lmp,
                       "[ConpGA] Post-treatment of E by progress = {:2.2f%}%% for "
                       "{{{}/{}}}, time elapsed: {:8.2f}[s]\n",
                       (double)(iGA + 1) / localGA_num * 100, iGA + 1, localGA_num, dtime);
      }
      double iz     = z_pos_local[iGA];
      double iz_sqr = iz * iz;
      matrix[iGA][iGA + my_disp] -= diag_factor;
      for (int jGA = 0; jGA < totalGA_num; ++jGA) {
        double dz = z_pos[jGA] - iz;
        matrix[iGA][jGA] -= efact_half * dz * dz + zprd_const;
        // matrix[iGA][jGA] += efact*(iz*jz - 0.5 * (jz*jz + iz_sqr) -
        // zprd_const) - bg_factor;
      }
    }
  } else {
    for (int iGA = 0; iGA < localGA_num; ++iGA) {
      if (me == 0 && iGA % 50 == 0) {
        double dtime = MPI_Wtime() - t_begin;
        utils::logmesg(lmp,
                       "[ConpGA] Post-treatment of E by progress = {:2.2f}%% for "
                       "{{{}/{}}}, time elapsed: {:8.2f}[s]\n",
                       (double)(iGA + 1) / localGA_num * 100, iGA + 1, localGA_num, dtime);
      }
      matrix[iGA][iGA + my_disp] -= diag_factor;
      for (int jGA = 0; jGA < totalGA_num; ++jGA) {
        matrix[iGA][jGA] -= bg_factor;
      }
    }
  }

  memory->destroy(gw_store);
  memory->destroy(GA2grid_all);
  memory->destroy2d_offset(GA2rho_all, nlower);

  if (slabflag == 1) {
    memory->destroy(z_pos);
    memory->destroy(z_pos_local);
  }
  double dt = MPI_Wtime() - t_begin;
  if (me == 0)
    utils::logmesg(lmp,
                   "[ConpGA] Elps. Time {:8.6f}[s] of another matrix multiplication for "
                   "Ewald matrix calculation.\n",
                   dt);

  return matrix;
}

void PPPM_LA_conp_GA::settings(int narg, char** arg)
{
  if (narg < 1) error->all(FLERR, "Illegal kspace_style pppm command");
  accuracy_relative = fabs(utils::numeric(FLERR, arg[0], false, lmp));
  int iarg          = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "sv_SOL") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pppm_conp/GA command for sv_SOL option.");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        sv_SOL_Flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        sv_SOL_Flag = false;
      } else
        error->all(FLERR, "Illegal pppm_conp/GA command for sv_SOL option, the "
                          "available options are on or off");
      iarg += 2;
    } else if (strcmp(arg[iarg], "sv_Xuv") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pppm_conp/GA command for sv_Xuv option.");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        matrix_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        matrix_flag = false;
      } else
        error->all(FLERR, "Illegal pppm_conp/GA command for sv_Xuv option, the "
                          "available options are on or off");
      iarg += 2;
    } else if (strcmp(arg[iarg], "fixed") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pppm_conp/GA command for sv_SOL option.");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        fixed_flag = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        fixed_flag = false;
      } else
        error->all(FLERR, "Illegal pppm_conp/GA command for sv_SOL option, the "
                          "available options are on or off");
      iarg += 2;
    } /*
      else if (strcmp(arg[iarg], "sv_GA") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal pppm_conp/GA command for
    sv_GA option."); if (strcmp(arg[iarg+1],"on") == 0) { sv_GA_Flag = true;s }
    else if (strcmp(arg[iarg+1],"off") == 0) { sv_GA_Flag = false; } else
    error->all(FLERR,"Illegal pppm_conp/GA command for sv_GA option, the
    available options are on or off"); iarg += 2;
    } */
    else {
      error->all(FLERR, "Illegal pppm_conp/GA_la command for symbol %.200s\n", arg[iarg]);
    }
  } // while iarg
}
void PPPM_LA_conp_GA::calc_GA_POT(double qsum_group, bool GG_Flag)
{

  if (matrix_saved) {
    ; // if (me == 0) printf("calc_GA_POT() via the intermediate matrix...\n");
  } else {
    // if (me == 0) printf("calc_GA_POT() via the default FFT...\n");
    PPPM_conp_GA::calc_GA_POT(qsum_group, GG_Flag);
    return;
  }

  // cg->reverse_comm(this, REVERSE_RHO);

  FixConpGA* FIX = electro->FIX;
  int nx_conp    = nxhi_out - nxlo_out + 1; // nx_pppm + order + 1;
  int ny_conp    = nyhi_out - nylo_out + 1; // ny_pppm + order + 1;
  int nz_conp    = nzhi_out - nzlo_out + 1; // nz_pppm + order + 1;
  int nxyz       = nx_conp * ny_conp * nz_conp;

  int* GA_seq_arr = &FIX->localGA_seq.front();
  int localGA_num = FIX->localGA_num;
  int totalGA_num = FIX->totalGA_num;

  double* q  = atom->q;
  double** x = atom->x;

  const double qscale     = qqrd2e * scale;
  const double a1         = g_ewald / MY_PIS;
  const double a2         = MY_PI2 * qsum_group / (g_ewald * g_ewald * volume);
  const double efact      = qscale * MY_2PI / volume;
  const double zprd       = domain->prd[2];
  const double zprd_const = qsum_group * zprd * zprd / 12.0;

  FFT_SCALAR* const density_arr = &(density_brick[nzlo_out][nylo_out][nxlo_out]);
  // MPI_Allreduce(MPI_IN_PLACE, density_arr, nxyz, MPI_FFT_SCALAR, MPI_SUM,
  // world);
  for (int iGA = 0; iGA < totalGA_num; ++iGA) {
    FFT_SCALAR etmp = 0;
    for (size_t i = 0; i < nxyz; ++i) {
      etmp += gw[iGA][i] * density_arr[i];
    }
    b_part[iGA] = etmp;
  }
  MPI_Allreduce(MPI_IN_PLACE, b_part, totalGA_num, MPI_FFT_SCALAR, MPI_SUM, world);
  int my_disp = FIX->localGA_EachProc_Dipl[me];
  for (int iGA = 0; iGA < localGA_num; ++iGA) {
    int iseq    = GA_seq_arr[iGA];
    eatom[iseq] = b_part[iGA + my_disp];
    if (GG_Flag) {
      eatom[iseq] -= a1 * q[iseq] + a2;
    } else {
      eatom[iseq] -= a2;
    }
    // g_ewald*q[i]/MY_PIS + MY_PI2*qsum /
    //   (g_ewald*g_ewald*volume);
    eatom[iseq] *= qscale; // q[i] *

    // slab correlation
    if (slabflag == 1) eatom[iseq] += efact * (x[iseq][2] * dipole_group - 0.5 * (dipole_r2_group + qsum_group * x[iseq][2] * x[iseq][2]) - zprd_const);
  }
}

void PPPM_LA_conp_GA::setup_FIX(int groupbit)
{
  PPPM_conp_GA::setup_FIX(groupbit);
  if (matrix_flag) gw = compute_intermediate_matrix(false);
}

/* ----------------------------------------------------
   calculate the intermediate matrix
---------------------------------------------------- */
FFT_SCALAR** PPPM_LA_conp_GA::compute_intermediate_matrix(bool AAA_matrix_flag)
{
  if (me == 0)
    utils::logmesg(lmp, "[ConpGA] compute intermediate matrix to replace FFT via "
                        "{pppm_la_conp/GA}\n");
  MPI_Barrier(world);
  double fft_time  = MPI_Wtime();
  size_t nxyz_pppm = nz_pppm * ny_pppm * nx_pppm;
  int k_disp, j_disp;
  FFT_SCALAR*** greens_real;
  memory->create(greens_real, nz_pppm, ny_pppm, nx_pppm, "pppm:greens_real");
  FFT_SCALAR* const greens_real_arr = &greens_real[0][0][0];
  memset(greens_real_arr, 0, nxyz_pppm * sizeof(FFT_SCALAR));

  // fft green's funciton k -> r
  for (int i = 0, n = 0; i < nfft; i++) {
    work2[n++] = greensfn[i];
    work2[n++] = ZEROF; // *= greensfn[i];
  }
  fft2->compute(work2, work2, -1);

  const double qscale            = qqrd2e * scale;
  const FFT_SCALAR greens_factor = 0.5 / nxyz_pppm;
  for (int k = nzlo_in, n = 0; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++) {
        greens_real_arr[ny_pppm * nx_pppm * k + nx_pppm * j + i] = greens_factor * work2[n];
        n += 2;
      }

  MPI_Allreduce(MPI_IN_PLACE, greens_real_arr, nxyz_pppm, MPI_FFT_SCALAR, MPI_SUM, world);

  int nx_out                 = nxhi_out - nxlo_out + 1; // nx_pppm + order + 1;
  int ny_out                 = nyhi_out - nylo_out + 1; // ny_pppm + order + 1;
  int nz_out                 = nzhi_out - nzlo_out + 1; // nz_pppm + order + 1;
  int nxyz_out               = nx_out * ny_out * nz_out;
  int nxy_pppm               = nx_pppm * ny_pppm;
  FixConpGA* FIX             = electro->FIX;
  int localGA_num            = FIX->localGA_num;
  int totalGA_num            = FIX->totalGA_num;
  int* localGA_EachProc      = &FIX->localGA_EachProc.front();
  int* localGA_EachProc_Dipl = &FIX->localGA_EachProc_Dipl.front();

  /*
  printf("local nxy: %d<->%d[%d:%d], %d<->%d[%d:%d], %d<->%d[%d:%d]\n",
    nxlo_out, nxhi_out, nx_out, nx_pppm,
    nylo_out, nyhi_out, ny_out, ny_pppm,
    nzlo_out, nzhi_out, nz_out, nz_pppm);
  */
  int order_Cb = order * order * order;
  // int *GA2grid_all,
  int* grid_pos;
  FFT_SCALAR **GA2rho_all, **gw_local, **gw_global, **gw_out = NULL;
  FFT_SCALAR* rho_GA;

  memory->create(gw_local, localGA_num, nxyz_pppm, "pppm:gw");
  memory->create(gw_global, totalGA_num, nxyz_pppm, "pppm:gw");
  memory->create(grid_pos, 3 * order_Cb, "pppm:grid_pos");
  memory->create(rho_GA, order_Cb, "pppm:rho_GA");

  int* iptr = grid_pos;
  for (int ni = nlower; ni <= nupper; ni++) {
    for (int mi = nlower; mi <= nupper; mi++) {
      for (int li = nlower; li <= nupper; li++) {
        iptr[0] = ni;
        iptr[1] = mi;
        iptr[2] = li;
        iptr += 3;
      }
    }
  }

  // memset(gw[0],          0, totalGA_num*nxyz_out*sizeof(FFT_SCALAR));
  // memset(gw_global[0],   0, totalGA_num*nxyz_pppm*sizeof(FFT_SCALAR));
  memset(gw_local[0], 0, localGA_num * nxyz_pppm * sizeof(FFT_SCALAR));

  /*
  printf("nx: %d, %d, nx_pppm: %d\n", nxlo_out, nxhi_out, nx_pppm);
  printf("ny: %d, %d, ny_pppm: %d\n", nylo_out, nyhi_out, ny_pppm);
  printf("nz: %d, %d, nz_pppm: %d\n", nzlo_out, nzhi_out, nz_pppm);
  */

  // memory->create(GA2grid_all,  totalGA_num * 3, "pppm:GA2rho_all");
  // memory->create2d_offset(GA2rho_all, totalGA_num * 3, nlower, nupper,
  // "pppm:GA2rho_all");
  /*
  MPI_Datatype GA2grid_Type;
  MPI_Type_contiguous(3, MPI_INT, &GA2grid_Type);
  MPI_Type_commit(&GA2grid_Type);
  MPI_Allgatherv(&GA2grid[0], localGA_EachProc[me], GA2grid_Type,
    &GA2grid_all[0], localGA_EachProc, localGA_EachProc_Dipl, GA2grid_Type,
  world); MPI_Type_free(&GA2grid_Type);

  MPI_Datatype GA2rho_Type;
  MPI_Type_contiguous(3*order, MPI_DOUBLE, &GA2rho_Type);
  MPI_Type_commit(&GA2rho_Type);
  MPI_Allgatherv(&GA2rho[0][nlower], localGA_EachProc[me], GA2rho_Type,
    &GA2rho_all[0][nlower], localGA_EachProc, localGA_EachProc_Dipl,
  GA2rho_Type, world); MPI_Type_free(&GA2rho_Type);
  */

  char* occupied = NULL;

  if (AAA_matrix_flag) {
    memory->create(occupied, nxyz_pppm, "pppm:occupied");
    memset(occupied, 0, nxyz_pppm * sizeof(bool));
    for (int iGA = 0; iGA < localGA_num; ++iGA) {
      int iGA3   = iGA * 3;
      int iGA_nx = GA2grid[iGA3];
      int iGA_ny = GA2grid[iGA3 + 1];
      int iGA_nz = GA2grid[iGA3 + 2]; // nix s
      for (int ni = nlower; ni <= nupper; ni++) {
        int miz = (nz_pppm + ni + iGA_nz) % nz_pppm;
        for (int mi = nlower; mi <= nupper; mi++) {
          int miy = (ny_pppm + mi + iGA_ny) % ny_pppm;
          for (int li = nlower; li <= nupper; li++) {
            int mix                                        = (nx_pppm + li + iGA_nx) % nx_pppm;
            occupied[miz * nxy_pppm + miy * nx_pppm + mix] = true;
          } // li
        }   // mi
      }     // ni
    }       // iGA
    MPI_Allreduce(MPI_IN_PLACE, occupied, nxyz_pppm, MPI_C_BOOL, MPI_LOR, world);
    int occupied_dots = 0;
    for (size_t i = 0; i < nxyz_pppm; ++i) {
      if (occupied[i]) occupied_dots++;
    }
    if (me == 0)
      utils::logmesg(lmp, "[ConpGA] for the mesh points covered by GA atoms is {:.2f}%% for {{{}/{}}}\n", 100 * double(occupied_dots) / nxyz_pppm,
                     occupied_dots, nxyz_pppm);
  }

  // printf("%d %d %d %d %d %d\n", GA2grid_all[0], GA2grid_all[1],
  // GA2grid_all[2],
  //  GA2grid_all[3], GA2grid_all[4], GA2grid_all[5]);
  // print_hash(&GA2rho_all[0][nlower], totalGA_num * 3 * order, "GA2rho_all");
  for (int iGA = 0; iGA < localGA_num; iGA++) {
    if (me == 0 && iGA % 50 == 0) {
      double dtime = MPI_Wtime() - fft_time;
      if (me == 0)
        utils::logmesg(lmp, "progress = {:.2f}%% for {{{}/{}}}, time elapsed: {:8.2f}[s]\n", (double)(iGA + 1) / localGA_num * 100, iGA + 1, localGA_num,
                       dtime);
    }
    int iGA3   = iGA * 3;
    int iGA_nx = GA2grid[iGA3];
    int iGA_ny = GA2grid[iGA3 + 1];
    int iGA_nz = GA2grid[iGA3 + 2]; // nix s
    /*
    int* iptr = grid_pos;
    int* jptr = grid_pos;
    for(int i = 0, int i3 = 0; i < order_Cb; i++) {
      jptr[0] = iptr[0] + iGA_nz;
      jptr[1] = iptr[1] + iGA_ny;
      jptr[2] = iptr[2] + iGA_nx;
      rho[i]  =
      iptr += 3; jptr += 3;
    }
    */

    FFT_SCALAR* ptr_x = GA2rho[iGA3 + 0];
    FFT_SCALAR* ptr_y = GA2rho[iGA3 + 1];
    FFT_SCALAR* ptr_z = GA2rho[iGA3 + 2];
    int* iptr         = grid_pos;
    int ii            = 0;
    for (int ni = nlower; ni <= nupper; ni++) {
      FFT_SCALAR iz0 = ptr_z[ni];
      int miz        = ni + iGA_nz;
      for (int mi = nlower; mi <= nupper; mi++) {
        FFT_SCALAR iy0 = iz0 * ptr_y[mi];
        int miy        = mi + iGA_ny;
        for (int li = nlower; li <= nupper; li++) {
          int mix        = li + iGA_nx;
          FFT_SCALAR ix0 = iy0 * ptr_x[li];
          rho_GA[ii]     = ix0;
          iptr[0]        = miz;
          iptr[1]        = miy;
          iptr[2]        = mix;
          iptr += 3;
          ii++;
        }
      }
    }
    /*
    for (int ni = nlower; ni <= nupper; ni++) {
      FFT_SCALAR iz0 = GA2rho[iGA3+2][ni];
      int miz = ni + iGA_nz;
      for (int mi = nlower; mi <= nupper; mi++) {
        FFT_SCALAR iy0 = iz0 * GA2rho[iGA3+1][mi];
        int miy = mi + iGA_ny;
        for (int li = nlower; li <= nupper; li++) {
          int mix = li + iGA_nx;
          FFT_SCALAR ix0 = iy0 * GA2rho[iGA3][li];
    */
    /*
    for (int ni = nlower; ni <= nupper; ni++) {
      FFT_SCALAR iz0 = GA2rho[iGA3+2][ni];
      int miz = ni + iGA_nz;
      for (int mi = nlower; mi <= nupper; mi++) {
        FFT_SCALAR iy0 = iz0 * GA2rho[iGA3+1][mi];
        int miy = mi + iGA_ny;
        for (int li = nlower; li <= nupper; li++) {
          int mix = li + iGA_nx;
          FFT_SCALAR ix0 = iy0 * GA2rho[iGA3][li];
          */
    /*
    for (int mjz = nzlo_out; mjz <= nzhi_out; mjz++) {
      int mz = abs(mjz - miz) % nz_pppm;
      for (int mjy = nylo_out; mjy <= nyhi_out; mjy++) {
        int my = abs(mjy - miy) % ny_pppm;
        for (int mjx = nxlo_out; mjx <= nxhi_out; mjx++) {
          int mx = abs(mjx - mix) % nx_pppm;
          gw[iGA][nx_out * ny_out * (mjz - nzlo_out) +
                   nx_out * (mjy - nylo_out) + (mjx - nxlo_out)] +=
              ix0 * greens_real_arr[mz * nx_pppm * ny_pppm + my * nx_pppm + mx];
        }
      }
    }
    */

    FFT_SCALAR* gw_ptr = gw_local[iGA];

    if (AAA_matrix_flag) {
      char* oPtr = occupied;
      for (int mjz = 0; mjz < nz_pppm; mjz++) {
        for (int mjy = 0; mjy < ny_pppm; mjy++) {
          for (int mjx = 0; mjx < nx_pppm; mjx++) {
            if (!oPtr[0]) {
              gw_ptr[0] = 0.0;
              gw_ptr++;
              oPtr++;
              continue;
            }
            FFT_SCALAR tmp = 0.0;
            iptr           = grid_pos;
            for (int ii = 0; ii < order_Cb; ++ii) {
              int mz  = abs(mjz - iptr[0]) % nz_pppm;
              int my  = abs(mjy - iptr[1]) % ny_pppm;
              int mx  = abs(mjx - iptr[2]) % nx_pppm;
              int ind = mz * nxy_pppm + my * nx_pppm + mx;
              tmp += rho_GA[ii] * greens_real_arr[ind];
              iptr += 3;
            }
            gw_ptr[0] = tmp;
            gw_ptr++;
            oPtr++;
          }
        }
      } // mjz
    } else {
      for (int mjz = 0; mjz < nz_pppm; mjz++) {
        for (int mjy = 0; mjy < ny_pppm; mjy++) {
          for (int mjx = 0; mjx < nx_pppm; mjx++) {

            FFT_SCALAR tmp = 0.0;
            iptr           = grid_pos;
            for (int ii = 0; ii < order_Cb; ++ii) {
              int mz  = abs(mjz - iptr[0]) % nz_pppm;
              int my  = abs(mjy - iptr[1]) % ny_pppm;
              int mx  = abs(mjx - iptr[2]) % nx_pppm;
              int ind = mz * nxy_pppm + my * nx_pppm + mx;
              tmp += rho_GA[ii] * greens_real_arr[ind];
              iptr += 3;
            }
            gw_ptr[0] = tmp;
            gw_ptr++;
          }
        }
      } // mjz
    }
    /*
    printf("<-%d->\n", iGA);
    print_hash(gw[iGA], nxyz_out, "gw_show");
    folder(gw[iGA], gw_local[iGA]);
    print_hash(gw[iGA], nxyz_out, "gw_show");
    */
  } // iGA

  MPI_Datatype PPPM_Type;
  MPI_Type_contiguous(nxyz_pppm, MPI_FFT_SCALAR, &PPPM_Type);
  MPI_Type_commit(&PPPM_Type);
  MPI_Allgatherv(&gw_local[0][0], localGA_EachProc[me], PPPM_Type, &gw_global[0][0], localGA_EachProc, localGA_EachProc_Dipl, PPPM_Type, world);
  MPI_Type_free(&PPPM_Type);

  if (AAA_matrix_flag) {
    gw_out    = gw_global;
    gw_global = gw_local; // to destroy gw_local;
  } else {
    memory->destroy(gw_local);
    memory->create(gw_local, totalGA_num, nxyz_out, "pppm:gw for nxyz_out");
    gw_out = gw_local;
    for (int iGA = 0; iGA < totalGA_num; ++iGA) {
      folder(gw_out[iGA], gw_global[iGA], occupied);
      // print_hash(gw[iGA], nxyz_out, "gw_show");
    }
  }

  // memory->destroy2d_offset(GA2rho_all, nlower);
  // memory->destroy(GA2grid_all);
  memory->destroy(gw_global);
  MPI_Barrier(world);
  fft_time = MPI_Wtime() - fft_time;
  if (me == 0)
    utils::logmesg(lmp,
                   "[ConpGA] Elps. Time {:8.6f}[s] for the intermediate matrix "
                   "calculation.\n",
                   fft_time);
  if (AAA_matrix_flag == false) {
    matrix_saved = true;
    memory->create(b_part, totalGA_num, "pppm_la:b_part");
  }
  return gw_out;
}

void PPPM_LA_conp_GA::folder(FFT_SCALAR* dst, FFT_SCALAR* src, char* occupied)
{
  int nx_out    = nxhi_out - nxlo_out + 1; // nx_pppm + order + 1;
  int ny_out    = nyhi_out - nylo_out + 1; // ny_pppm + order + 1;
  int nz_out    = nzhi_out - nzlo_out + 1; // nz_pppm + order + 1;
  int nxyz_out  = nx_out * ny_out * nz_out;
  int nxyz_pppm = nx_pppm * ny_pppm * nz_pppm;
  int nxy_pppm  = nx_pppm * ny_pppm;
  for (int mjz = nzlo_out; mjz <= nzhi_out; mjz++) {
    int mz = (mjz + nz_pppm) % nz_pppm;
    for (int mjy = nylo_out; mjy <= nyhi_out; mjy++) {
      int my = (mjy + ny_pppm) % ny_pppm;
      for (int mjx = nxlo_out; mjx <= nxhi_out; mjx++) {
        int mx = (mjx + nx_pppm) % nx_pppm;
        // int ind = nx_out * ny_out * (mjz - nzlo_out) +
        //         nx_out * (mjy - nylo_out) + (mjx - nxlo_out);
        int src_ind = nx_pppm * ny_pppm * (mz - 0) + nx_pppm * (my - 0) + (mx - 0);

        // if (occupied[src_ind]) { dst++; continue; }
        *dst = src[src_ind];
        dst++;
      }
    }
  }
}
