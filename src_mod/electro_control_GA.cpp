/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   Electro_Control_GA: An independent Electro Control Class for CONP fix
------------------------------------------------------------------------- */

#include "electro_control_GA.h"
#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "fix_conp_ME.h"
#include "force.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include <cmath>

// #define CONP_NO_DEBUG
const char name_stat[2][10] = {"Energy", "Potential"};

using namespace LAMMPS_NS;
// Allocate the class Electro_Control_GA
Electro_Control_GA::Electro_Control_GA(LAMMPS* lmp, FixConpGA* fixPtr0)
    : Pointers(lmp), pair(NULL), kspace(NULL), sendproc(NULL), recvproc(NULL), sendnum(NULL), recvnum(NULL), sendlist(NULL), recvlist(NULL), buf_send(NULL),
      buf_recv(NULL), ghost_buf(NULL), DEBUG_LOG_LEVEL(fixPtr0->DEBUG_LOG_LEVEL), Ele_num(fixPtr0->Ele_num), Ele_qsum(fixPtr0->Ele_qsum),
      localGA_eleIndex(fixPtr0->localGA_eleIndex)
{
  kspace_conp_Flag = false;
  DEBUG_GAtoGA     = true;
  DEBUG_SOLtoGA    = true;
  FIX              = fixPtr0;
  FIX_groupbit     = FIX->groupbit;
  me               = comm->me;
  ek_stat          = STAT::E;
  cl_stat          = STAT::E;
  compute_id       = 0;
  pair             = dynamic_cast<Pair_CONP_GA*>(force->pair);
  kspace           = dynamic_cast<KSpaceConpGA*>(force->kspace);
  if (kspace) {
    kspace_conp_Flag = true;
  } else {
    kspace_conp_Flag = false;
  };
  if (pair != NULL) {
    pair->DEBUG_LOG_LEVEL = DEBUG_LOG_LEVEL;
    pair->electro         = this;
    compute_id            = pair->compute_id;
  }
  // char* compute_str[4] = {"__Electro_Control_GA_Compute__", "all", "pe/atom", "pair"};
  // modify->add_compute(4, compute_str, 0);
}
double* Electro_Control_GA::ek_atom() { return force->kspace->eatom; }
double* Electro_Control_GA::cl_atom() { return pair->cl_atom; }

void Electro_Control_GA::compute_GAtoGA()
{
  FUNC_MACRO(0);
  if (kspace_conp_Flag && DEBUG_GAtoGA) {
    kspace->compute_GAtoGA();
    ek_stat = STAT::POT;
  } else {
    force->kspace->qsum_qsq();
    force->kspace->compute(3, 0);
    ek_stat = STAT::E;
  }
}

void Electro_Control_GA::compute_SOLtoGA(bool GG_Flag)
{
  FUNC_MACRO(0);
  if (kspace_conp_Flag && DEBUG_SOLtoGA) {
    kspace->compute_SOLtoGA(GG_Flag);
    ek_stat = STAT::POT;
  } else {
    force->kspace->qsum_qsq();
    force->kspace->compute(3, 0);
    ek_stat = STAT::E;
  }
  /*
  double norm0 = FIX->check_atomvec(eatom, "kspace->compute");
  // force->kspace->compute(4, 0);
  calc_GA_potential(cl_stat, POT, false);
  double norm1 = FIX->check_atomvec(eatom, "kspace->compute");
  FIX->force_clear();
  // FIX->force_clear();
  double norm2 = FIX->check_atomvec(eatom, "kspace->compute_SOLtoGA");
  kspace->compute_SOLtoGA(GG_Flag);
  double norm3 = FIX->check_atomvec(eatom, "kspace->compute_SOLtoGA");
  // if(0 && fabs(norm1-norm) > 1e-11) {
  //	printf("wrong norm\n");
  //	exit(0);
  // }
  ek_stat = POT;
  */
}

void Electro_Control_GA::qsum_qsq()
{
  FUNC_MACRO(1);
  return force->kspace->qsum_qsq();
}

void Electro_Control_GA::calc_qsqsum_SOL()
{
  FUNC_MACRO(0);
  const double* const q = atom->q;
  const int nlocal      = atom->nlocal;
  double qsum_local(0.0), qsqsum_local(0.0);
  int* type     = atom->type;
  int* GA_Flags = FIX->GA_Flags;

  for (int i = 0; i < nlocal; i++) {
    if (GA_Flags[type[i]] != 0) {
      qsum_local += q[i];
      qsqsum_local += q[i] * q[i];
    }
  }

  MPI_Allreduce(&qsum_local, &qsum_SOL, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&qsqsum_local, &qsqsum_SOL, 1, MPI_DOUBLE, MPI_SUM, world);
  if (me == 0) {
    utils::logmesg(lmp, "qsum = {:.10f}; qsqsum = {:.10f}\n", qsum_SOL, qsqsum_SOL);
  }
}

double** Electro_Control_GA::compute_AAA(double** AAA)
{
  if (is_compute_AAA() == false) {
    error->all(FLERR, "The kspace part cannot compute AAA matrix!");
  }
  AAA = FIX->pair_matrix(AAA);
  AAA = kspace->compute_AAA(AAA);
  return AAA;
}

void Electro_Control_GA::compute_addstep(bigint n) { modify->compute[compute_id]->addstep(n); }

void Electro_Control_GA::setup()
{
  FUNC_MACRO(0);

  int me = comm->me;
  if (force->kspace) {

  } else {
    error->all(FLERR, "Cannot find kspace to invoke!");
  }
  pair = dynamic_cast<Pair_CONP_GA*>(force->pair);
  if (pair == NULL) {
    error->all(FLERR, "Only the pair_style is not compatible with fix conp/GA!");
  } else {
    pair->DEBUG_LOG_LEVEL = DEBUG_LOG_LEVEL;
    pair->electro         = this;
    compute_id            = pair->compute_id;
  }

  kspace = dynamic_cast<KSpaceConpGA*>(force->kspace);
  // kspace = NULL;
  if (kspace) {
    kspace_conp_Flag = true;
    if (me == 0) {
      utils::logmesg(lmp, "[ConpGA] using optimized electrostatic solver: {}\n", kspace->type_name());
    }
    kspace->electro         = this;
    kspace->DEBUG_LOG_LEVEL = DEBUG_LOG_LEVEL;
    kspace->setup_FIX(FIX_groupbit);
  } else {
    kspace_conp_Flag = false;
    if (me == 0) {
      utils::logmesg(lmp, "[ConpGA] using LAMMPS's default electrostatic solver\n");
    }
  }
  // eatom = force->kspace->eatom;
  calc_qsqsum();
}

int* Electro_Control_GA::setup_comm(int* all_ghostGA_ind)
{
  if (me == 0) utils::logmesg(lmp, "[ConpGA] Setup inner communicator\n");

  int nprocs = comm->nprocs;
  std::map<int, int> tag_to_proc;
  const int* const localGA_tag = &FIX->localGA_tag.front();
  const int* const totalGA_tag = &FIX->totalGA_tag.front();
  int* const localGA_EachProc  = &FIX->localGA_EachProc.front();
  int localGA_num              = FIX->localGA_num;
  int totalGA_num              = FIX->totalGA_num;
  int all_ghostGA_num          = FIX->all_ghostGA_num;
  // int* all_ghostGA_seq         = FIX->all_ghostGA_seq;
  int* tag   = atom->tag;
  int itotal = 0;
  for (int iproc = 0; iproc < nprocs; iproc++) {
    for (int i = 0; i < localGA_EachProc[iproc]; i++) {
      tag_to_proc[totalGA_tag[itotal]] = iproc;
      itotal++;
    }
  }
  int *recvnum_disp, *recvnum_count;

  memory->create(ghost_buf, all_ghostGA_num, "Electro_Control_GA::setup_comm()::ghost_buf");
  memory->create(sendproc, nprocs, "Electro_Control_GA::setup_comm()::sendpros");
  memory->create(recvnum, nprocs, "Electro_Control_GA::setup_comm()::revcnum");
  memory->create(sendnum, nprocs, "Electro_Control_GA::setup_comm()::sendnum");
  memory->create(recvnum_disp, nprocs + 1, "Electro_Control_GA::setup_comm()::recvnum_disp");
  memory->create(recvnum_count, nprocs, "Electro_Control_GA::setup_comm()::recvnum_count");
  set_zeros(recvnum, 0, nprocs);

  /*
  map<int, int>& tag_to_ghostGA = FIX->tag_to_ghostGA;
  map<int, int>::iterator it;
  for (it = tag_to_ghostGA.begin(); it != tag_to_ghostGA.end(); it++) {
  */
  int* mask  = atom->mask;
  int nlocal = atom->nlocal;
  int ntotal = atom->nghost + nlocal;
  for (int i = 0; i < all_ghostGA_num; ++i) {
    int iseq  = all_ghostGA_ind[i];
    int itag  = tag[iseq];
    int irecv = tag_to_proc[itag];
    recvnum[irecv]++;
  }
  /*
  for (int iGA = localGA_num; iGA < used_ghostGA_num; iGA++) {
          int irecv = tag_to_proc[localGA_tag[iGA]];
          recvnum[irecv] ++;
  }
  */
  recvnum_disp[0] = 0;
  int buf_maxsize = -1;
  for (int iproc = 0; iproc < nprocs; iproc++) {
    MPI_Scatter(recvnum, 1, MPI_INT, &sendnum[iproc], 1, MPI_INT, iproc, world);
    recvnum_disp[iproc + 1] = recvnum_disp[iproc] + recvnum[iproc];
    buf_maxsize             = buf_maxsize > recvnum[iproc] ? buf_maxsize : recvnum[iproc];
    buf_maxsize             = buf_maxsize > sendnum[iproc] ? buf_maxsize : sendnum[iproc];
  }

  // "printf"("[me:%d] recvnum_max = %d\n", me, buf_maxsize);
  memory->create(buf_recv, buf_maxsize, "Electro_Control_GA::setup_comm()::buf_recv");
  memory->create(buf_send, buf_maxsize, "Electro_Control_GA::setup_comm()::buf_send");

  int *recv_store, *recv_store_tag;
  int** recvlist_tag;
  memory->create(recv_store, recvnum_disp[nprocs], "Electro_Control_GA::setup_comm()::recv_store");
  memory->create(recv_store_tag, recvnum_disp[nprocs], "Electro_Control_GA::setup_comm()::recv_store");

  sendlist     = (int**)memory->smalloc(nprocs * sizeof(int*), "Electro_Control_GA::setup_comm()::sendlist");
  recvlist     = (int**)memory->smalloc(nprocs * sizeof(int*), "Electro_Control_GA::setup_comm()::recvlist");
  recvlist_tag = (int**)memory->smalloc(nprocs * sizeof(int*), "Electro_Control_GA::setup_comm()::recvlist");

  for (int iproc = 0; iproc < nprocs; iproc++) {
    sendlist[iproc]     = (int*)memory->smalloc(sendnum[iproc] * sizeof(int), "Electro_Control_GA::setup_comm()::sendlist[iproc]");
    recvlist_tag[iproc] = &recv_store_tag[recvnum_disp[iproc]];
    // printf("[me = %d] for iproc %d at %d[size:%d]/%d\n", me, iproc,
    // 		recvnum_disp[iproc], recvnum[iproc], recvnum_disp[nprocs]);
    recvlist[iproc] = &recv_store[recvnum_disp[iproc]];
    // recvlist = (int*) malloc(recvnum[iproc]);
    recvnum_count[iproc] = 0;
  }

  // for (it = tag_to_ghostGA.begin(); it != tag_to_ghostGA.end(); it++) {
  //  int itag                = it->first;
  for (int i = 0; i < all_ghostGA_num; ++i) {
    int itag                    = tag[all_ghostGA_ind[i]];
    int irecv                   = tag_to_proc[itag];
    int icount                  = recvnum_count[irecv];
    recvlist_tag[irecv][icount] = itag;
    recvlist[irecv][icount]     = all_ghostGA_ind[i];
    all_ghostGA_ind[i]          = &recvlist[irecv][icount] - recv_store;
    recvnum_count[irecv]++;
  }
  for (int iproc = 0; iproc < nprocs; iproc++) {
    MPI_Scatterv(recv_store_tag, recvnum, recvnum_disp, MPI_INT, sendlist[iproc], sendnum[iproc], MPI_INT, iproc, world);
    // printf("%d %d sendnum[iproc] = %d\n", recvnum[iproc], recvnum_disp[iproc], sendnum[iproc]);
  }

  /*
  for (int iproc = 0; iproc < nprocs; iproc++) {
    int* list = recvlist[iproc];
    for (int i = 0; i < recvnum[iproc]; i++) {
      list[i] = recv_list_seq[iproc][i];
      // printf("itag = %d <-> %d\n", itag, list[i]);
    }
  }
  */

  map<int, int> tag_to_localGA;
  MAP_RESERVE(tag_to_localGA, localGA_num);
  for (int i = 0; i < localGA_num; i++) {
    tag_to_localGA[tag[i]] = i;
  }
  for (int iproc = 0; iproc < nprocs; iproc++) {
    int* list = sendlist[iproc];
    for (int i = 0; i < sendnum[iproc]; i++) {
      int itag = list[i];
      list[i]  = tag_to_localGA[itag];
    }
  }
  int iswap = iswap_me = 0;
  for (int iproc = 0; iproc < nprocs; iproc++) {
    if (sendnum[iproc] != 0 || recvnum[iproc] != 0 || iproc == me) {
      sendproc[iswap] = iproc;
      if (iproc == me) {
        iswap_me = iswap;
      }
      if (iswap != iproc) {
        sendnum[iswap]  = sendnum[iproc];
        sendlist[iswap] = sendlist[iproc];
        recvnum[iswap]  = recvnum[iproc];
        recvlist[iswap] = recvlist[iproc];
      }
      iswap++;
    }
  }
  nswap = iswap;
  if (DEBUG_LOG_LEVEL > 0) {
    for (int iproc = 0; iproc < nprocs; iproc++) {
      if (me == iproc) {
        utils::logmesg(lmp, "[Debug] Proc_id: {}\n", iproc);
        utils::logmesg(lmp, "Sendlist: ");
        for (int i = 0; i < nswap; i++) {
          utils::logmesg(lmp, "{}[{}] ", sendnum[i], sendproc[i]);
        }
        utils::logmesg(lmp, "\nRecvlist: ");
        for (int i = 0; i < nswap; i++) {
          utils::logmesg(lmp, "{}[{}] ", recvnum[i], sendproc[i]);
        }
        utils::logmesg(lmp, "Total: {}\n", recvnum_disp[nprocs]);
      }
      MPI_Barrier(world);
      /* code */
    }
  }

  // free memory;
  memory->destroy(recv_store_tag);
  memory->sfree(recvlist_tag);
  return recv_store;
}

void Electro_Control_GA::forward_comm(int pflag)
{
  FUNC_MACRO(0);
  MPI_Request request;
  if (recvnum[iswap_me] > 0) {
    int n = pack_forward_comm(pflag, sendnum[iswap_me], sendlist[iswap_me], buf_send);
    unpack_forward_comm(pflag, recvnum[iswap_me], recvlist[iswap_me], buf_send);
  }
  for (int iswap = 1; iswap < nswap; iswap++) {
    int isend = (iswap + iswap_me) % nswap;         // e.g. start at 3, send   to 4, 5, 6,  7
    int irecv = (nswap + iswap_me - iswap) % nswap; // e.g. start at 3, recv from 2, 1, 0, -1
    int n     = pack_forward_comm(pflag, sendnum[isend], sendlist[isend], buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer
    if (recvnum[irecv]) MPI_Irecv(buf_recv, recvnum[irecv], MPI_DOUBLE, sendproc[irecv], 0, world, &request);
    if (sendnum[isend]) MPI_Send(buf_send, sendnum[isend], MPI_DOUBLE, sendproc[isend], 0, world);
    if (recvnum[irecv]) MPI_Wait(&request, MPI_STATUS_IGNORE);

    unpack_forward_comm(pflag, recvnum[irecv], recvlist[irecv], buf_recv);
  }
}

void Electro_Control_GA::reverse_comm(int pflag)
{
  FUNC_MACRO(0);
  MPI_Request request;
  if (recvnum[iswap_me] > 0) {
    // printf("recvnum[iswap_me]  = %d\n", recvnum[iswap_me]);
    int n = pack_reverse_comm(pflag, recvnum[iswap_me], recvlist[iswap_me], buf_send);
    unpack_reverse_comm(pflag, sendnum[iswap_me], sendlist[iswap_me], buf_send);
  }

  for (int iswap = 1; iswap < nswap; iswap++) {
    int isend = (iswap + iswap_me) % nswap;         // e.g. start at 3, send   to 4, 5, 6,  7
    int irecv = (nswap + iswap_me - iswap) % nswap; // e.g. start at 3, recv from 2, 1, 0, -1
    int n     = pack_reverse_comm(pflag, recvnum[isend], recvlist[isend], buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer
    if (sendnum[irecv]) MPI_Irecv(buf_recv, sendnum[irecv], MPI_DOUBLE, sendproc[irecv], 0, world, &request);
    if (recvnum[isend]) MPI_Send(buf_send, recvnum[isend], MPI_DOUBLE, sendproc[isend], 0, world);
    if (sendnum[irecv]) MPI_Wait(&request, MPI_STATUS_IGNORE);

    unpack_reverse_comm(pflag, sendnum[irecv], sendlist[irecv], buf_recv);
  }
}

int Electro_Control_GA::pack_forward_comm(int pflag, int n, int* list, double* buf)
{
  int m;
  // static int times = 0;
  // printf("[me:%d], pflag = %d, n = %d | %d\n", me, pflag, n, times);
  // charge
  if (pflag == 0) {
    double* q = atom->q;
    for (m = 0; m < n; m++) {
      buf[m] = q[list[m]];
      // printf("m = %d/%d, q[%d]= %f\n", m, n, list[m], q[list[m]]);
    }
    // cl_atom
  } else if (pflag == 1) {
    double* eatom = cl_atom();
    for (m = 0; m < n; m++)
      buf[m] = eatom[list[m]];
  }
  // return m;
}

void Electro_Control_GA::unpack_forward_comm(int pflag, int n, int* list, double* buf)
{
  int m;
  // static int times = 0;
  // printf("unpack [me:%d], pflag = %d, n = %d | %d\n", me, pflag, n, times);
  if (pflag == 0) {
    for (m = 0; m < n; m++) {
      double* q = atom->q;
      // printf("[me = %d] m/n = %d/%d\n", me, m, n);
      // printf("m = %d/%d, buf[%d]= %f\n", m, n, list[m], buf[m]);
      // int iseq = all_ghostGA_seq[list[m]];
      q[list[m]] = buf[m];
    }
  } else if (pflag == 1) {
    double* _cl_atom = cl_atom();
    for (m = 0; m < n; m++) {
      // int iseq       = all_ghostGA_seq[list[m]];
      _cl_atom[list[m]] = buf[m];
    }
  }
  // printf("end of unpack [me:%d], pflag = %d, n = %d | %d\n", me, pflag, n, times);
  // times++;
}

int Electro_Control_GA::pack_reverse_comm(int pflag, int n, int* list, double* buf)
{
  int m;
  // charge
  if (pflag == 2) {
    double* _cl_atom = cl_atom();
    for (m = 0; m < n; m++) {
      buf[m] = _cl_atom[list[m]];

      _cl_atom[list[m]] = 0;
    }
  } else if (pflag == 0) {
    double* q = atom->q;
    for (m = 0; m < n; m++) {
      buf[m] = q[list[m]];
    }
  } else if (pflag == 1) {
    double* _cl_atom = cl_atom();
    for (m = 0; m < n; m++) {
      buf[m] = _cl_atom[list[m]];
    }
  }
  return m;
}

void Electro_Control_GA::unpack_reverse_comm(int pflag, int n, int* list, double* buf)
{
  int m;
  // charge
  if (pflag == 0) {
    double* q = atom->q;
    for (m = 0; m < n; m++) {
      q[list[m]] += buf[m];
      // printf("m = %d/%d, q[%d]= %f\n", m, n, list[m], q[list[m]]);
    }
    // cl_atom
  } else if (pflag == 1 || pflag == 2) {
    double* _cl_atom = cl_atom();
    for (m = 0; m < n; m++)
      _cl_atom[list[m]] += buf[m];
  }
}

/*----------------------------------------------------------------
   The function to calcualte the potential between GA atoms
   cl_stat_new = {E, POT} w
   ek_stat_new = {E, POT}  where E for Energy and POT for Potential
   sum_op      = {0, 1, 2} where 0 for no combine
                                 1 for all combine to cl_atom
                                 2 with Pot combined to cl_atom and E combined to ek_atom
----------------------------------------------------------------- */

void Electro_Control_GA::calc_GA_potential(STAT cl_stat_new, STAT ek_stat_new, SUM_OP sum_op)
{
  // OP: {0, NIL}, {1: PE->Potential}, {2: Potential->PE}
  // printf("ek_stat = %d\n", ek_stat);
  if (me == 0 && DEBUG_LOG_LEVEL > 1)
    utils::logmesg(lmp, "[Debug] U<=>POT conv. {{cl: {}=>{}}} {{ek: {}=>{}}} sum_op {}\n", cl_stat, cl_stat_new, ek_stat, ek_stat_new, sum_op);

  if (sum_op != SUM_OP::BOTH) {
    int cl_op        = (3 + static_cast<int>(cl_stat_new) - static_cast<int>(cl_stat)) % 3;
    int ek_op        = (3 + static_cast<int>(ek_stat_new) - static_cast<int>(ek_stat)) % 3;
    bool sum_op_flag = sum_op == SUM_OP::ON;
    if (sum_op_flag && cl_stat_new != ek_stat_new) {
      if (me == 0)
        utils::logmesg(lmp, "Sum between cl{{{}: {}}} vs ek{{{}: {}}}\n", cl_stat, name_stat[static_cast<int>(cl_stat)], ek_stat,
                       name_stat[static_cast<int>(ek_stat)]);
      error->all(FLERR, "inconsistent sum type!");
    }
    conv_GA_potential(cl_op, ek_op, sum_op_flag);
    ek_stat = ek_stat_new;
    cl_stat = cl_stat_new;
  } else {
    // SUM_OP::BOTH
    conv_GA_potential_both(cl_stat, ek_stat);
    cl_stat = STAT::E;
    ek_stat = STAT::POT;
  }
}

void Electro_Control_GA::pre_reverse()
{
  pair->clear_saved();
  if (kspace) kspace->clear_saved();
}

void Electro_Control_GA::pair_GAtoGA()
{
  FUNC_MACRO(0);
  const bool POT_Flag = true;
  if (FIX->pair_flag == true) {
    FIX->fix_pair_compute(POT_Flag, false);
  } else {
    FIX->clear_cl_atom();
    pair->pair_GAtoGA_energy(FIX_groupbit, POT_Flag);
  }
  FIX->comm_REV_COUL_RM();
  cl_stat = POT_Flag ? STAT::POT : STAT::E;
}

void Electro_Control_GA::pair_SOLtoGA(bool GG_Flag)
{
  FUNC_MACRO(0);
  const bool POT_Flag = true;
  if (GG_Flag == true && FIX->pair_flag == true) {
    // printf("using FIX->fix_pair_compute\n");
    FIX->fix_pair_compute(POT_Flag, false);
    pair->pair_SOLtoGA_energy(FIX_groupbit, false, POT_Flag);
  } else {
    pair->pair_SOLtoGA_energy(FIX_groupbit, GG_Flag, POT_Flag);
  }
  FIX->comm_REV_COUL_RM();
  // printf("End of using FIX->fix_pair_compute\n");
  cl_stat = POT_Flag ? STAT::POT : STAT::E;
}

void Electro_Control_GA::conv_GA_potential_both(STAT cl_stat, STAT ek_stat)
{
  // Save both electrostatic energy and potential to cl_atom and ek_atom, respectively.
  int all_op            = static_cast<int>(cl_stat) + 2 * static_cast<int>(ek_stat);
  int* const GA_seq_arr = &FIX->localGA_seq.front();
  int localGA_num       = FIX->localGA_num;
  int i, iseq;

  double* ek_atom = force->kspace->eatom;
  double* cl_atom = pair->cl_atom;
  double *q       = atom->q, qtmp, u_tmp, pe_tmp;

  if (me == 0 && DEBUG_LOG_LEVEL > 1)
    utils::logmesg(lmp, "[Debug] Both sv_Op{{{}}} from cl_stat = {{{}: {}}} ek_stat = {{{}: {}}} \n", all_op, cl_stat, name_stat[static_cast<int>(cl_stat)],
                   ek_stat, name_stat[static_cast<int>(ek_stat)]);
  switch (all_op) {
    case 0 + 0:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        cl_atom[iseq] += ek_atom[iseq];
        ek_atom[iseq] = cl_atom[iseq] / qtmp;
      };
      break;
    case 1 + 0:
      for (i = 0; i < localGA_num; i++) {
        iseq   = GA_seq_arr[i];
        qtmp   = q[iseq];
        pe_tmp = cl_atom[iseq] * qtmp;
        u_tmp  = ek_atom[iseq] / qtmp;
        cl_atom[iseq] += u_tmp;
        ek_atom[iseq] += pe_tmp;
        qtmp          = cl_atom[iseq];
        cl_atom[iseq] = ek_atom[iseq];
        ek_atom[iseq] = qtmp;
      };
      break;
    case 0 + 2:
      for (i = 0; i < localGA_num; i++) {
        iseq   = GA_seq_arr[i];
        qtmp   = q[iseq];
        u_tmp  = cl_atom[iseq] / qtmp;
        pe_tmp = ek_atom[iseq] * qtmp;
        cl_atom[iseq] += pe_tmp;
        ek_atom[iseq] += u_tmp;
      };
      break;
    case 1 + 2:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] += cl_atom[iseq];
        cl_atom[iseq] = ek_atom[iseq] * qtmp;
      };
      break;
    default:
      if (me == 0)
        utils::logmesg(lmp, "[Debug] invalid sv_Op{{{}}} from cl_stat = {{{}: {}}} ek_stat = {{{}: {}}} \n", all_op, cl_stat,
                       name_stat[static_cast<int>(cl_stat)], ek_stat, name_stat[static_cast<int>(ek_stat)]);
      error->all(FLERR, "unkown type!");
      break;
  }
}

void Electro_Control_GA::update_qstat()
{
  //# update the charge state for the small charges.
  /*
  myKSpace->compute(3001,0);
  neighbor_ago_store = neighbor->ago;
neighbor->ago = 0;
myKSpace->compute(3001,0);
neighbor->ago = neighbor_ago_store;
  */
}

void Electro_Control_GA::conv_GA_potential(int cl_op, int ek_op, bool sum_op)
{

  int* const GA_seq_arr = &FIX->localGA_seq.front();
  int localGA_num       = FIX->localGA_num;
  int i, iseq;

  double* ek_atom = force->kspace->eatom;
  double* cl_atom = pair->cl_atom;
  double *q       = atom->q, qtmp;
  int all_op      = cl_op + (3 * ek_op) + (9 * sum_op);
  if (me == 0 && DEBUG_LOG_LEVEL > 1) utils::logmesg(lmp, "[Debug] all_op[{}] with cl_op = {} ek_op = {} and sum_op {}\n", all_op, cl_op, ek_op, sum_op);
  switch (all_op) {
    case 0 + 0:
      break;
    case 1 + 0:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        cl_atom[iseq] /= q[iseq];
      };
      break;
    case 2 + 0:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        cl_atom[iseq] *= q[iseq];
      };
      break;
    case 0 + 3:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        ek_atom[iseq] /= q[iseq];
      };
      break;
    case 1 + 3:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] /= qtmp;
        cl_atom[iseq] /= qtmp;
      };
      break;
    case 2 + 3:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] *= qtmp;
        cl_atom[iseq] /= qtmp;
      };
      break;
    case 0 + 6:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        ek_atom[iseq] *= q[iseq];
      };
      break;
    case 1 + 6:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] /= qtmp;
        cl_atom[iseq] *= qtmp;
      };
      break;
    case 2 + 6:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] *= qtmp;
        cl_atom[iseq] *= qtmp;
      };
      break;
    case 0 + 0 + 9:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] += cl_atom[iseq];
        cl_atom[iseq] = 0;
      };
      break;
    case 1 + 9:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] += cl_atom[iseq] / qtmp;
        cl_atom[iseq] = 0;
      };
      break;
    case 2 + 9:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] += cl_atom[iseq] * qtmp;
        cl_atom[iseq] = 0;
      };
      break;
    case 0 + 3 + 9:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] /= qtmp;
        ek_atom[iseq] += cl_atom[iseq];
        cl_atom[iseq] = 0;
      };
      break;
    case 1 + 3 + 9:
      for (i = 0; i < localGA_num; i++) {
        iseq          = GA_seq_arr[i];
        qtmp          = q[iseq];
        ek_atom[iseq] = (cl_atom[iseq] + ek_atom[iseq]) / qtmp;
        cl_atom[iseq] = 0;
      };
      break;
    case 0 + 6 + 9:
      for (i = 0; i < localGA_num; i++) {
        iseq = GA_seq_arr[i];
        qtmp = q[iseq];
        ek_atom[iseq] *= qtmp;
        ek_atom[iseq] += cl_atom[iseq];
        cl_atom[iseq] = 0;
      };
      break;
    case 2 + 6 + 9:
      for (i = 0; i < localGA_num; i++) {
        iseq          = GA_seq_arr[i];
        qtmp          = q[iseq];
        ek_atom[iseq] = (cl_atom[iseq] + ek_atom[iseq]) * qtmp;
        cl_atom[iseq] = 0;
      };
      break;
    default:
      if (me == 0) utils::logmesg(lmp, "invalid all_op[{}] with cl_op = {} ek_op = {} and sum_op {}\n", all_op, cl_op, ek_op, sum_op);
      error->all(FLERR, "unkown type!");
      break;
  }
} // conv_GA_potential

void Electro_Control_GA::compute(int eflag, int vflag)
{
  FUNC_MACRO(1);
  if (kspace_conp_Flag) {
    return force->kspace->compute(eflag, vflag); // TODO  implement kspace->compute
    ek_stat = STAT::E;
  } else {
    return force->kspace->compute(eflag, vflag);
    ek_stat = STAT::E;
  }
}

void Electro_Control_GA::set_qsqsum(double qsum0, double qsqsum0)
{
  FUNC_MACRO(1);
  if (kspace_conp_Flag) {
    return kspace->set_qsqsum(qsum0, qsqsum0);
  } else {
    force->kspace->qsum_qsq();
#ifndef CONP_NO_DEBUG
    double qsum   = calc_qsum();
    double qsqsum = calc_qsqsum();
    if (!isclose(qsum0, qsum, 1e-10) && fabs(qsum0) > 1e-6 || !isclose(qsqsum0, qsqsum, 1e-10)) {
      if (me == 0) {
        utils::logmesg(lmp, "qsum   = {:.12g} vs {:.12g} \n", qsum, qsum0);
        utils::logmesg(lmp, "qsqsum = {:.12g} vs {:.12g} \n", qsqsum, qsqsum0);
      }
      error->all(FLERR, "Invalid qsum or qsqsum!");
    }
#endif
  }
}

double Electro_Control_GA::calc_qsum()
{
  const double* const q = atom->q;
  const int nlocal      = atom->nlocal;
  double qsum_local(0.0), qsqsum_local(0.0), qsum;
#if defined(_OPENMP)
#pragma omp parallel for default(none) reduction(+ : qsum_local, qsqsum_local)
#endif
  for (int i = 0; i < nlocal; i++) {
    qsum_local += q[i];
    // qsqsum_local += q[i]*q[i];
  }
  MPI_Allreduce(&qsum_local, &qsum, 1, MPI_DOUBLE, MPI_SUM, world);
  // MPI_Allreduce(&qsqsum_local,&qsqsum,1,MPI_DOUBLE,MPI_SUM,world);
  return qsum;
}

double Electro_Control_GA::calc_qsqsum()
{
  FUNC_MACRO(0);
  const double* const q = atom->q;
  const int nlocal      = atom->nlocal;
  double qsum_local(0.0), qsqsum_local(0.0), qsqsum;

#if defined(_OPENMP)
#pragma omp parallel for default(none) reduction(+ : qsum_local, qsqsum_local)
#endif
  for (int i = 0; i < nlocal; i++) {
    qsqsum_local += q[i] * q[i];
  }
  MPI_Allreduce(&qsqsum_local, &qsqsum, 1, MPI_DOUBLE, MPI_SUM, world);
  return qsqsum;
}

// Deallocate the class Electro_Control_GA
Electro_Control_GA::~Electro_Control_GA()
{
  if (FIX->firstgroup_flag) {
    memory->destroy(sendproc);
    memory->destroy(recvproc);
    memory->destroy(sendnum);
    memory->destroy(recvnum);
    memory->destroy(recvlist[0]);
    memory->destroy(buf_send);
    memory->destroy(buf_recv);
    memory->destroy(ghost_buf);

    for (int i = 0; i < nswap; i++) {
      memory->sfree(sendlist[i]);
    }
    memory->sfree(sendlist);
    memory->sfree(recvlist);
  }
  if (pair != NULL) {
    pair->DEBUG_LOG_LEVEL = 0;
    pair->electro         = nullptr;
  }
  if (kspace_conp_Flag == true && kspace != nullptr) {
    kspace->DEBUG_LOG_LEVEL = 0;
    kspace->electro         = nullptr;
  }
}
