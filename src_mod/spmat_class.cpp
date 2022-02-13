/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   SpMatCLS: a sparse matrix using STL:map to generate pcg_matrix
   in PCG_DC method.
------------------------------------------------------------------------- */

#include "spmat_class.h"
#include <set>

using namespace LAMMPS_NS;

SpMatCLS::SpMatCLS(size_t Natom) : data(Natom) { arr = &data.front(); }
SpMatCLS::SpMatCLS(spMatEle *ele, size_t num_ele, spMatEle *eleG,
                   size_t num_eleG, size_t Natom, size_t L_, size_t R_)
    : data(Natom) {
  L = L_;
  R = R_;
  std::set<int> neighbor_GA_index;
  for (size_t k = 0; k < num_ele; ++k) {
    // data.insert(std::make_pair<std::array<int, 2>, double>((ele->index),
    // ele->value));
    int i = ele->i;
    int j = ele->j;

    bool iflag = (i >= L && i < R);
    bool jflag = (j >= L && j < R);
    if (iflag || jflag) {
      neighbor_GA_index.insert(i);
      neighbor_GA_index.insert(j);
      double value = ele->value;
#if __cplusplus < 201103L // For C++11
      if (iflag)
        data[i].insert(std::pair<int, double>(j, value));
      if (jflag)
        data[j].insert(std::pair<int, double>(i, value));
#else
      // if (iflag)
      data[i].emplace(j, value);
      // if (jflag)
      data[j].emplace(i, value);
#endif
      ele->value = 0;
    }
    // nonzeros.at(index[0])++;
    // nonzeros.at(index[1])++;
    ele++;
  }
  ele -= num_ele;

  const spMatEle *eleG_arr = eleG;
  for (size_t k = 0; k < num_eleG; ++k) {
    // data.insert(std::make_pair<std::array<int, 2>, double>((ele->index),
    // ele->value));
    // printf("k=%ld/%ld = %, value = %g\n", k, num_eleG, eleG-eleG_arr,
    // eleG->value);
    int i = eleG->i;
    int j = eleG->j;

    bool iflag = (i >= L && i < R);
    bool jflag = (j >= L && j < R);
    if (iflag || jflag) {
      neighbor_GA_index.insert(i);
      neighbor_GA_index.insert(j);
      double value = eleG->value;
#if __cplusplus < 201103L // For Not C++11
      if (iflag)
        data[i].insert(std::pair<int, double>(j, value));
      if (jflag)
        data[j].insert(std::pair<int, double>(i, value));
#else
      // if (iflag)
      data[i].emplace(j, value);
      // if (jflag)
      data[j].emplace(i, value);
#endif
      eleG->value = 0;
    }
    // nonzeros.at(index[0])++;
    // nonzeros.at(index[1])++;
    eleG++;
  }
  eleG -= num_eleG;

  for (size_t k = 0; k < num_ele; ++k, ele++) {
    if (ele->value == 0) {
      continue;
    }
    int i = ele->i;
    int j = ele->j;
    if (neighbor_GA_index.count(i) != 0 && neighbor_GA_index.count(j) != 0) {
      // printf("fill in %d %d\n", i, j);
      double value = ele->value;
#if __cplusplus < 201103L // For C++11
      data[i].insert(std::pair<int, double>(j, value));
      data[j].insert(std::pair<int, double>(i, value));
#else
      data[i].emplace(j, value);
      data[j].emplace(i, value);
#endif
    }
  }

  for (size_t k = 0; k < num_eleG; ++k, eleG++) {
    if (eleG->value == 0) {
      continue;
    }
    int i = eleG->i;
    int j = eleG->j;
    if (neighbor_GA_index.count(i) != 0 && neighbor_GA_index.count(j) != 0) {
      // printf("fill in %d %d\n", i, j);
      double value = eleG->value;
#if __cplusplus < 201103L // For C++11
      data[i].insert(std::pair<int, double>(j, value));
      data[j].insert(std::pair<int, double>(i, value));
#else
      data[i].emplace(j, value);
      data[j].emplace(i, value);
#endif
    }
  }
  arr = &data.front();
} // SpMatCLS(spMatEle

spMap_type *SpMatCLS::atom(size_t i) { return arr + i; }

size_t SpMatCLS::size() {
  size_t total_size = 0, N = data.size();
  for (size_t i = 0; i < N; ++i) {
    total_size += arr[i].size();
  }
  return total_size;
}

double SpMatCLS::at(size_t i, size_t j) {
  // arr = &data.front();
  // printf("i = %d; j = %d\n", i, j);
  // printf("i = %d; j = %d\n", i, j);
  /*
  bool iflag = (i >= L && i < R);
  bool jflag = (j >= L && j < R);
  if (!iflag && !jflag) printf("wrong structure?");
  if ((i >= L && i < R)) {
    ;
  } else {
    std::swap(i, j);
  }
  */
  spMap_type::iterator it = arr[i].find(j);
  if (it == arr[i].end()) {
    return 0.0;
  } else {
    return it->second;
  }
  return it->second;
};
