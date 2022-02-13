/* -*- c++ -*- -------------------------------------------------------------
   Constant Potential Simulatiom for Graphene Electrode (Conp/GA)

   Version 1.60.0 06/10/2021
   by Gengping JIANG @ WUST

   SpMatCLS: a sparse matrix using STL:map to generate pcg_matrix
   in PCG_DC method.
------------------------------------------------------------------------- */

#ifndef CONP_GA_SPMAT_CLASS_H
#define CONP_GA_SPMAT_CLASS_H
#include <map>
#include <vector>

namespace LAMMPS_NS
{
struct spMatEle {
  int i, j;
  double value;
  inline void operator()(int i_, int j_, double v_)
  {
    i     = i_;
    j     = j_;
    value = v_;
  }
};

typedef std::map<int, double> spMap_type;
class SpMatCLS {
  spMap_type* arr;
  std::vector<spMap_type> data;
  size_t L, R;
  inline void swap(int& i, int& j)
  {
    if (i < j) {
      if (j % 2 == 1) std::swap(i, j);
    } else {
      if (i % 2 == 0) std::swap(i, j);
      // std::swap(i, j);
    }
  }

public:
  SpMatCLS(size_t Natom);
  SpMatCLS(spMatEle* ele, size_t num_ele, spMatEle* eleG, size_t num_eleG, size_t Natom, size_t L_, size_t R_);
  spMap_type* atom(size_t i);
  size_t size();
  double at(size_t i, size_t j);
};
} // namespace LAMMPS_NS
#endif