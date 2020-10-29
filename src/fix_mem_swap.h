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

#ifdef FIX_CLASS

FixStyle(mem/swap,FixMemSwap)

#else

#ifndef LMP_FIX_MEM_SWAP_H
#define LMP_FIX_MEM_SWAP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMemSwap : public Fix {
 public:
  FixMemSwap(class LAMMPS *, int, char **);
  ~FixMemSwap();
  int setmask();
  void init();
  void pre_exchange();
  double memory_usage();
  void bad_topology(tagint, tagint, tagint, tagint, tagint, tagint, tagint, tagint);
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);

 private:
  int btype, atype, dtype;
  int nevery;
  int ndih_swap, ndih_swap_all;
  double normal[3], area, prob;

  double *data_dih, *data_dih_all;
  tagint *data_adj, *data_adj_all;

  int seed;

  class RanMars *random;

  void communicate(int, int);

  int create_bond(int, int);
  int create_angle(int, int, int, double*, double*);
  int create_dihedral(int, int, int, int);

  int delete_bond(int, int);
  int delete_angle(int, int, int);
  int delete_dihedral(int, int, int, int);

  char fname[FILENAME_MAX];

};

}

#endif
#endif
