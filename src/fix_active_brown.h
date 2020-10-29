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

FixStyle(active/brown,FixActiveBrown)

#else

#ifndef LMP_FIX_ACTIVE_BROWN_H
#define LMP_FIX_ACTIVE_BROWN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixActiveBrown : public Fix {
 public:
  FixActiveBrown(class LAMMPS *, int, char **);
  ~FixActiveBrown();
  int setmask();
  void init();
  void setup(int);
  void post_force(int);
  double memory_usage();

 private:
  double prop_force, rot_diff;
  int init_orien;
  double sqdr;

  class RanMars *random;
  int seed;

};

}

#endif
#endif
