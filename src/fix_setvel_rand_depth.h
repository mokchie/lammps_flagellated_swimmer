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

FixStyle(setvel/rand/depth,FixSetVelRandDepth)

#else

#ifndef LMP_FIX_SETVEL_RAND_DEPTH_H
#define LMP_FIX_SETVEL_RAND_DEPTH_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSetVelRandDepth : public Fix {
 public:
  FixSetVelRandDepth(class LAMMPS *, int, char **);
  ~FixSetVelRandDepth();
  int setmask();
  void init();
  void post_integrate();
  void post_force(int);

 private:
  double xvalue,yvalue,zvalue;
  int xstyle,ystyle,zstyle;
  int ind_const,iregion,step_each;
  char *idregion;
  class RanPark *random;
  char direction;
  double l1,l2;
};

}

#endif
#endif

