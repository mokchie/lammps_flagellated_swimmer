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

FixStyle(dump/qmatrix,FixDumpQMatrix)

#else

#ifndef LMP_FIX_DUMPQMATRIX_H
#define LMP_FIX_DUMPQMATRIX_H

#include "fix.h"

namespace LAMMPS_NS {

class FixDumpQMatrix : public Fix {
 public:
  FixDumpQMatrix(class LAMMPS *, int, char **);
  ~FixDumpQMatrix();
  int setmask();
  void init();
  void post_integrate();

 private:
  int me;
  int step_each;
  FILE *fp;
  int actuation_type; //0: angle; 1: bond
};

}

#endif
#endif

