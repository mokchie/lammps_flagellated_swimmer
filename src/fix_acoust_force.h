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

FixStyle(acoust/force,FixAcoustForce)

#else

#ifndef LMP_FIX_ACOUST_FORCE_H
#define LMP_FIX_ACOUST_FORCE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixAcoustForce : public Fix {
 public:
  FixAcoustForce(class LAMMPS *, int, char **);
  ~FixAcoustForce();
  int setmask();
  void setup(int);
  void post_force(int);

 private:
  int nw;
  double *coeff, *lambda, **dir; 
  tagint n_mol_max, n_mol_limit;
  double *dath, *datt;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
