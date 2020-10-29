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

FixStyle(e,FixE)

#else

#ifndef LMP_FIX_E_H
#define LMP_FIX_E_H

#include "fix.h"

namespace LAMMPS_NS {
  
class FixE : public Fix {
 public:
  FixE(class LAMMPS *, int, char **);
  ~FixE();
  int setmask();
  void init();
  void end_of_step();

 private:
  int iregion,mstyle;
  double en0,c_v;
  char *idregion;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix heat does not exist

Self-explanatory.

E: Fix heat group has no atoms

Self-explanatory.

E: Fix heat kinetic energy went negative

This will cause the velocity rescaling about to be performed by fix
heat to be invalid.

*/
