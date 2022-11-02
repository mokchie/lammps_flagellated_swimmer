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

FixStyle(spring/mol,FixSpringMol)

#else

#ifndef LMP_FIX_SPRING_MOL_H
#define LMP_FIX_SPRING_MOL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSpringMol : public Fix {
 public:
  FixSpringMol(class LAMMPS *, int, char **);
  ~FixSpringMol();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);

 private:
  double k_spring;
  int ilevel_respa;
  double espring;
  int m_start;
  int m_end;
  double **xc0;
  double **xcm;
  double *masstot;
  double **ftotal;    

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: R0 < 0 for fix spring command

Equilibrium spring length is invalid.

E: Fix spring couple group ID does not exist

Self-explanatory.

E: Two groups cannot be the same in fix spring couple

Self-explanatory.

*/
