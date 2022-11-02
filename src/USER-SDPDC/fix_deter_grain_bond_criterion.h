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

FixStyle(deter/grain/bond/criterion,FixDeterGrainBondCriterion)

#else

#ifndef LMP_FIX_DETER_GRAIN_BOND_CRITERION_H
#define LMP_FIX_DETER_GRAIN_BOND_CRITERION_H

#include "fix.h"

namespace LAMMPS_NS {

class FixDeterGrainBondCriterion : public Fix {
 public:
  FixDeterGrainBondCriterion(class LAMMPS *, int, char **);
  virtual ~FixDeterGrainBondCriterion();
  int setmask();
  virtual void init();
  void setup(int);
  void min_setup(int);
  virtual void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_vector(int);

  double memory_usage();
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 protected:
  int bnc; //critical bond number
  double xvalue,yvalue,zvalue;
  int varflag,iregion;
  char *xstr,*ystr,*zstr;
  char *idregion;
  int xvar,yvar,zvar,xstyle,ystyle,zstyle;
  double foriginal[3],foriginal_all[3];
  int force_flag;
  int nlevels_respa,ilevel_respa;
  int *totalnum_bond;
  int *totalnum_hardbond;
  int nmax;
  int btype;
  int commflag;

  int maxatom;
  double **sforce;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix setforce does not exist

Self-explanatory.

E: Variable name for fix setforce does not exist

Self-explanatory.

E: Variable for fix setforce is invalid style

Only equal-style variables can be used.

E: Cannot use non-zero forces in an energy minimization

Fix setforce cannot be used in this manner.  Use fix addforce
instead.

*/
