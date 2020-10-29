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
/*------------------------------------------------------------------------
		Created on  : 17 June 2015
		Modified on : 19 Dec 2017
		Created by  : Masoud Hoore
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(polymer/activate,FixPolymerActivate)

#else

#ifndef LMP_FIX_POLYMER_ACTIVATE_H
#define LMP_FIX_POLYMER_ACTIVATE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPolymerActivate : public Fix {
/*----------------------------------------------------------------------------------
Hint:
* Everything which is aware of the specified header, is aware of a public member.
* Only the children (and their children) are aware of protected member.
* No one but the base here is aware of a private member.
------------------------------------------------------------------------------------*/
 public:
  FixPolymerActivate(class LAMMPS *, int, char **);
 // ~FixPolymerActivate();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
  void final_integrate();
  void final_integrate_respa(int, int);

 private:
  int nevery, itype, jtype, btype;
  double cutoff, angoff; // distance and angle cutoffs
  int arg_shift, styleflag;
  int nlevels_respa;
  double delx, dely, delz, delx2, dely2, delz2, distsq, distsq2, dist, cutoffsq, cos_theta, cos_angoff;
  int num_dens, ang_flag;
  class NeighList *list;

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
