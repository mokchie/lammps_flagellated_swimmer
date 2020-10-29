/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(active/fluct,FixActiveFluct)

#else

#ifndef LMP_FIX_ACTIVE_FLUCT_H
#define LMP_FIX_ACTIVE_FLUCT_H

#include "fix.h"

namespace LAMMPS_NS {

class FixActiveFluct : public Fix {
 public:
  FixActiveFluct(class LAMMPS *, int, char **);
  ~FixActiveFluct();
  int setmask();
  void setup(int);
  void pre_force(int);
  void calc_norm();
  void write_restart(FILE *);
  void restart(char *);

 private:
  double f0,prob,f_time,f_sigma,dt,alpha,beta,d,c,r_mom;
  tagint ind_min,ind_max;
  int time_style,ind_norm,norm_every,nn,read_restart_ind; 
  int step_start,num_steps,write_each,momentum_ind,groupbit_sol;
  double **norm, *dirr, *active_sp_dist;
  class RanMars *random;

  double generate_gamma(); 
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
