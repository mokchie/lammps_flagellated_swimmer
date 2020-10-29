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

FixStyle(inflow,FixInflow)

#else

#ifndef LMP_FIX_INFLOW_H
#define LMP_FIX_INFLOW_H

#include "fix.h"

namespace LAMMPS_NS {

class RanMars;

class FixInflow : public Fix {
 public:
  FixInflow(class LAMMPS *, int, char **);
  ~FixInflow();
  int setmask();
  void setup(int);
  void post_integrate();
  void pre_exchange();
  void post_force(int);
  void reset_target(double);

 private:
  int num_shapes, num_shapes_tot, mirror, nnmax, max_count;  
  int max_shapes, num_flow, num_at_types, num_flow_tot, me;
  int *ptype, *shapes_local, *tot_shapes, *num_bin_shapes, *at_type, *at_groupbit;
  int **ndiv, **ind_shapes, **bin_shapes, **info_flow;
  int nbinx, nbiny, nbinz, nbin, bin_max_shape;
  int groupbit_p, ind_press, n_press;
  double r_cut_n, r_cut_t, r_press, kbt, dtv;
  double binsize, binsizeinv, bboxlo[3], bboxhi[3];
  double ***area, *f_press, **ncount;
  double **x0, **aa, ****vel, ***rot, *at_dens, *vls;
  RanMars *ranmars0;

  int check_shapes(int);
  void setup_rot(int, double[], double[]);
  void rot_forward(double &, double &, double &, int);
  void rot_back(double &, double &, double &, int);
  void shape_decide();
  void grow_shape_arrays();  
  void setup_bins();
  int coord2bin(double *);
  int check_bins(int, int);
};

}

#endif
#endif
