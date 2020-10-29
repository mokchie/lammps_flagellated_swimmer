/* ----------------------------------------------------------------------
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

FixStyle(solid/bound,FixSolidBound)

#else

#ifndef LMP_FIX_SOLID_BOUND_H
#define LMP_FIX_SOLID_BOUND_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSolidBound : public Fix {
 public:
  FixSolidBound(class LAMMPS *, int, char **);
  ~FixSolidBound();
  int setmask();
  void setup(int);
  void min_setup(int);
  void post_integrate();
  void post_force(int);

 private:
  int ind_shear, ind_press, num_shapes, num_shapes_tot, ind_read_shear, ind_write_shear; 
  int nnmax, mirror, n_per, n_press, iter, cur_iter, mmax_iter, n_accum, max_count;  
  int max_shapes, s_apply, p_apply, numt, numt_s, numt_loc, groupbit_s, groupbit_p, bin_max_shape, nbinx, nbiny, nbinz, nbin;
  int *ptype, *refl, **ndiv, *face_order, *shapes_local, *shapes_global_to_local;
  int *tot_shapes, **ind_shapes, **bin_shapes, *num_bin_shapes;
  double **x0, **aa, **vel, ***rot, *f_press, *weight;
  double coeff, power, r_cut_n, r_cut_t, r_shear, r_press, dr_inv;
  double binsize, binsizeinv, bboxlo[3], bboxhi[3];
  double ****velx, ****vely, ****velz, ****num;
  double ****fsx, ****fsy, ****fsz;

  int check_shapes(int);
  void grow_shape_arrays();
  void setup_rot(int, double[], double[]);
  void rot_forward(double &, double &, double &, int);
  void rot_back(double &, double &, double &, int);
  void shape_decide();
  void setup_bins();
  int check_bins(int, int);
  int coord2bin(double *);
  void recalc_force(); 
  void read_shear_forces();
  void write_shear_forces();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal fix solid/bound command

Self-explanatory.  Check the input script

E: Could not open input solid-boundary file

Self-explanatory. Check solid-boundary file 

E: Could not open input pressure file

Self-explanatory. Check pressure file

*/

