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

FixStyle(outflow,FixOutflow)

#else

#ifndef LMP_FIX_OUTFLOW_H
#define LMP_FIX_OUTFLOW_H

#include "fix.h"

namespace LAMMPS_NS {

class FixOutflow : public Fix {
 public:
  FixOutflow(class LAMMPS *, int, char **);
  ~FixOutflow();
  int setmask();
  void setup(int);
  void pre_exchange();
  void post_force(int);
  void write_restart(FILE *);
  void restart(char *);
  void reset_target(double);

 private:
  int num_shapes, num_shapes_tot, nnmax, max_shapes, num_flux, del_mol_max, del_mol_num, n_accum, noutf, max_area, qk_max;
  int groupbit_sol, n_press, nbinx, nbiny, nbinz, nbin, bin_max_shape, iter, cur_iter, mmax_iter, del_every, nw_max, read_restart_ind, numt, me, max_loc_shapes;
  int *ptype, *shapes_local, *tot_shapes, *num_bin_shapes, *displs, *rcounts, *flux_ind, *face_order, *glob_order;
  int **ndiv, **ind_shapes, **bin_shapes;
  tagint *del_mol_list; 
  double r_cut_n, r_cut_t, r_press, r_shear, r_shift, coeff_s, coeff_p, power, gamma, dr_inv, area_tot, rho_target;
  double binsize, binsizeinv, bboxlo[3], bboxhi[3];
  double *flux, *weight, *f_press, *area;
  double **x0, **aa, ***rot, ***beta;
  double ****veln, ****num, ****save_beta;
  MPI_Comm bc_comm;
  int bc_comm_rank, bc_comm_size; 

  int check_shapes(int);
  void setup_rot(int, double[], double[]);
  void rot_forward(double &, double &, double &, int);
  void rot_back(double &, double &, double &, int);
  void shape_decide();
  void grow_shape_arrays();  
  void grow_basic_arrays(int);
  void setup_bins();
  int coord2bin(double *);
  int check_bins(int, int);
  void recalc_force();
  void set_comm_groups();
  };

}

#endif
#endif
