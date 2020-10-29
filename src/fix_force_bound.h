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

FixStyle(force/bound,FixForceBound)

#else

#ifndef LMP_FIX_FORCE_BOUND_H
#define LMP_FIX_FORCE_BOUND_H

#include "fix.h"

namespace LAMMPS_NS {

class FixForceBound : public Fix {
 public:
  FixForceBound(class LAMMPS *, int, char **);
  ~FixForceBound();
  int setmask();
  void setup(int);
  void min_setup(int);
  void post_integrate();
  void post_force(int);
  void write_restart(FILE *);
  void restart(char *);

 private:
  int ind_shear, ind_press, num_shapes, num_shapes_tot, le_max, read_restart_ind; 
  int ind_force, ind_press_cell, ind_force_cell, n_force, f_apply, n_accum;  
  int groupbit_solid, nnmax, mirror, n_per, n_press, iter, groupbit_no_move, cur_iter, mmax_iter;  
  int max_shapes, s_apply, p_apply, numt, numt_s, numt_loc, groupbit_s, groupbit_p;
  int groupbit_p_cell, groupbit_force, groupbit_force_cell, le_npart;
  int ntri, npart, max_npart, max_ntri, npart_loc, build_tri_delay, build_tri_next;
  int groupbit_comm, max_count, groupbit_inner, norm_count, cell_update, max_faces, ind_bounce;
  int *part_degree, *tags_part, *part_part_loc;
  int *ptype, *refl, **ndiv, *face_order, *tot_shapes, **ind_shapes, *shapes_local, *shapes_global_to_local;
  int *norm_ind, *norm_list, *num_faces, **ind_faces;
  int **bin_shapes, *num_bin_shapes, **bin_vertex, *num_bin_vertex;
  int **bin_stensil, *num_bin_stensil, *le_sh;
  int nbinx, nbiny, nbinz, nbin, bin_max_shape, bin_max_vertex, max_tri_part;
  int ntri_tot, npart_tot, max_ntri_tot, max_npart_tot; 
  tagint min_ind, max_ind, tags_tot, max_tags_tot;
  int **tri_part, **part_tri, *part_list, *tri_list, *tri_type, *part_loc_map;
  tagint *tri_mol, *part_tags;
  double **x_part, **v_part, **v_delt, **f_part;
  double **x0, **aa, **vel, ***rot, *f_press, *f_force, *weight;
  double coeff, power, r_cut_n, r_cut_t, r_shear, r_press, prd[3], d_cut, d_cut_sq, comm_cut, r_force, dr_inv;
  double ****velx, ****vely, ****velz, ****num;
  double ****fsx, ****fsy, ****fsz;
  double **norm, **edge1, **edge2, **dif1, **dif2, **dd12;

  int nswap;                        // # of swaps to perform = sum of maxneed
  int recvneed[3][2];               // # of procs away I recv atoms from
  int sendneed[3][2];               // # of procs away I send atoms to
  int maxneed[3];  
  int size_border, size_forward, size_reverse, size_tri;
  int maxsend, maxrecv, maxswap;
  int *sendnum, *recvnum;
  int *size_forward_recv;           // # of values to recv in each forward comm
  int *size_reverse_send;           // # to send in each reverse comm
  int *size_reverse_recv;           // # to recv in each reverse comm
  int *sendproc,*recvproc;          // proc to send/recv to/from at each swap
  int *maxsendlist, **sendlist, *firstrecv;
  double *buf_send, *buf_recv;
  double *slablo,*slabhi;
  int *pbc_flag;                    // general flag for sending atoms thru PBC
  int **pbc;                        // dimension flags for PBC adjustments
  double binsize, binsizeinv, bboxlo[3], bboxhi[3];

  int n_neigh, user_part, me;
  int nbp[3], nbp_max[3], period_own[3], bin_ext[3], nbp_loc[3];
  bigint *send_bit;
  bigint ***m_comm;
  double box_min[3];

  int check_shapes(int);
  void grow_shape_arrays();
  void setup_rot(int, double[], double[]);
  void rot_forward(double &, double &, double &, int);
  void rot_back(double &, double &, double &, int);
  void shape_decide();
  void recalc_force();
  void setup_bins();
  double bin_distance(int, int, int);
  int check_bins(int, int); 
  void build_tri_lists();
  void bin_vertices();
  int coord2bin(double *);
  void borders();
  void communicate();
  void reverse_communicate(int);
  void local_faces(int);
  void grow_send(int);
  void grow_list(int, int);
  void grow_recv(int);
  void grow_tri_arrays();
  void grow_tri_lists();
  void grow_part_arrays();
  void grow_part_lists();
  void grow_face_arrays();
  int pack_border_h(int, double *, int, int *);
  int unpack_border_h(double *);
  int pack_comm_h(int, int *, double *, int, int *);
  void unpack_comm_h(int, int, double *);
  int pack_reverse_h(int, int, double *, int);
  void unpack_reverse_h(int, int *, double *, int);
  void face_decide();
  void calc_norm(int);
  double find_root(double *);
  double newton(double, double *, double *);
  double solve_quadratic(double *);
  void move_norm_arrays(int);
  void lees_edwards(int);           // lees-edwards
  void allocate_swap(int);          // allocate swap arrays
  void free_swap();                 // free swap arrays
};

}

#endif
#endif
