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

FixStyle(force/bound/2d,FixForceBound2D)

#else

#ifndef LMP_FIX_FORCE_BOUND2D_H
#define LMP_FIX_FORCE_BOUND2D_H

#include "fix.h"

namespace LAMMPS_NS {

class FixForceBound2D : public Fix {
 public:
  FixForceBound2D(class LAMMPS *, int, char **);
  ~FixForceBound2D();
  int setmask();
  void setup(int);
  void min_setup(int);
  void post_integrate();
  void post_force(int);
  void write_restart(FILE *);
  void restart(char *);

 private:
  int ind_shear, ind_press, num_shapes, num_shapes_tot, le_max, read_restart_ind, n_accum;  
  int groupbit_solid, nnmax, mirror, n_per, n_press, iter, groupbit_no_move, cur_iter, mmax_iter;  
  int max_shapes, s_apply, p_apply, numt, numt_s, numt_loc, groupbit_s, groupbit_p, le_npart;
  int nbond, npart, max_npart, max_nbond, npart_loc, build_bond_delay, build_bond_next;
  int groupbit_comm, max_count, groupbit_inner, cell_update, max_faces, ind_bounce;
  int *part_degree, *tags_part, *part_part_loc;
  int *ptype, *refl, *ndiv, *face_order, *tot_shapes, **ind_shapes, *shapes_local, *shapes_global_to_local;
  int *num_faces, **ind_faces;
  int **bin_shapes, *num_bin_shapes, **bin_vertex, *num_bin_vertex;
  int **bin_stensil, *num_bin_stensil, *le_sh;
  int nbinx, nbiny, nbin, bin_max_shape, bin_max_vertex, max_bond_part;
  int nbond_tot, npart_tot, max_nbond_tot, max_npart_tot;
  tagint min_ind, max_ind, tags_tot, max_tags_tot;
  int **bond_part, **part_bond, *part_list, *bond_type, *part_loc_map;
  tagint *bond_mol, *part_tags;
  double **x_part, **v_part, **v_delt;
  double **x0, *aa, **vel, **rot, *f_press, *weight;
  double coeff, power, r_cut_n, r_cut_t, r_shear, r_press, prd[3], d_cut, d_cut_sq, comm_cut, dr_inv;
  double ***velx, ***vely, ***num, ***fsx, ***fsy;

  int nswap;                        // # of swaps to perform = sum of maxneed
  int recvneed[3][2];               // # of procs away I recv atoms from
  int sendneed[3][2];               // # of procs away I send atoms to
  int maxneed[3];  
  int size_border, size_forward, size_reverse, size_bond;
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
  double binsize, binsizeinv, bboxlo[2], bboxhi[2];

  int n_neigh, user_part, me;
  int nbp[3], nbp_max[3], period_own[3], bin_ext[3], nbp_loc[3];
  bigint *send_bit;
  bigint ***m_comm;
  double box_min[3];

  int check_shapes(int);
  void grow_shape_arrays();
  void setup_rot(int, double[]);
  void rot_forward(double &, double &, int);
  void rot_back(double &, double &, int);
  void shape_decide();
  void recalc_force();
  void setup_bins();
  double bin_distance(int, int);
  int check_bins(int, int); 
  void build_bond_lists();
  void bin_vertices();
  int coord2bin(double *);
  void borders();
  void communicate();
  void reverse_communicate();
  void local_faces(int);
  void grow_send(int);
  void grow_list(int, int);
  void grow_recv(int);
  void grow_bond_lists();
  void grow_part_arrays();
  void grow_part_lists();
  void grow_face_arrays();
  int pack_border_h(int, double *, int, int *);
  int unpack_border_h(double *);
  int pack_comm_h(int, int *, double *, int, int *);
  void unpack_comm_h(int, int, double *);
  int pack_reverse_h(int, int, double *);
  void unpack_reverse_h(int, int *, double *);
  void face_decide();
  double solve_quadratic(double *);
  void lees_edwards(int);           // lees-edwards
  void allocate_swap(int);          // allocate swap arrays
  void free_swap();                 // free swap arrays
};

}

#endif
#endif
