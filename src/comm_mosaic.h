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

#ifndef LMP_COMM_MOSAIC_H
#define LMP_COMM_MOSAIC_H

#include "comm.h"

namespace LAMMPS_NS {

class CommMosaic : public Comm {
 public:
  CommMosaic(class LAMMPS *);
  CommMosaic(class LAMMPS *, class Comm *);
  virtual ~CommMosaic();

  void init();
  void setup();                        // setup comm pattern
  void forward_comm(int dummy = 0);    // forward comm of atom coords
  void reverse_comm();                 // reverse comm of forces
  void exchange();                     // move atoms to new procs
  void borders();                      // setup list of atoms to comm

  void forward_comm_pair(class Pair *);    // forward comm from a Pair
  void reverse_comm_pair(class Pair *);    // reverse comm from a Pair
  virtual void forward_comm_fix(class Fix *, int size=0);
                                                   // forward comm from a Fix
  virtual void reverse_comm_fix(class Fix *, int size=0);
                                                   // reverse comm from a Fix
  virtual void reverse_comm_fix_variable(class Fix *);
                                     // variable size reverse comm from a Fix
  void forward_comm_compute(class Compute *);  // forward from a Compute
  void reverse_comm_compute(class Compute *);  // reverse from a Compute
  void forward_comm_dump(class Dump *);    // forward comm from a Dump
  void reverse_comm_dump(class Dump *);    // reverse comm from a Dump

  void forward_comm_array(int, double **);         // forward comm of array
  int exchange_variable(int, double *, double *&);  // exchange on neigh stencil
  bigint memory_usage();

 protected: 
  int n_neigh, n_neigh_orig, n_own, bin_max;
  bigint own_bit[6];
  int nbp[3], period_own[3];
  double bin_size_sq[3], bin_size_inv[3], bin_diag;
  int **own_bin_orig, *nbinh_orig, *my_neigh_orig;
  int **own_bin, *nbinh, *my_neigh, *twice_comm, *neigh_ind, *periodicity;
  bigint *send_bit, *exchange_bit;

  void read_user_file();
  void process_bin_data(int &, int &, int *, int *, int *, int *, double *, double *);
  void build_comm_matrix(int, bigint ***, bigint *, int *, int *, int *, int *, int *, double *, int *, int *, int *, int **, int);
  void bin_dist_periodic(int *, int &, int &, int, int *);
  void bin_dist(int *, int &, int &, int *, int *);
  void bin_coords(int *, int &, int *);
  void bin_coords_local(int *, int &, int *);
  int coords_to_bin(double *, int *, int *, double *);
  int coords_to_bin_exchange(double *, int *);

  int nswap;                        // # of swaps to perform = sum of maxneed
  int maxswap; 
  int *sendnum,*recvnum;            // # of atoms to send/recv in each swap
  int *sendproc,*recvproc;          // proc to send/recv to/from at each swap
  int *size_forward_recv;           // # of values to recv in each forward comm
  int *size_reverse_send;           // # to send in each reverse comm
  int *size_reverse_recv;           // # to recv in each reverse comm
  int *pbc_flag;                    // general flag for sending atoms thru PBC
  int **pbc;                        // dimension flags for PBC adjustments

  int *firstrecv;                   // where to put 1st recv atom in each swap
  int **sendlist;                   // list of atoms to send in each swap
  int *maxsendlist;                 // max size of send list for each swap

  double *buf_send;                 // send buffer for all comm
  double *buf_recv;                 // recv buffer for all comm
  int maxsend,maxrecv;              // current size of send/recv buffer
  int bufextra;                     // extra space beyond maxsend in send buffer
  int smax,rmax;             // max size in atoms of single borders send/recv

  void init_buffers();

  virtual void grow_send(int, int);         // reallocate send buffer
  virtual void grow_recv(int);              // free/allocate recv buffer
  virtual void grow_list(int, int);         // reallocate one sendlist
  virtual void grow_swap(int);              // grow swap and multi arrays
  virtual void allocate_swap(int);          // allocate swap arrays
  virtual void free_swap();                 // free swap arrays
};

}

#endif

/* ERROR/WARNING messages:

*/
