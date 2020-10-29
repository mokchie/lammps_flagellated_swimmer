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

FixStyle(inflow/periodic,FixInflowPeriodic)

#else

#ifndef LMP_FIX_INFLOW_PERIODIC_H
#define LMP_FIX_INFLOW_PERIODIC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixInflowPeriodic : public Fix {
 public:
  FixInflowPeriodic(class LAMMPS *, int, char **);
  ~FixInflowPeriodic();
  int setmask();
  void setup(int);
  void post_integrate();
  void end_of_step();
  void reset_target(double);

 private:
  int num_inflow, groupbit_reflect, part_size, spin_ind, part_size_track, part_size_vel;  
  int tot_tracked_part, save_max;
  int exist_max, new_max, new_list_max, sp_max, send_max, recv_max, sp_recv_max, send_back_max;
  int *list_new, *sp_list, *send_counts, *rcounts, *displs, *buf_send_back, *sp_recv;
  int *send_partner, *recv_partner;
  double binsize, skin, cut, sublo[3], subhi[3];
  double **x0, **aa, ***rot, **xp_save;
  double *send_exist, *send_new, *buf_send, *buf_recv;
  MPI_Comm *bc_send_comm, *bc_recv_comm;
  int *bc_send_comm_rank, *bc_send_comm_size, *bc_recv_comm_rank, *bc_recv_comm_size;

  void set_comm_groups();
  void setup_rot(int);
  void rot_forward(double &, double &, double &, int);
  void rot_back(double &, double &, double &, int);

  void consistency_check(int);
};

}

#endif
#endif
