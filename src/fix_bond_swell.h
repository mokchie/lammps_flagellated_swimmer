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

FixStyle(bond/swell,FixBondSwell)

#else

#ifndef LMP_FIX_BOND_SWELL_H
#define LMP_FIX_BOND_SWELL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondSwell : public Fix {
 public:
  FixBondSwell(class LAMMPS *, int, char **);
  ~FixBondSwell();
  int setmask();
  void init();
  void post_integrate();
  void post_integrate_respa(int,int);
  void init_list(int /*id*/, NeighList *ptr);

 private:
  int me,nprocs;
  int groupbit_rho, btype;
  double ks, rhoc, t_start, rmax_factor, cutoff;
  int set_weight_ind;
  double dr_inv;
  int nlevels_respa;
  class NeighList *list;
 protected:
  double ***weight1,***weight2;
  void set_weight();
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  int pack_forward_comm(int , int *, double *, int, int *);
  void unpack_forward_comm(int , int , double *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid bond type in fix bond/swell command

Self-explanatory.

E: Cannot use fix bond/break with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Cannot yet use fix bond/break with this improper style

This is a current restriction in LAMMPS.

E: Fix bond/break needs ghost atoms from further away

This is because the fix needs to walk bonds to a certain distance to
acquire needed info, The comm_modify cutoff command can be used to
extend the communication range.

*/
