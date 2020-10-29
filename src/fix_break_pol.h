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

FixStyle(break/pol,FixBreakPol)

#else

#ifndef LMP_FIX_BREAK_POL_H
#define LMP_FIX_BREAK_POL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBreakPol : public Fix {
 public:
  FixBreakPol(class LAMMPS *, int, char **);
  ~FixBreakPol();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void post_integrate();
  void post_integrate_respa(int, int);
  double memory_usage();

 private:
  int me;
  int seed;
  tagint lastcheck;

  int nbreak;

  int cleave_ind;
  char fname[FILENAME_MAX];


  class RanMars *random;
  class NeighList *list;

  int nlevels_respa;

  int type_flag;
  int atype, btype;
  double krate;
  double vec_fraction;
  double bond_break ();
  void check_ghosts();

  // DEBUG

  void print_bb();
  void print_copy(const char *, tagint, int, int, int, int *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix break/pol command

Self-explanatory.

E: Invalid bond type in fix break/pol command

Self-explanatory.

E: Cannot use fix break/pol with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Inconsistent iparam/jparam values in fix break/pol command

If itype and jtype are the same, then their maxbond and newtype
settings must also be the same.

E: Fix break/pol cutoff is longer than pairwise cutoff

This is not allowed because bond creation is done using the
pairwise neighbor list.

E: Fix break/pol requires special_bonds lj = 0,1,1

Self-explanatory.

E: Fix break/pol requires special_bonds coul = 0,1,1

Self-explanatory.

W: Created bonds will not create angles, dihedrals, or impropers

See the doc page for fix break/pol for more info on this
restriction.

E: Could not count initial bonds in fix break/pol

Could not find one of the atoms in a bond on this processor.

E: New bond exceeded bonds per atom in fix break/pol

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

E: New bond exceeded special list size in fix break/pol

See the special_bonds extra command for info on how to leave space in
the special bonds list to allow for additional bonds to be formed.

*/
