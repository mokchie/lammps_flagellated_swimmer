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

FixStyle(bond/create/break/seb,FixBondCreateBreakSeb)

#else

#ifndef LMP_FIX_BOND_CREATE_BREAK_SEB_H
#define LMP_FIX_BOND_CREATE_BREAK_SEB_H

#include "fix.h"
#include <string>

namespace LAMMPS_NS {

class FixBondCreateBreakSeb : public Fix {
 public:
  FixBondCreateBreakSeb(class LAMMPS *, int, char **);
  ~FixBondCreateBreakSeb();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
  void post_integrate();
  void post_integrate_respa(int, int);

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  double compute_vector(int);
  double memory_usage();

 private:
  int me;
  int iatomtype,jatomtype;
  int btype,seed;
  int imaxbond,jmaxbond;
  int inewtype,jnewtype;
  double cutsq_on, cutsq_off,fraction_b, fraction_c;
  int overflow;
  tagint lastcheck;

  int *bondcount;
  int createcount,createcounttotal;
  int createcount_try, createcounttotal_try;
  double kon, kon_total;
  int breakcount,breakcounttotal;
  int breakcount_try, breakcounttotal_try;
  double koff, koff_total;
  int nmax;
  tagint *partner_b, *partner_c;
  double *distsq_b,*distsq_c,*probability_b,*probability_c;

  int ncreate,maxcreate,ncreate_try;
  double average_kon;
  tagint **created;
  int nbreak,maxbreak,nbreak_try;
  double average_koff;
  tagint **broken;

  class RanMars *random;
  class NeighList *list;
  
  int countflag,commflag;
  int nlevels_respa;

  int type_flag, catch_flag;
  double dt;
  double k0_on, k0_off, spring, sigma_on, sigma_off, temp;
  double sigma, delta;
  double k0_off_slip, k0_off_catch, delta_slip, delta_catch;
  int const_on_rate;
  int ieach, jeach;
  double *vec_fraction_b, *vec_fraction_c;
  double dembo_create (double);
  double dembo_break (int, int, double);
  double catch_slip_create(double);
  double catch_slip_break(double);
  double bell_create (double);
  double bell_break (int, int, double);

  std::string interface_bonds;
  int set_template_bonds;


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

E: Invalid atom type in fix bond/create/break command

Self-explanatory.

E: Invalid bond type in fix bond/create/break command

Self-explanatory.

E: Cannot use fix bond/create/break with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Inconsistent iparam/jparam values in fix bond/create/break command

If itype and jtype are the same, then their maxbond and newtype
settings must also be the same.

E: Fix bond/create/break cutoff is longer than pairwise cutoff

This is not allowed because bond creation is done using the
pairwise neighbor list.

E: Fix bond/create/break requires special_bonds lj = 0,1,1

Self-explanatory.

E: Fix bond/create/break requires special_bonds coul = 0,1,1

Self-explanatory.

W: Created bonds will not create angles, dihedrals, or impropers

See the doc page for fix bond/create/break for more info on this
restriction.

E: Could not count initial bonds in fix bond/create/break

Could not find one of the atoms in a bond on this processor.

E: New bond exceeded bonds per atom in fix bond/create/break

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

E: New bond exceeded special list size in fix bond/create/break

See the special_bonds extra command for info on how to leave space in
the special bonds list to allow for additional bonds to be formed.

*/
