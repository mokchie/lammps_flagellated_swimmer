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

#ifdef DIHEDRAL_CLASS

DihedralStyle(bend/dual,DihedralBendDual)

#else

#ifndef LMP_DIHEDRAL_BEND_DUAL_H
#define LMP_DIHEDRAL_BEND_DUAL_H

#include <cstdio>
#include "dihedral.h"

namespace LAMMPS_NS {

class DihedralBendDual : public Dihedral {
 public:
  DihedralBendDual(class LAMMPS *);
  virtual ~DihedralBendDual();
  virtual void compute(int, int);
  void coeff(int, char **);
  void write_restart(FILE *);
  void read_restart(FILE *);

 protected:
  double *k, *c_ij, *n_sum, *n_i, *z_i, *dot_k, *dot_l, *s_k, *s_l, *nr_ij;
  double **n_ij, **r_ij, **r_ik, **r_jk, **r_il, **r_jl, **dc_ij, **A_ij, **B_ij, **M_ij, **fh;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Individual has wrong value or is not set! Using dihedral bend only possible with individual =1

Self-explanatory.  Check the input script second parameter at atom_style.

E: Incorrect args in dihedral_coeff command

Self-explanatory. Check the input script or data file.

*/

