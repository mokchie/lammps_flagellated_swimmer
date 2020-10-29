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

DihedralStyle(bend/zero,DihedralBendZero)

#else

#ifndef LMP_DIHEDRAL_BEND_ZERO_H
#define LMP_DIHEDRAL_BEND_ZERO_H

#include <cstdio>
#include "dihedral.h"

namespace LAMMPS_NS {

class DihedralBendZero : public Dihedral {
 public:
  DihedralBendZero(class LAMMPS *);
  virtual ~DihedralBendZero();
  virtual void compute(int, int);
  void coeff(int, char **);
  void write_restart(FILE *);
  void read_restart(FILE *);

 protected:
  double *k;

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

