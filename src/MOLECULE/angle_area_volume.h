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

#ifdef ANGLE_CLASS

AngleStyle(area/volume,AngleAreaVolume)

#else

#ifndef LMP_ANGLE_AREA_VOLUME_H
#define LMP_ANGLE_AREA_VOLUME_H

#include <cstdio>
#include "angle.h"

namespace LAMMPS_NS {

class AngleAreaVolume : public Angle {
 public:
  AngleAreaVolume(class LAMMPS *);
  virtual ~AngleAreaVolume();
  virtual void compute(int, int);
  void coeff(int, char **);
  double equilibrium_angle(int){return 1.0;};
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  double single(int, int, int, int){return 0.0;};
 protected:
  double *ka, *a0, *kv, *v0, *kl; 
  tagint n_mol_max, n_mol_limit;
  double *dath, *datt;

  void allocate();
};

}
#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for angle coefficients

Self-explanatory.  Check the input script or data file.

E: Individual has wrong value or is not set! Using angle area/volume only possible with individual =1

Self-explanatory.  Check the input script second parameter at atom_style.

*/

