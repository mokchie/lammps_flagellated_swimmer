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

FixStyle(wall/lj126/force,FixWallLJ126Force)

#else

#ifndef LMP_FIX_WALL_LJ126_FORCE_H
#define LMP_FIX_WALL_LJ126_FORCE_H

#include "fix_wall_force.h"

namespace LAMMPS_NS {

class FixWallLJ126Force : public FixWallForce {
 public:
  FixWallLJ126Force(class LAMMPS *, int, char **);
  void precompute(int);
  double wall_particle(int, int, double);

 private:
  double coeff1[6],coeff2[6],coeff3[6],coeff4[6],offset[6];
  double force, total_force;
  int me;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Particle on or inside fix wall surface

Particles must be "exterior" to the wall in order for energy/force to
be calculated.

*/
