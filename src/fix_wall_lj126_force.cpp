/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cmath>
#include "fix_wall_lj126_force.h"
#include <iostream>
#include <mpi.h>

#include <cstring>
#include <cstdlib>
#include "fix_bond_create_break_seb.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include <fstream>
#include <vector>
#include <map>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixWallLJ126Force::FixWallLJ126Force(LAMMPS *lmp, int narg, char **arg) :
  FixWallForce(lmp, narg, arg) {
    MPI_Comm_rank(world,&me);
    total_force = 0;
}

/* ---------------------------------------------------------------------- */

void FixWallLJ126Force::precompute(int m)
{
  coeff1[m] = 48.0 * epsilon[m] * pow(sigma[m],12.0);
  coeff2[m] = 24.0 * epsilon[m] * pow(sigma[m],6.0);
  coeff3[m] = 4.0 * epsilon[m] * pow(sigma[m],12.0);
  coeff4[m] = 4.0 * epsilon[m] * pow(sigma[m],6.0);

  double r2inv = 1.0/(cutoff[m]*cutoff[m]);
  double r6inv = r2inv*r2inv*r2inv;
  offset[m] = r6inv*(coeff3[m]*r6inv - coeff4[m]);
}

/* ----------------------------------------------------------------------
   interaction of all particles in group with a wall
   m = index of wall coeffs
   which = xlo,xhi,ylo,yhi,zlo,zhi
   error if any particle is on or behind wall
------------------------------------------------------------------------- */

double FixWallLJ126Force::wall_particle(int m, int which, double coord)
{
  double delta,rinv,r2inv,r6inv,fwall;

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int dim = which / 2;
  int side = which % 2;
  if (side == 0) side = -1;

  int onflag = 0;

  force = 0;
  total_force = 0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (side < 0) delta = x[i][dim] - coord;
      else delta = coord - x[i][dim];
      if (delta >= cutoff[m]) continue;
      if (delta <= 0.0) {
        onflag = 1;
        continue;
      }
      rinv = 1.0/delta;
      r2inv = rinv*rinv;
      r6inv = r2inv*r2inv*r2inv;
      fwall = side * r6inv*(coeff1[m]*r6inv - coeff2[m]) * rinv;
      f[i][dim] -= fwall;
      force += fwall;
      ewall[0] += r6inv*(coeff3[m]*r6inv - coeff4[m]) - offset[m];
      ewall[m+1] += fwall;
    }

  MPI_Allreduce(&force,&total_force,1,MPI_DOUBLE,MPI_SUM,world);

  if (onflag) error->one(FLERR,"Particle on or inside fix wall surface");

  return total_force;
}
