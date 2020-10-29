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

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "fix_active_brown.h"
#include "atom.h"
#include "atom_masks.h"
#include "update.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- 
Syntax: fix  ID group-ID  active/brown  prop_force  rot_diff  init_orien  keywords arguments
arg[]        0  1         2             3           4         5           iarg     iarg+...

prop_force:  propulsion force
rot_diff:  rotational diffusion
init_orien: initialize orientations

keywords:   seed
arguments:
seed:       seed_number
-------------------------------------------------------------------------*/

FixActiveBrown::FixActiveBrown(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix active/brown command");

  dynamic_group_allow = 1;

  prop_force = force->numeric(FLERR,arg[3]);
  rot_diff = force->numeric(FLERR,arg[4]);
  init_orien = force->inumeric(FLERR,arg[5]);

  // optional args
  int seed = 12345;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"seed") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix active/brown command");
      seed = force->inumeric(FLERR,arg[iarg+1]);
      if (seed <= 0) error->all(FLERR,"Illegal fix active/brown command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix active/brown command");
  }

  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);

}

/* ---------------------------------------------------------------------- */

FixActiveBrown::~FixActiveBrown()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixActiveBrown::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixActiveBrown::init()
{
  // rotational diffusion parameters
  double dt = update->dt;
  sqdr = sqrt(12*2*rot_diff*dt);

  if (init_orien) {
    int i;
    int nlocal = atom->nlocal;
    double **omega = atom->omega;
    double osq;

    for (i = 0; i < nlocal; i++) {
      omega[i][0] = random->uniform()-0.5;
      omega[i][1] = random->uniform()-0.5;
      omega[i][2] = random->uniform()-0.5;

      osq = omega[i][0]*omega[i][0] + omega[i][1]*omega[i][1] + omega[i][2]*omega[i][2];
      osq = 1.0/sqrt(osq);

      omega[i][0] *= osq;
      omega[i][1] *= osq;
      omega[i][2] *= osq;
    }
  }

}

/* ---------------------------------------------------------------------- */

void FixActiveBrown::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixActiveBrown::post_force(int vflag)
{
  int i;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double **omega = atom->omega;

  double osq, xisq, xi[3], o[3];

 // fprintf(screen,"here dt=%g, step=%li, %g, sqdr=%g Dr=%g \n",
   //       update->dt,update->ntimestep,sqrt(12*2*rot_diff*update->dt), sqdr, rot_diff);

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      xi[0] = sqdr*(random->uniform() - 0.5);
      xi[1] = sqdr*(random->uniform() - 0.5);
      xi[2] = sqdr*(random->uniform() - 0.5);

      o[0] = xi[1] * omega[i][2] - xi[2] * omega[i][1];
      o[1] = xi[2] * omega[i][0] - xi[0] * omega[i][2];
      o[2] = xi[0] * omega[i][1] - xi[1] * omega[i][0];

      omega[i][0] += o[0];
      omega[i][1] += o[1];
      omega[i][2] += o[2];

      osq = omega[i][0]*omega[i][0] + omega[i][1]*omega[i][1] + omega[i][2]*omega[i][2];
      osq = 1.0/sqrt(osq);

      omega[i][0] *= osq;
      omega[i][1] *= osq;
      omega[i][2] *= osq;

      f[i][0] += prop_force*omega[i][0];
      f[i][1] += prop_force*omega[i][1];
      f[i][2] += prop_force*omega[i][2];
    }

}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixActiveBrown::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}
