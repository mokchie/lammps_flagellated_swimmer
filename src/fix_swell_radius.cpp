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

#include <cstring>
#include <cstdlib>
#include "fix_swell_radius.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "comm.h"
#include "random_park.h"
#include "universe.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;
enum{CONSTANT,UPPERBOUND,LOWERBOUND};
/* ---------------------------------------------------------------------- */

FixSwellRadius::FixSwellRadius(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix swell/radius command");

  step_each = force->inumeric(FLERR,arg[3]);
  tstart = force->numeric(FLERR,arg[4]);
  tend = force->numeric(FLERR,arg[5]);
  max_swelling_ratio = force->numeric(FLERR,arg[6]);

  // optional args

  iregion = -1;
  idregion = NULL;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix swell/radius command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix swell/radius does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix swell/radius command");
  }
}

/* ---------------------------------------------------------------------- */

FixSwellRadius::~FixSwellRadius()
{
  delete [] idregion;
}

/* ---------------------------------------------------------------------- */

int FixSwellRadius::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSwellRadius::init()
{
    // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix setvel/ind does not exist");
  }
}

/* ---------------------------------------------------------------------- */

void FixSwellRadius::post_integrate()
{
  int i;
  double factor;

  double **x = atom->x;
  double *radius = atom->radius;
  double *radius0 = atom->radius0;
  int *mask = atom->mask;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double ct,sr;

  if (update->ntimestep%step_each == 0){
    ct = (update->ntimestep*update->dt);
    if (ct<tstart) sr = 1.0;
    else if (ct>tend) sr = max_swelling_ratio;
    else sr = (ct-tstart)/(tend-tstart)*(max_swelling_ratio-1.0)+1.0;      
    for (i = 0; i < nlocal; i++){
      if (mask[i] & groupbit) {
        if (iregion >= 0 && !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
            continue;
        radius[i] = radius0[i]*sr;
      }
    }
  }
}


/* ---------------------------------------------------------------------- */
