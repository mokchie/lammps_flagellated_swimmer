/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   Contributing author: Chaojie Mo
------------------------------------------------------------------------- */

#include <cstring>
#include <cstdlib>
#include "fix_setvel_rand_depth.h"
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

enum{NONE,CONSTANT};

/* ---------------------------------------------------------------------- */

FixSetVelRandDepth::FixSetVelRandDepth(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 12) error->all(FLERR,"Illegal fix setvel command");

  ind_const = force->inumeric(FLERR,arg[3]);

  if(strcmp(arg[4],"NULL") == 0) xstyle=NONE;
  else{
    xvalue = force->numeric(FLERR,arg[4]);
    xstyle=CONSTANT;
  }

  if(strcmp(arg[5],"NULL") == 0) ystyle=NONE;
  else{
    yvalue = force->numeric(FLERR,arg[5]);
    ystyle=CONSTANT;
  }  

  if(strcmp(arg[6],"NULL") == 0) zstyle=NONE;
  else{
    zvalue = force->numeric(FLERR,arg[6]);
    zstyle=CONSTANT;
  }    

  step_each = force->inumeric(FLERR,arg[7]);
  int seed = force->inumeric(FLERR,arg[8]);
  if(!(strcmp(arg[9],"x")==0 ||
       strcmp(arg[9], "y")==0 ||
       strcmp(arg[9], "z")==0))
    error->all(FLERR,"Incorrect fix setvel/rand/depth argument values");
  else{
    direction = arg[9][0];
  }
  l1 = force->numeric(FLERR,arg[10]);
  l2 = force->numeric(FLERR,arg[11]);
  // optional args

  random = NULL;
  iregion = -1;
  idregion = NULL;

  if (ind_const == 0)
    random = new RanPark(lmp,seed + comm->me);

  int iarg = 12;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix setvel/rand/depth command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix setvel/rand/depth does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix setvel/rand/depth command");
  }
}

/* ---------------------------------------------------------------------- */

FixSetVelRandDepth::~FixSetVelRandDepth()
{
  delete [] idregion;
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixSetVelRandDepth::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSetVelRandDepth::init()
{
    // set index and check validity of region
  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix setvel does not exist");
  }
  int *fixfine = atom->fixfine;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  int i;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      fixfine[i]=0;
    }
}

/* ---------------------------------------------------------------------- */

void FixSetVelRandDepth::post_integrate()
{
  int i;
  double factor;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *fixfine = atom->fixfine;
  int *mask = atom->mask;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double randi;

  if (update->ntimestep%step_each == 0){

    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (iregion >= 0 &&
            !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
          continue;
        if(!fixfine[i]){
          randi = (abs((long long) (cos(atom->tag[i])*1e9))%980)/1000.0+0.02;
          // get a pseudo random number determined by the global ID of the particle
          if((direction=='x' && (x[i][0]-l1)*(x[i][0]-l1-randi*(l2-l1))>0) ||
             (direction=='y' && (x[i][1]-l1)*(x[i][1]-l1-randi*(l2-l1))>0) ||
             (direction=='z' && (x[i][2]-l1)*(x[i][2]-l1-randi*(l2-l1))>0))
            continue;
          else
            fixfine[i] = 1;
        }
        if (ind_const) {
          if(xstyle==CONSTANT) v[i][0] = xvalue;
          if(ystyle==CONSTANT) v[i][1] = yvalue;
          if(zstyle==CONSTANT) v[i][2] = zvalue;
        } else {
          factor = sqrt(xvalue/mass[type[i]]);
          if(xstyle==CONSTANT) v[i][0] = factor*random->gaussian();
          if(ystyle==CONSTANT) v[i][1] = factor*random->gaussian();
          if(zstyle==CONSTANT) v[i][2] = factor*random->gaussian();
        }
    }
  }
}
void FixSetVelRandDepth::post_force(int /*vflag*/){
  int i;
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *fixfine = atom->fixfine;
  if (update->ntimestep%step_each == 0){

    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        //if (iregion >= 0 &&
        //    !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
        //  continue;
        if(fixfine[i]){
          if(xstyle==CONSTANT && direction=='x') f[i][0] = 0.0;
          if(ystyle==CONSTANT && direction=='y') f[i][1] = 0.0;
          if(zstyle==CONSTANT && direction=='z') f[i][2] = 0.0;
        }
    }
  }  
}
/* ---------------------------------------------------------------------- */
