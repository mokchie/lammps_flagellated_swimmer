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
#include "fix_setvel_var.h"
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
enum{NONE,CONSTANT,VARIABLE};
enum{EQUAL,ATOM};
/* ---------------------------------------------------------------------- */

FixSetVelVar::FixSetVelVar(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), xvarstr(NULL), yvarstr(NULL), zvarstr(NULL)
{
  if (narg < 9) error->all(FLERR,"Illegal fix setvel/var command");

  ind_const = force->inumeric(FLERR,arg[3]);
  if (strcmp(arg[4],"NULL") == 0)
    xstyle = NONE;
  else if (strstr(arg[4],"v_")==arg[4]){
    xstyle = VARIABLE;
    int n = strlen(&arg[4][2]) + 1;
    xvarstr = new char[n];
    strcpy(xvarstr,&arg[4][2]);
  }
  else{
    xstyle = CONSTANT;
    xvalue = force->numeric(FLERR,arg[4]);
  }

 if (strcmp(arg[5],"NULL") == 0)
    ystyle = NONE;
  else if (strstr(arg[5],"v_")==arg[5]){
    ystyle = VARIABLE;
    int n = strlen(&arg[5][2]) + 1;
    yvarstr = new char[n];
    strcpy(yvarstr,&arg[5][2]);
  }  
  else{
    ystyle = CONSTANT;
    yvalue = force->numeric(FLERR,arg[5]);
  }
 if (strcmp(arg[6],"NULL") == 0)
    zstyle = NONE;
  else if (strstr(arg[6],"v_")==arg[6]){
    zstyle = VARIABLE;
    int n = strlen(&arg[6][2]) + 1;
    zvarstr = new char[n];
    strcpy(zvarstr,&arg[6][2]);
  }    
  else{
    zstyle = CONSTANT;
    zvalue = force->numeric(FLERR,arg[6]);
  }

  step_each = force->inumeric(FLERR,arg[7]);
  int seed = force->inumeric(FLERR,arg[8]);

  // optional args

  random = NULL;
  iregion = -1;
  idregion = NULL;

  if (ind_const == 0)
    random = new RanPark(lmp,seed + comm->me);

  int iarg = 9;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix setvel/var command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix setvel/var does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix setvel/var command");
  }
}

/* ---------------------------------------------------------------------- */

FixSetVelVar::~FixSetVelVar()
{
  delete [] idregion;
  delete random;
  delete [] xvarstr;
  delete [] yvarstr;
  delete [] zvarstr;
}

/* ---------------------------------------------------------------------- */

int FixSetVelVar::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSetVelVar::init()
{
    // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix setvel/var does not exist");
  }
  if (xstyle==VARIABLE){
    if (xvarstr){
      xvar = input->variable->find(xvarstr);
      if (xvar < 0) error->all(FLERR,"Variable name for fix setvel/var does not exist");
      if (input->variable->equalstyle(xvar)) xvarstyle = EQUAL;
      else error->all(FLERR,"Variable for fix setvel/var is invalid style");
    }
  }
  if (ystyle==VARIABLE){
    if (yvarstr){
      yvar = input->variable->find(yvarstr);
      if (yvar < 0) error->all(FLERR,"Variable name for fix setvel/var does not exist");
      if (input->variable->equalstyle(yvar)) yvarstyle = EQUAL;
      else error->all(FLERR,"Variable for fix setvel/var is invalid style");
    }
  }  
  if (zstyle==VARIABLE){
    if (zvarstr){
      zvar = input->variable->find(zvarstr);
      if (zvar < 0) error->all(FLERR,"Variable name for fix setvel/var does not exist");
      if (input->variable->equalstyle(zvar)) zvarstyle = EQUAL;
      else error->all(FLERR,"Variable for fix setvel/var is invalid style");
    }
  }  
}

/* ---------------------------------------------------------------------- */

void FixSetVelVar::post_integrate()
{
  int i;
  double factor;

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  if (update->ntimestep%step_each == 0){
    if (ind_const) {
      for (i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (iregion >= 0 &&
              !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
            continue;
          if(xstyle==CONSTANT)
            v[i][0] = xvalue;
          if(ystyle==CONSTANT)
            v[i][1] = yvalue;
          if(zstyle==CONSTANT)
            v[i][2] = zvalue;
          if(xstyle==VARIABLE && xvarstr)
            v[i][0] = input->variable->compute_equal(xvar);
          if(ystyle==VARIABLE && yvarstr)
            v[i][1] = input->variable->compute_equal(yvar);
          if(zstyle==VARIABLE && zvarstr)
            v[i][2] = input->variable->compute_equal(zvar);
        }
    } else {
      for (i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (iregion >= 0 &&
              !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
            continue;
          if(xstyle==CONSTANT){
            factor = sqrt(xvalue/mass[type[i]]); 
            v[i][0] = factor*random->gaussian();
          }        
          if(ystyle==CONSTANT){
            factor = sqrt(yvalue/mass[type[i]]);  
            v[i][1] = factor*random->gaussian();
          }
          if(zstyle==CONSTANT){
            factor = sqrt(zvalue/mass[type[i]]);
            v[i][2] = factor*random->gaussian();
          }
	      }
    }
  }
}

/* ---------------------------------------------------------------------- */
