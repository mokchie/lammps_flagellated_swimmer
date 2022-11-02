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
#include "fix_setconc_ind.h"
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

FixSetConcInd::FixSetConcInd(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix setconc/ind command");

  ind_const = force->inumeric(FLERR,arg[3]);
  if(ind_const < 1 || ind_const>atom->individual) error->all(FLERR,"Illegal fix setconc/ind command: ind_const must be >0 and <=atom->individual");
  if (strcmp(arg[4],"CONSTANT") == 0 || strcmp(arg[4],"constant") == 0)
    cstyle = CONSTANT;
  else if (strcmp(arg[4],"LOWERBOUND") == 0 || strcmp(arg[4],"lowerbound") == 0)
    cstyle = LOWERBOUND;
  else if (strcmp(arg[4],"UPPERBOUND") == 0 || strcmp(arg[4],"upperbound") == 0)
    cstyle = UPPERBOUND;    
  else error->all(FLERR,"Illegal fix setconc/ind command");
  cvalue = force->numeric(FLERR,arg[5]);


  step_each = force->inumeric(FLERR,arg[6]);

  // optional args

  iregion = -1;
  idregion = NULL;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix setconc/ind command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix setconc/ind does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix setconc/ind command");
  }
}

/* ---------------------------------------------------------------------- */

FixSetConcInd::~FixSetConcInd()
{
  delete [] idregion;
}

/* ---------------------------------------------------------------------- */

int FixSetConcInd::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSetConcInd::init()
{
    // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix setvel/ind does not exist");
  }
}

/* ---------------------------------------------------------------------- */

void FixSetConcInd::post_integrate()
{
  int i;
  double factor;

  double **x = atom->x;
  double **conc = atom->conc;
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
          if(cstyle==CONSTANT)
            conc[i][ind_const-1] = cvalue;
          if(cstyle==LOWERBOUND && conc[i][ind_const-1]<cvalue)
            conc[i][ind_const-1] = cvalue;
          if(cstyle==UPPERBOUND && conc[i][ind_const-1]>cvalue)
            conc[i][ind_const-1] = cvalue;          

	      }
    }
  }
}

/* ---------------------------------------------------------------------- */
