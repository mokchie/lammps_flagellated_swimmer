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

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "fix_e.h"
#include "atom.h"
#include "domain.h"
#include "region.h"
#include "group.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CONSTFLUX,TOTFLUX,TEMP};

/* ---------------------------------------------------------------------- */

FixE::FixE(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix e command");

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix e command");
  if (strcmp(arg[4],"constflux") == 0) 
    mstyle = CONSTFLUX;
  else if (strcmp(arg[4],"totflux") == 0)
    mstyle = TOTFLUX;
  else if (strcmp(arg[4],"temp") == 0)
    mstyle = TEMP;
  else error->all(FLERR,"Illegal fix e command");
    
  en0 = force->numeric(FLERR,arg[5]);

  // optional args

  iregion = -1;
  idregion = NULL;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix e command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix e does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix e command");
  }
}

/* ---------------------------------------------------------------------- */

FixE::~FixE()
{
  delete [] idregion;
}

/* ---------------------------------------------------------------------- */

int FixE::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixE::init()
{
  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix e does not exist");
  }

  c_v = atom->c_v;
}

/* ---------------------------------------------------------------------- */

void FixE::end_of_step()
{
  int i, j, nt, nt1;
  double temph;
  double **x = atom->x;
  double *e = atom->e;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (mstyle == CONSTFLUX){ 
    temph = en0/c_v;
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (iregion >= 0 &&
            !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
          continue;
        e[i] += temph; 
      }
  } else if (mstyle == TOTFLUX){
    nt = 0;
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (iregion >= 0 &&
            !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
          continue;
        nt++; 
      }

    nt1 = 0; 
    temph = 0.0;
    if (comm->nprocs > 1)
      MPI_Allreduce(&nt, &nt1, 1, MPI_INT, MPI_SUM, world);
    else
      nt1 = nt;
    if (nt1 > 0)
      temph = en0/nt1/c_v;

    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (iregion >= 0 &&
            !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
          continue;
        e[i] += temph; 
      }    
  } else if (mstyle == TEMP){
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (iregion >= 0 &&
            !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
          continue;
        e[i] = en0; 
      }
  }
}

/* ---------------------------------------------------------------------- */

