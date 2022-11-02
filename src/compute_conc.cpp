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
   Contributing author: Naveen Michaud-Agrawal (Johns Hopkins U)
     K-space terms added by Stan Moore (BYU)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <cstring>
#include <cmath>
#include "compute_conc.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "group.h"
#include "kspace.h"
#include "error.h"
#include "comm.h"
#include "domain.h"
#include "region.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

enum{OFF,INTER,INTRA};

/* ---------------------------------------------------------------------- */

ComputeConc::ComputeConc(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (!atom->conc_flag) error->all(FLERR,"Illegal compute conc command");
  if (narg < 4) error->all(FLERR,"Illegal compute conc command");

  scalar_flag = vector_flag = 1;
  size_vector = 3;
  extscalar = 1;
  extvector = 1;
  nc = force->inumeric(FLERR,arg[3]);
  if (nc>atom->individual) error->all(FLERR,"Illegal compute conc command");
  iregion = -1;
  idregion = NULL;

  int iarg = 4;
  while (iarg < narg) {
    if(strcmp(arg[iarg],"region") == 0){
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute conc command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for conc does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2; 
    } else error->all(FLERR,"Illegal compute conc command");
  }

  vector = new double[3];
}

/* ---------------------------------------------------------------------- */

ComputeConc::~ComputeConc()
{
  delete [] vector;
  delete [] idregion;
}

/* ---------------------------------------------------------------------- */

void ComputeConc::init()
{
  //check validity of the region ID
  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for compute conc does not exist");
  }
}

/* ---------------------------------------------------------------------- */

double ComputeConc::compute_scalar()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  vector[0] = vector[1] = vector[2] = 0.0;
  atoms_summation();

  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeConc::compute_vector()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  vector[0] = vector[1] = vector[2] = 0.0;

  atoms_summation();
}

/* ---------------------------------------------------------------------- */

void ComputeConc::atoms_summation()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double **conc = atom->conc;
  double one;

  for (i = 0; i < nlocal; i++){
    if (!(mask[i] & groupbit)) continue;
    if (iregion>=0 && !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2])) continue;
    one += conc[i][nc-1];
  }

  double all;
  MPI_Allreduce(&one,&all,1,MPI_DOUBLE,MPI_SUM,world);
  scalar += all;
  vector[0] += all; vector[1] += all; vector[2] += all;
}

/* ----------------------------------------------------------------- */
