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
#include "fix_dump_qmatrix.h"
#include "atom.h"
#include "angle.h"
#include "bond.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
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

FixDumpQMatrix::FixDumpQMatrix(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), fp(NULL)
{
  if (narg < 5) error->all(FLERR,"Illegal fix setconc/ind command");
  MPI_Comm_rank(world,&me);
  step_each = force->inumeric(FLERR,arg[3]);
  if (me == 0){
    fp = fopen(arg[4],"w");
    if (fp == NULL) {
        char str[128];
        snprintf(str,128,"Cannot open fix dump/qmatrix file %s",arg[4]);
        error->one(FLERR,str);
      }
  }
  actuation_type = 1;
  if (narg > 5){
    if (strcmp(arg[5],"angle") == 0) actuation_type = 0;
    else {
      if (strcmp(arg[5],"bond") == 0) actuation_type = 1;
    }
  }
}

/* ---------------------------------------------------------------------- */

FixDumpQMatrix::~FixDumpQMatrix()
{
  if (fp && me == 0 ) 
   fclose(fp);
}

/* ---------------------------------------------------------------------- */

int FixDumpQMatrix::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDumpQMatrix::init() {}

/* ---------------------------------------------------------------------- */

void FixDumpQMatrix::post_integrate()
{
  if (update->ntimestep%step_each == 0 && me == 0 && fp) {
    if (actuation_type == 0)
      force->angle->write_Qmatrix(fp);
    else force->bond->write_Qmatrix(fp);
    fflush(fp);
  }
}

/* ---------------------------------------------------------------------- */
