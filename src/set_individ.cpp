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
#include <cstdlib>
#include <cstring>
#include "set_individ.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "force.h"
#include "error.h"

using namespace LAMMPS_NS;

enum{BOND,ANGLE,DIHEDRAL};

/* ---------------------------------------------------------------------- */

void Set_Individ::command(int narg, char **arg)
{
  int allcount;

  if (domain->box_exist == 0)
    error->all(FLERR,"Set/individ command before simulation box is defined");
  if (atom->natoms == 0)
    error->all(FLERR,"Set/individ command with no atoms existing");
  if (narg < 3) error->all(FLERR,"Illegal set/individ command");
  if (!atom->individual) error->all(FLERR,"Usage of set/individ is not allowed for not atom individual styles");

  count = 0;
  // set parameters 

  if (strcmp(arg[0],"bond") == 0) style = BOND;
  else if (strcmp(arg[0],"angle") == 0) style = ANGLE;
  else if (strcmp(arg[0],"dihedral") == 0) style = DIHEDRAL;
  else error->all(FLERR,"Illegal set/individ command");

  type = force->inumeric(FLERR,arg[1]);
  value = force->numeric(FLERR,arg[2]);
  set(style);

  MPI_Allreduce(&count,&allcount,1,MPI_INT,MPI_SUM,world);

  if (comm->me == 0) {
    if (screen) fprintf(screen,"  %d individual settings made for %s\n",
                        allcount,arg[0]);
    if (logfile) fprintf(logfile,"  %d individual settings made for %s\n",
                         allcount,arg[0]);
  }  
}


/* ----------------------------------------------------------------------
   set individual properties
------------------------------------------------------------------------- */

void Set_Individ::set(int keyword)
{
  int i,m;
  int nlocal = atom->nlocal;

  if (keyword == BOND) {
    for (i = 0; i < nlocal; i++)
      for (m = 0; m < atom->num_bond[i]; m++)
        if (atom->bond_type[i][m] == type) {
          atom->bond_length[i][m] = value;
          count++;
        }
  }

  if (keyword == ANGLE) {
    for (i = 0; i < nlocal; i++)
      for (m = 0; m < atom->num_angle[i]; m++)
        if (atom->angle_type[i][m] == type) {
          atom->angle_area[i][m] = value;
          count++;
        }
  }

  if (keyword == DIHEDRAL) {
    for (i = 0; i < nlocal; i++)
      for (m = 0; m < atom->num_dihedral[i]; m++)
        if (atom->dihedral_type[i][m] == type) {
          atom->dihedral_angle[i][m] = value;
          count++;
        }
  }
}

/* ---------------------------------------------------------------------- */

