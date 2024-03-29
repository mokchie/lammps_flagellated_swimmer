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
#include "ntopo_angle_all.h"
#include "atom.h"
#include "force.h"
#include "domain.h"
#include "update.h"
#include "output.h"
#include "thermo.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define DELTA 10000

/* ---------------------------------------------------------------------- */

NTopoAngleAll::NTopoAngleAll(LAMMPS *lmp) : NTopo(lmp)
{
  allocate_angle();
}

/* ---------------------------------------------------------------------- */

void NTopoAngleAll::build()
{
  int i,m,atom1,atom2,atom3;

  int nlocal = atom->nlocal;
  int *num_angle = atom->num_angle;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  int **angle_type = atom->angle_type;
  int newton_bond = force->newton_bond;

  int individual = atom->individual;
  int lostbond = output->thermo->lostbond;
  int nmissing = 0;
  nanglelist = 0;

  double **angle_area = atom->angle_area;

  for (i = 0; i < nlocal; i++)
    for (m = 0; m < num_angle[i]; m++) {
      atom1 = atom->map(angle_atom1[i][m]);
      atom2 = atom->map(angle_atom2[i][m]);
      atom3 = atom->map(angle_atom3[i][m]);
      if (atom1 == -1 || atom2 == -1 || atom3 == -1) {
        nmissing++;
        if (lostbond == Thermo::ERROR) {
          char str[128];
          sprintf(str,"Angle atoms "
                  TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT
                  " missing on proc %d at step " BIGINT_FORMAT,
                  angle_atom1[i][m],angle_atom2[i][m],angle_atom3[i][m],
                  me,update->ntimestep);
          error->one(FLERR,str);
        }
        continue;
      }
      atom1 = domain->closest_image(i,atom1);
      atom2 = domain->closest_image(i,atom2);
      atom3 = domain->closest_image(i,atom3);
      if (newton_bond || (i <= atom1 && i <= atom2 && i <= atom3)) {
        if (nanglelist == maxangle) {
          maxangle += DELTA;
          memory->grow(anglelist,maxangle,4,"neigh_topo:anglelist");
          if (individual)
            memory->grow(anglelist_area,maxangle,"neigh_topo:anglelist_area");
        }
        anglelist[nanglelist][0] = atom1;
        anglelist[nanglelist][1] = atom2;
        anglelist[nanglelist][2] = atom3;
        anglelist[nanglelist][3] = angle_type[i][m];
        if (individual)
          anglelist_area[nanglelist] = angle_area[i][m];
        nanglelist++;
      }
    }

  if (cluster_check) angle_check();
  if (lostbond == Thermo::IGNORE) return;

  int all;
  MPI_Allreduce(&nmissing,&all,1,MPI_INT,MPI_SUM,world);
  if (all) {
    char str[128];
    sprintf(str,
            "Angle atoms missing at step " BIGINT_FORMAT,update->ntimestep);
    if (me == 0) error->warning(FLERR,str);
  }
}
