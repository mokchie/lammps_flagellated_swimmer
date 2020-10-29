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

#include <cstdlib>
#include <cstring>
#include "read_bonds.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "comm.h"
#include "group.h"
#include "special.h"
#include "error.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ReadBonds::ReadBonds(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

void ReadBonds::command(int narg, char **arg)
{
  if (domain->box_exist == 0)
    error->all(FLERR,"read_bonds command before simulation box is defined");
  if (atom->tag_enable == 0)
    error->all(FLERR,"Cannot use read_bonds unless atoms have IDs");
  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use read_bonds with non-molecular system");

  if (narg != 1) error->all(FLERR,"Illegal read_bonds command");

  if (!force->newton_bond)
    error->all(FLERR,"read_bonds is compatible with newton_bond only.");

  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  double **bond_length = atom->bond_length;

  int i,j,k,dumint;
  int *btype;
  tagint *atom1, *atom2;
  double *blength;
  bigint nbonds_new;

  if (!comm->me) {
    if (screen) fprintf(screen,"  reading new bonds ...\n");
    if (logfile) fprintf(logfile,"  reading new bonds ...\n");
    f_bond.open (arg[0]);
    getline(f_bond, line);	// read the first line (ITEM: TIMESTEP)
    getline(f_bond, line);
    getline(f_bond, line);	// read line (ITEM: NUMBER OF ENTRIES)
    getline(f_bond, line);
    std::stringstream(line) >> nbonds_new;
    getline(f_bond, line);	// read line (ITEM: BOX BOUNDS ?? ?? ??)
    getline(f_bond, line);
    getline(f_bond, line);
    getline(f_bond, line);
    getline(f_bond, line);	// read line (ITEM: ENTRIES index c_b[1] c_b[2] c_b[3])
  }

  MPI_Bcast(&nbonds_new, 1, MPI_LMP_BIGINT, 0, world);

  memory->create(atom1,nbonds_new,"read_bonds:atom1");
  memory->create(atom2,nbonds_new,"read_bonds:atom2");
  memory->create(btype,nbonds_new,"read_bonds:btype");
  memory->create(blength,nbonds_new,"read_bonds:blength");

  if (!comm->me) {
    for (i=0; i<nbonds_new; i++) {
      getline(f_bond, line);
      std::stringstream(line) >> dumint >> atom1[i] >> atom2[i] >> btype[i];
    }
    f_bond.close();
  }

  MPI_Bcast(atom1, nbonds_new, MPI_LMP_TAGINT, 0, world);
  MPI_Bcast(atom2, nbonds_new, MPI_LMP_TAGINT, 0, world);
  MPI_Bcast(btype, nbonds_new, MPI_INT, 0, world);

  if (!comm->me) {
    if (screen) fprintf(screen,"  creating new bonds ...\n");
    if (logfile) fprintf(logfile,"  creating new bonds ...\n");
  }

  // find bond lengths
  for (j = 0; j < nbonds_new; j++)
    for (i = 0; i < nlocal; i++)
      if (tag[i] == atom1[j]) {
        for (k = 0; k < num_bond[i]; k++)
          if (bond_atom[i][k] == atom2[j]) {blength[j] = bond_length[i][k]; break;}
        break;
      }

  // reset all bonds
  for (i = 0; i < nlocal; i++)
    num_bond[i] = 0;

  // create bonds
  for (j = 0; j < nbonds_new; j++)
    for (i = 0; i < nlocal; i++)
      if (tag[i] == atom1[j]) {
        bond_atom[i][num_bond[i]] = atom2[j];
        bond_type[i][num_bond[i]] = btype[j];
        if (atom->individual) bond_length[i][num_bond[i]] = blength[j];
        num_bond[i]++;
        break;
      }

  atom->nbonds = nbonds_new;

  memory->destroy(atom1);
  memory->destroy(atom2);
  memory->destroy(btype);

  if (!comm->me) {
    if (screen) fprintf(screen,"  created new bonds ...\n");
    if (logfile) fprintf(logfile,"  created new bonds ...\n");
  }

}
