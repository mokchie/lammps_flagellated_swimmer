/* ----------------------------------------------------------------------
  Dmitry Fedosov - 08/12/05  accumulation of statistics
 
  LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

/* ----------------------------------------------------------------------
   Contributing author: Masoud Hoore (FZJ), Dmitry Fedosov (FZJ)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
# Statistics of the average Number of Types
# Timestep mol nitype njtype
------------------------------------------------------------------------- */

#include <cmath>
#include "statistic_type.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "molecule.h"
#include "memory.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticType::StatisticType(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  itype = nx;
  jtype = ny;
  
  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"# Statistics of the average Number of Types \n");
    fprintf(out_stat,"# Timestep mol nitype njtype; itype = %u, jtype = %u ", itype, jtype);
    fclose(out_stat);
  }
}

/* ---------------------------------------------------------------------- */

StatisticType::~StatisticType()
{
  // nothing to destroy
}

/* ---------------------------------------------------------------------- */

void StatisticType::calc_stat()
{
  // statistic type does not calculate and average over time intervals 
}

/* ---------------------------------------------------------------------- */

void StatisticType::write_stat(bigint step)
{
  int i;

  tagint n_mol_max = atom->n_mol_max;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int *type = atom->type;
  int *mask = atom->mask;

  int *nitype, *njtype;
  int *mol_flag;

  memory->create(nitype,n_mol_max,"statistic_type:nitype");
  memory->create(njtype,n_mol_max,"statistic_type:njtype");
  memory->create(mol_flag,n_mol_max,"statistic_type:mol_flag");

  for (i=0; i< atom->n_mol_max; i++)
    nitype[i] = njtype[i] = mol_flag[i] = 0;

  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit) {
      mol_flag[molecule[i]-1] = 1;
      if (itype == type[i]) nitype[molecule[i]-1] += 1;
      else if (jtype == type[i]) njtype[molecule[i]-1] += 1;
    }

  int *nitype_tmp, *njtype_tmp, *mol_flag_tmp;

  if (!comm->me) {
    memory->create(nitype_tmp,n_mol_max,"statistic_type:nitype_tmp");
    memory->create(njtype_tmp,n_mol_max,"statistic_type:njtype_tmp");
    memory->create(mol_flag_tmp,n_mol_max,"statistic_type:mol_flag_tmp");
  }

  MPI_Reduce(nitype,nitype_tmp,n_mol_max,MPI_INT,MPI_SUM,0,world);
  MPI_Reduce(njtype,njtype_tmp,n_mol_max,MPI_INT,MPI_SUM,0,world);
  MPI_Reduce(mol_flag,mol_flag_tmp,n_mol_max,MPI_INT,MPI_SUM,0,world);

  if (!comm->me){
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    out_stat=fopen(f_name,"a");
    for (i=0; i<n_mol_max; i++)
      if (mol_flag_tmp[i]) {
        fprintf(out_stat,"\n%8lu", step);
        fprintf(out_stat,"\t%3lu", i+1);
        fprintf(out_stat,"\t%i",nitype_tmp[i]);
        fprintf(out_stat,"\t%i",njtype_tmp[i]);
      }
    fclose(out_stat);

    memory->destroy(nitype_tmp);
    memory->destroy(njtype_tmp);
    memory->destroy(mol_flag_tmp);
  }

  memory->destroy(nitype);
  memory->destroy(njtype);
  memory->destroy(mol_flag);

}

/* ---------------------------------------------------------------------- */
