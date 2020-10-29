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
# Statistics of the average Number of Bonds
# Timestep btype[i] Nbonds[i] ; i=1-Nbondtypes 
------------------------------------------------------------------------- */

#include <cmath>
#include "statistic_bond.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "molecule.h"
#include "memory.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticBond::StatisticBond(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  memory->create(nbonds,atom->nbondtypes,"statistic_bond:nbonds");
  for (int i=0;i< atom->nbondtypes;i++)
    nbonds[i] = 0.0;
  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"# Statistics of the average Number of Bonds \n");
    fprintf(out_stat,"# Timestep btype[i] Nbonds[i] ; i=1-Nbondtypes \n");
    fclose(out_stat);
  }
}

/* ---------------------------------------------------------------------- */

StatisticBond::~StatisticBond()
{
  memory->destroy(nbonds);
}

/* ---------------------------------------------------------------------- */

void StatisticBond::calc_stat()
{
  int i,nb,atom1,atom2,imol,iatom,btype;
  tagint tagprev;

  tagint *tag = atom->tag;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *mask = atom->mask;

  int *molindex = atom->molindex;
  int *molatom = atom->molatom;
  Molecule **onemols = atom->avec->onemols;

  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  int molecular = atom->molecular;

  for (atom1 = 0; atom1 < nlocal; atom1++) {
    if (!(mask[atom1] & groupbit)) continue;
    // specifying number of bonds for atom1 (nb)
    if (molecular == 1) nb = num_bond[atom1];
    else {
      if (molindex[atom1] < 0) continue;
      imol = molindex[atom1];
      iatom = molatom[atom1];
      nb = onemols[imol]->num_bond[iatom];
    }
    // specifying bond type and atom2
    for (i = 0; i < nb; i++) {
      if (molecular == 1) {
        btype = bond_type[atom1][i];
        atom2 = atom->map(bond_atom[atom1][i]);
      } else {
        tagprev = tag[atom1] - iatom - 1;
        btype = atom->map(onemols[imol]->bond_type[iatom][i]);
        atom2 = atom->map(onemols[imol]->bond_atom[iatom][i]+tagprev);
      }
      // check up
      if (atom2 < 0 || !(mask[atom2] & groupbit)) continue;
      if (newton_bond == 0 && tag[atom1] > tag[atom2]) continue;
      if (btype == 0) continue;
      nbonds[btype-1]++;
    }
  }

  num_step++;
}

/* ---------------------------------------------------------------------- */

void StatisticBond:: write_stat(bigint step)
{
  int nbtmp = atom->nbondtypes;
  double *ntmp, *tmp;

  memory->create(ntmp,nbtmp,"statistic_bond:ntmp");
  memory->create(tmp,nbtmp,"statistic_bond:tmp");

  double dum = 1.0/num_step;
  for (int i=0; i<nbtmp; i++){
    tmp[i] = nbonds[i]*dum;
    ntmp[i] = 0.0;
  }
  
  MPI_Reduce(tmp,ntmp,nbtmp,MPI_DOUBLE,MPI_SUM,0,world);
  
  for (int i=0;i< atom->nbondtypes;i++)
    nbonds[i] = 0.0;
  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    out_stat=fopen(f_name,"a");
    fprintf(out_stat,"%8lu", step);
    for (int i=0; i<nbtmp; i++){
      fprintf(out_stat,"\t%4.2f",ntmp[i]);
    }
    fprintf(out_stat,"\n"); 
    fclose(out_stat);
  }
  memory->destroy(ntmp);
  memory->destroy(tmp);    
}

/* ---------------------------------------------------------------------- */
