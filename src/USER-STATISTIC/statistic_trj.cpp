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
# Trajectory statistics
# timestep	mol	atoms_in_mol	x y z	Rsx Rsy Rsz Rg
------------------------------------------------------------------------- */
#include <cmath>
#include "statistic_trj.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "memory.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticTRJ::StatisticTRJ(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{

  if (!comm->me){
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"# Trajectory statistics \n");
    fprintf(out_stat,"# timestep	mol atoms_in_mol	x y z	Rsx Rsy Rsz Rg ");
    fclose(out_stat);
  }

}

/* ---------------------------------------------------------------------- */

StatisticTRJ::~StatisticTRJ()
{
  // nothing to destroy
}

/* ---------------------------------------------------------------------- */

void StatisticTRJ::calc_stat()
{
  // statistic trj does not calculate and average over time intervals 
}

/* ---------------------------------------------------------------------- */

void StatisticTRJ:: write_stat(bigint step)
{
  int i;
  
  double **x = atom->x;
  tagint *molecule = atom->molecule;
  imageint *image = atom->image;
  int *mol_type = atom->mol_type;
  tagint *mol_size = atom->mol_size;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint n_mol_max = atom->n_mol_max;
    
  double unwrap[3];
  
  int *mol_flag_tmp;
  memory->create(mol_flag,n_mol_max,"statistic_trj:mol_flag");

  memory->create(xcm,n_mol_max,"statistic_trj:xcm");
  memory->create(ycm,n_mol_max,"statistic_trj:ycm");
  memory->create(zcm,n_mol_max,"statistic_trj:zcm");
  memory->create(xcm_tmp,n_mol_max,"statistic_trj:xcm_tmp");
  memory->create(ycm_tmp,n_mol_max,"statistic_trj:ycm_tmp");
  memory->create(zcm_tmp,n_mol_max,"statistic_trj:zcm_tmp");

  memory->create(Rg,n_mol_max,"statistic_trj:Rg");

  memory->create(dxmin,n_mol_max,"statistic_trj:dxmin");
  memory->create(dymin,n_mol_max,"statistic_trj:dymin");
  memory->create(dzmin,n_mol_max,"statistic_trj:dzmin");
  memory->create(dxmax,n_mol_max,"statistic_trj:dxmax");
  memory->create(dymax,n_mol_max,"statistic_trj:dymax");
  memory->create(dzmax,n_mol_max,"statistic_trj:dzmax");

  if (!comm->me){
    memory->create(Rsx_tmp,n_mol_max,"statistic_trj:Rsx_tmp");
    memory->create(Rsy_tmp,n_mol_max,"statistic_trj:Rsy_tmp");
    memory->create(Rsz_tmp,n_mol_max,"statistic_trj:Rsz_tmp");
    memory->create(Rg_tmp,n_mol_max,"statistic_trj:Rg_tmp");

    memory->create(dxmin_tmp,n_mol_max,"statistic_trj:dxmin_tmp");
    memory->create(dymin_tmp,n_mol_max,"statistic_trj:dymin_tmp");
    memory->create(dzmin_tmp,n_mol_max,"statistic_trj:dzmin_tmp");
    memory->create(dxmax_tmp,n_mol_max,"statistic_trj:dxmax_tmp");
    memory->create(dymax_tmp,n_mol_max,"statistic_trj:dymax_tmp");
    memory->create(dzmax_tmp,n_mol_max,"statistic_trj:dzmax_tmp");
  }

  memory->create(mol_flag_tmp,n_mol_max,"statistic_trj:mol_flag_tmp");

  for (i=0; i<n_mol_max; i++)
    mol_flag_tmp[i] = 0;

  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit)
      mol_flag_tmp[molecule[i]-1] = 1;

  MPI_Allreduce(mol_flag_tmp,mol_flag,n_mol_max,MPI_INT,MPI_MAX,world);
  memory->destroy(mol_flag_tmp);
  
  for (i=0; i<n_mol_max; i++){
    xcm[i] = ycm[i] = zcm[i] = 0.0;
    Rg[i] = 0.0;
    dxmin[i] = dxmax[i] = dymin[i] = dymax[i] = dzmin[i] = dzmax[i] = 0.0;
  }
  
  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit) {
      domain->unmap(x[i],image[i],unwrap);
      xcm[molecule[i]-1] += unwrap[0];
      ycm[molecule[i]-1] += unwrap[1];
      zcm[molecule[i]-1] += unwrap[2];
    }
  
  MPI_Allreduce(xcm,xcm_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(ycm,ycm_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(zcm,zcm_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,world);

  double dum;
  for (i=0; i<n_mol_max; i++)
    if (mol_flag[i]){
      dum = 1.0/mol_size[mol_type[i]];
      xcm[i] = xcm_tmp[i]*dum;
      ycm[i] = ycm_tmp[i]*dum;
      zcm[i] = zcm_tmp[i]*dum;
    }

  double dx, dy, dz;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - xcm[molecule[i]-1];
      dy = unwrap[1] - ycm[molecule[i]-1];
      dz = unwrap[2] - zcm[molecule[i]-1];
      Rg[molecule[i]-1] += dx*dx + dy*dy + dz*dz;
      if (dxmin[molecule[i]-1] > dx) dxmin[molecule[i]-1] = dx;
      if (dxmax[molecule[i]-1] < dx) dxmax[molecule[i]-1] = dx;
      if (dymin[molecule[i]-1] > dy) dymin[molecule[i]-1] = dy;
      if (dymax[molecule[i]-1] < dy) dymax[molecule[i]-1] = dy;
      if (dzmin[molecule[i]-1] > dz) dzmin[molecule[i]-1] = dz;
      if (dzmax[molecule[i]-1] < dz) dzmax[molecule[i]-1] = dz;
    }

  MPI_Reduce(Rg,Rg_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);

  MPI_Reduce(dxmin,dxmin_tmp,n_mol_max,MPI_DOUBLE,MPI_MIN,0,world);
  MPI_Reduce(dxmax,dxmax_tmp,n_mol_max,MPI_DOUBLE,MPI_MAX,0,world);
  MPI_Reduce(dymin,dymin_tmp,n_mol_max,MPI_DOUBLE,MPI_MIN,0,world);
  MPI_Reduce(dymax,dymax_tmp,n_mol_max,MPI_DOUBLE,MPI_MAX,0,world);
  MPI_Reduce(dzmin,dzmin_tmp,n_mol_max,MPI_DOUBLE,MPI_MIN,0,world);
  MPI_Reduce(dzmax,dzmax_tmp,n_mol_max,MPI_DOUBLE,MPI_MAX,0,world);

  if (!(comm->me)){
    for (i=0; i<n_mol_max; i++)
      if (mol_flag[i]){
        Rg_tmp[i] = sqrt(Rg_tmp[i]/mol_size[mol_type[i]]);
        Rsx_tmp[i] = dxmax_tmp[i] - dxmin_tmp[i];
        Rsy_tmp[i] = dymax_tmp[i] - dymin_tmp[i];
        Rsz_tmp[i] = dzmax_tmp[i] - dzmin_tmp[i];
      }

    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    out_stat=fopen(f_name,"a");
    for (i=0; i<n_mol_max; i++)
      if (mol_flag[i]){
        fprintf(out_stat,"\n%8lu", step);
        fprintf(out_stat,"\t%3lu", i+1);
        fprintf(out_stat,"\t%4lu", mol_size[mol_type[i]]);
        fprintf(out_stat,"\t%4.3f",xcm[i]);
        fprintf(out_stat,"\t%4.3f",ycm[i]);
        fprintf(out_stat,"\t%4.3f",zcm[i]);
        fprintf(out_stat,"\t%4.3f",Rsx_tmp[i]);
        fprintf(out_stat,"\t%4.3f",Rsy_tmp[i]);
        fprintf(out_stat,"\t%4.3f",Rsz_tmp[i]);
        fprintf(out_stat,"\t%4.3f",Rg_tmp[i]);
      }
    fclose(out_stat);

    memory->destroy(Rsx_tmp);
    memory->destroy(Rsy_tmp);
    memory->destroy(Rsz_tmp);
    memory->destroy(Rg_tmp);

    memory->destroy(dxmin_tmp);
    memory->destroy(dymin_tmp);
    memory->destroy(dzmin_tmp);
    memory->destroy(dxmax_tmp);
    memory->destroy(dymax_tmp);
    memory->destroy(dzmax_tmp);
  }

  memory->destroy(xcm);
  memory->destroy(ycm);
  memory->destroy(zcm);
  memory->destroy(Rg);
  memory->destroy(dxmin);
  memory->destroy(dymin);
  memory->destroy(dzmin);
  memory->destroy(dxmax);
  memory->destroy(dymax);
  memory->destroy(dzmax);
  memory->destroy(mol_flag);

}

/* ---------------------------------------------------------------------- */
