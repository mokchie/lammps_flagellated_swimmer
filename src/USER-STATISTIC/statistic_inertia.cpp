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
# Inertia statistics
# timestep	mol atoms_in_mol	mass	x y z	px py pz	Ixx Iyy Izz Ixy Iyz Ixz	Lx Ly Lz 
------------------------------------------------------------------------- */
#include <cmath>
#include "statistic_inertia.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticINERTIA::StatisticINERTIA(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  
  if (!comm->me){
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"# Inertia statistics \n");
    fprintf(out_stat,"# timestep	mol atoms_in_mol	mass	x y z	px py pz	Ixx Iyy Izz Ixy Iyz Ixz	Lx Ly Lz ");
    fclose(out_stat);
  }
  
}

/* ---------------------------------------------------------------------- */

StatisticINERTIA::~StatisticINERTIA()
{
  // nothing to destroy
}

/* ---------------------------------------------------------------------- */

void StatisticINERTIA::calc_stat()
{
  // statistic inertia does not calculate and average over time intervals 
}

/* ---------------------------------------------------------------------- */

void StatisticINERTIA:: write_stat(bigint step)
{
  int i;
  
  double **x = atom->x;
  double **v = atom->v;
  int *type = atom->type;
  double *mass = atom->mass;
  tagint *molecule = atom->molecule;
  imageint *image = atom->image;
  int *mol_type = atom->mol_type;
  tagint *mol_size = atom->mol_size;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint n_mol_max = atom->n_mol_max;
  
  double unwrap[3];
  
  int *mol_flag_tmp;
  // memory construction
  memory->create(mol_flag,n_mol_max,"statistic_inertia:mol_flag");

  memory->create(xcm,n_mol_max,"statistic_inertia:xcm");
  memory->create(ycm,n_mol_max,"statistic_inertia:ycm");
  memory->create(zcm,n_mol_max,"statistic_inertia:zcm");
  memory->create(xcm_tmp,n_mol_max,"statistic_inertia:xcm_tmp");
  memory->create(ycm_tmp,n_mol_max,"statistic_inertia:ycm_tmp");
  memory->create(zcm_tmp,n_mol_max,"statistic_inertia:zcm_tmp");
  
  memory->create(mtot,n_mol_max,"statistic_inertia:mtot");
  
  memory->create(vxcm,n_mol_max,"statistic_inertia:vxcm");
  memory->create(vycm,n_mol_max,"statistic_inertia:vycm");
  memory->create(vzcm,n_mol_max,"statistic_inertia:vzcm");
  
  memory->create(Ixx,n_mol_max,"statistic_inertia:Ixx");
  memory->create(Iyy,n_mol_max,"statistic_inertia:Iyy");
  memory->create(Izz,n_mol_max,"statistic_inertia:Izz");
  memory->create(Ixy,n_mol_max,"statistic_inertia:Ixy");
  memory->create(Iyz,n_mol_max,"statistic_inertia:Iyz");
  memory->create(Ixz,n_mol_max,"statistic_inertia:Ixz");
  
  memory->create(Lx,n_mol_max,"statistic_inertia:Lx");
  memory->create(Ly,n_mol_max,"statistic_inertia:Ly");
  memory->create(Lz,n_mol_max,"statistic_inertia:Lz");
  
  if (!comm->me){
    memory->create(mtot_tmp,n_mol_max,"statistic_inertia:mtot_tmp");

    memory->create(vxcm_tmp,n_mol_max,"statistic_inertia:vxcm_tmp");
    memory->create(vycm_tmp,n_mol_max,"statistic_inertia:vycm_tmp");
    memory->create(vzcm_tmp,n_mol_max,"statistic_inertia:vzcm_tmp");

    memory->create(Ixx_tmp,n_mol_max,"statistic_inertia:Ixx_tmp");
    memory->create(Iyy_tmp,n_mol_max,"statistic_inertia:Iyy_tmp");
    memory->create(Izz_tmp,n_mol_max,"statistic_inertia:Izz_tmp");
    memory->create(Ixy_tmp,n_mol_max,"statistic_inertia:Ixy_tmp");
    memory->create(Iyz_tmp,n_mol_max,"statistic_inertia:Iyz_tmp");
    memory->create(Ixz_tmp,n_mol_max,"statistic_inertia:Ixz_tmp");

    memory->create(Lx_tmp,n_mol_max,"statistic_inertia:Lx_tmp");
    memory->create(Ly_tmp,n_mol_max,"statistic_inertia:Ly_tmp");
    memory->create(Lz_tmp,n_mol_max,"statistic_inertia:Lz_tmp");
  }
  
  memory->create(mol_flag_tmp,n_mol_max,"statistic_inertia:mol_flag_tmp");
  
  for (i=0; i<n_mol_max; i++)
    mol_flag_tmp[i] = 0;
  
  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit)
    mol_flag_tmp[molecule[i]-1] = 1;
  
  MPI_Allreduce(mol_flag_tmp,mol_flag,n_mol_max,MPI_INT,MPI_MAX,world);
  memory->destroy(mol_flag_tmp);
  
  for (i=0; i<n_mol_max; i++){
    xcm[i] = ycm[i] = zcm[i] = 0.0;
    vxcm[i] = vycm[i] = vzcm[i] = 0.0;
    Ixx[i] = Iyy[i] = Izz[i] = Ixy[i] = Iyz[i] = Ixz[i] = 0.0;
    Lx[i] = Ly[i] = Lz[i] = 0.0;
    mtot[i] = 0.0;
  }
  
  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit) {
      domain->unmap(x[i],image[i],unwrap);
      xcm[molecule[i]-1] += unwrap[0];
      ycm[molecule[i]-1] += unwrap[1];
      zcm[molecule[i]-1] += unwrap[2];
      mtot[molecule[i]-1] += mass[type[i]];
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
  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit) {
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - xcm[molecule[i]-1];
      dy = unwrap[1] - ycm[molecule[i]-1];
      dz = unwrap[2] - zcm[molecule[i]-1];

      vxcm[molecule[i]-1] += mass[type[i]]*v[i][0];
      vycm[molecule[i]-1] += mass[type[i]]*v[i][1];
      vzcm[molecule[i]-1] += mass[type[i]]*v[i][2];

      Ixx[molecule[i]-1] += mass[type[i]]*(dy*dy + dz*dz);
      Iyy[molecule[i]-1] += mass[type[i]]*(dx*dx + dz*dz);
      Izz[molecule[i]-1] += mass[type[i]]*(dx*dx + dy*dy);
      Ixy[molecule[i]-1] -= mass[type[i]]*dx*dy;
      Iyz[molecule[i]-1] -= mass[type[i]]*dy*dz;
      Ixz[molecule[i]-1] -= mass[type[i]]*dx*dz;

      Lx[molecule[i]-1] += mass[type[i]]*(dy*v[i][2] - dz*v[i][1]);
      Ly[molecule[i]-1] += mass[type[i]]*(dz*v[i][0] - dx*v[i][2]);
      Lz[molecule[i]-1] += mass[type[i]]*(dx*v[i][1] - dy*v[i][0]);
    }
  
  MPI_Reduce(mtot,mtot_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);

  MPI_Reduce(vxcm,vxcm_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(vycm,vycm_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(vzcm,vzcm_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);

  MPI_Reduce(Ixx,Ixx_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(Iyy,Iyy_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(Izz,Izz_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(Ixy,Ixy_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(Iyz,Iyz_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(Ixz,Ixz_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);

  MPI_Reduce(Lx,Lx_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(Ly,Ly_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(Lz,Lz_tmp,n_mol_max,MPI_DOUBLE,MPI_SUM,0,world);

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    out_stat=fopen(f_name,"a");
    for (int i=0; i<n_mol_max; i++)
      if (mol_flag[i]){
        fprintf(out_stat,"\n%8lu", step);
        fprintf(out_stat,"\t%3lu", i+1);
        fprintf(out_stat,"\t%4lu", mol_size[mol_type[i]]);
 
        fprintf(out_stat,"\t%4.0f",mtot_tmp[i]);

        fprintf(out_stat,"\t%4.2f",xcm[i]);
        fprintf(out_stat,"\t%4.2f",ycm[i]);
        fprintf(out_stat,"\t%4.2f",zcm[i]);

        fprintf(out_stat,"\t%3.2f",vxcm_tmp[i]);
        fprintf(out_stat,"\t%3.2f",vycm_tmp[i]);
        fprintf(out_stat,"\t%3.2f",vzcm_tmp[i]);

        fprintf(out_stat,"\t%4.2f",Ixx_tmp[i]);
        fprintf(out_stat,"\t%4.2f",Iyy_tmp[i]);
        fprintf(out_stat,"\t%4.2f",Izz_tmp[i]);
        fprintf(out_stat,"\t%2.2f",Ixy_tmp[i]);
        fprintf(out_stat,"\t%2.2f",Iyz_tmp[i]);
        fprintf(out_stat,"\t%2.2f",Ixz_tmp[i]);

        fprintf(out_stat,"\t%4.2f",Lx_tmp[i]);
        fprintf(out_stat,"\t%4.2f",Ly_tmp[i]);
        fprintf(out_stat,"\t%4.2f",Lz_tmp[i]);
      }
    fclose(out_stat);
  }

  memory->destroy(mol_flag);

  memory->destroy(xcm);
  memory->destroy(ycm);
  memory->destroy(zcm);
  memory->destroy(xcm_tmp);
  memory->destroy(ycm_tmp);
  memory->destroy(zcm_tmp);

  memory->destroy(mtot);

  memory->destroy(vxcm);
  memory->destroy(vycm);
  memory->destroy(vzcm);

  memory->destroy(Ixx);
  memory->destroy(Iyy);
  memory->destroy(Izz);
  memory->destroy(Ixy);
  memory->destroy(Iyz);
  memory->destroy(Ixz);

  memory->destroy(Lx);
  memory->destroy(Ly);
  memory->destroy(Lz);

  if (!comm->me){

    memory->destroy(mtot_tmp);

    memory->destroy(vxcm_tmp);
    memory->destroy(vycm_tmp);
    memory->destroy(vzcm_tmp);

    memory->destroy(Ixx_tmp);
    memory->destroy(Iyy_tmp);
    memory->destroy(Izz_tmp);
    memory->destroy(Ixy_tmp);
    memory->destroy(Iyz_tmp);
    memory->destroy(Ixz_tmp);

    memory->destroy(Lx_tmp);
    memory->destroy(Ly_tmp);
    memory->destroy(Lz_tmp);
  }

}

/* ---------------------------------------------------------------------- */
