/* ----------------------------------------------------------------------
  Dmitry Fedosov - 08/12/05  accumulation of statistics
 
  LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include "statistic_corr_group.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"
#include "update.h"
#include "domain.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticCorrGroup::StatisticCorrGroup(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  st_accum = 0;
  num_step = 0;
  avg_step = 0;
  cyl_ind = 0;
  step_each = 1;

  corr_map = rot_list = NULL; 
  z_cr = corr = z_cur = NULL;
}

/* ---------------------------------------------------------------------- */

StatisticCorrGroup::~StatisticCorrGroup()
{
  if (comm->me == 0){
    memory->sfree(corr_map);
    memory->sfree(rot_list);
    memory->destroy(z_cr); 
    memory->destroy(corr);
    memory->destroy(z_cur);
  }
}

/* ---------------------------------------------------------------------- */

void StatisticCorrGroup::calc_stat()
{
  int i,j,i_b,ij;
  double cm[3],cmt[3],lgmin,lgmax,dlg;
  tagint mol;

  int *mask = atom->mask;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  if (st_accum == 0){
    time_corr = update->dt*dump_each;

    npart = 0;
    ij = 0; 
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
	ij++;

    MPI_Allreduce(&ij,&npart,1,MPI_INT,MPI_SUM,world); 

    if (comm->me == 0){  
       
      corr_map = (int *) memory->smalloc(ny*sizeof(int),"statistic_corr_group:corr_map"); 
      corr_map[0] = 0;
      corr_map[1] = 1; 
      lgmax = log(time_corr);
      lgmin = log(update->dt);      
      dlg = (lgmax - lgmin)/(ny-2);
      nwind = 2;
      for (i = 1; i <= ny-2; i++){
        j = static_cast<int>(floor(exp(lgmin+i*dlg)/update->dt + 0.5));
        if (j > corr_map[nwind-1] && j < dump_each){
          corr_map[nwind] = j;
          nwind++; 
        }
      }     
       
      if (nwind) {
        memory->create(corr,nwind,"statistic_corr_group:corr");
        memory->create(z_cur,nwind,"statistic_corr_group:z_cur");
      }
      for (i = 0; i < nwind; i++){
        corr[i] = 0.0;
        z_cur[i] = 0.0;
      }
      memory->create(z_cr,dump_each,"statistic_corr_group:z_cr");
      z_avg = 0.0;
      rot_list = (int *) memory->smalloc(nwind*sizeof(int),"statistic_corr:rot_list");   
    }
  }

  for (i = 0; i < 3; i++)
    cmt[i] = 0.0;
  
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit){
      domain->unmap(x[i],atom->image[i],cm);
      for (j = 0; j < 3; j++)
	cmt[j] += cm[j];
    }

  for (i = 0; i < 3; i++){
    cmt[i] /= npart;
    cm[i] = 0.0;
  }
  MPI_Reduce(&cmt[0],&cm[0],3,MPI_DOUBLE,MPI_SUM,0,world);

  st_accum++;
  if (st_accum > dump_each)
    st_accum = dump_each + 1;

  if (comm->me == 0){
    avg_step++;
    mol = (update->ntimestep - st_start - 1)%dump_each;
    z_cr[mol] = cm[2];
    z_avg += cm[2];

    if (st_accum > dump_each && update->ntimestep%nz == 0){
      num_step++;
      for (i = 0; i < nwind; i++)
        rot_list[i] = (mol+1+corr_map[i])%dump_each;
      i_b = rot_list[0];
      for (j = 0; j < nwind; j++){
        ij = rot_list[j];
        corr[j] += z_cr[i_b]*z_cr[ij];
        z_cur[j] += z_cr[ij]; 
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void StatisticCorrGroup::write_stat(bigint step)
{
  int i;
  double tt, zz;
  double dt = update->dt;
  double *corr_bin;
  char f_name[FILENAME_MAX];

  if (st_accum <= dump_each){
    num_step = 0; 
    return;
  }

  if (comm->me == 0){
    if (nwind)
      memory->create(corr_bin,nwind,"statistic_corr_group:corr_bin");
    for (i = 0; i < nwind; i++) 
      corr_bin[i] = 0.0;

    zz = z_avg/avg_step; 
    for (i = 0; i < nwind; i++){
      corr_bin[i] = (corr[i] - zz*(z_cur[0] + z_cur[i]))/num_step + zz*zz;
      //corr[i] = 0.0;
    }
    //z_avg = 0.0;
     
    //num_step = 0;
    //avg_step = 0;

    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    for (i = 0; i < nwind; i++){
      tt = corr_map[i]*dt;
      fprintf(out_stat,"%g %15.10lf \n", tt,corr_bin[i]);
    }
    fclose(out_stat);

    memory->destroy(corr_bin);
  }  
}

/* ---------------------------------------------------------------------- */

