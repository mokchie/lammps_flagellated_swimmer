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
#include "statistic_corr.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"
#include "update.h"
#include "domain.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticCorr::StatisticCorr(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  st_accum = 0;
  num_step = 0;
  avg_step = 0;
  cyl_ind = 0;
  step_each = 1;

  nr_bin = corr_map = rot_list = NULL; 
  bin_list = NULL;
  z_avg = NULL;
  z_cr = corr = z_cur = NULL;
}

/* ---------------------------------------------------------------------- */

StatisticCorr::~StatisticCorr()
{
  if (comm->me == 0){
    memory->sfree(nr_bin);
    memory->sfree(corr_map);
    memory->sfree(rot_list);
    memory->sfree(z_avg);
    memory->destroy(bin_list);
    memory->destroy(z_cr); 
    memory->destroy(corr);
    memory->destroy(z_cur);
  }
}

/* ---------------------------------------------------------------------- */

void StatisticCorr::calc_stat()
{
  int i,j,k,l,i_b,ij;
  double rr,cm[3],cmt[3],lgmin,lgmax,dlg;
  tagint ii,mol;

  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  tagint *tag = atom->tag;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  if (st_accum == 0){
    time_corr = update->dt*dump_each;
    size_bin = xlo/nx;

    id_min = 99999999;
    ii = 99999999;
    npart = 0;
    g_num = 0; 
    for(j = 0; j < 3; j++) cmt[j] = 0.0;
    for (i = 0; i < nlocal; i++){
      mol = molecule[i];
      if (mol && mask[i] & groupbit){
        if (tag[i] < ii) ii = tag[i];
        domain->unmap(x[i],atom->image[i],cm);
        for (j = 0; j < 3; j++) cmt[j] += cm[j];
        npart++;  
      }
    }
    MPI_Allreduce(&ii,&id_min,1,MPI_LMP_TAGINT,MPI_MIN,world);
    MPI_Allreduce(&npart,&g_num,1,MPI_INT,MPI_SUM,world); 

    for (j = 0; j < 3; j++){ 
      cmt[j] /= g_num;
      cm[j] = 0.0; 
    }
    MPI_Allreduce(&cmt[0],&cm[0],3,MPI_DOUBLE,MPI_SUM,world);

    int bn1[g_num], bn2[g_num];
    for (i = 0; i < g_num; i++){
      bn1[i] = 0;
      bn2[i] = 0;
    }
    for (i = 0; i < nlocal; i++){
      mol = molecule[i];
      if (mol && mask[i] & groupbit){
        domain->unmap(x[i],atom->image[i],cmt);
        rr = sqrt((cmt[0]-cm[0])*(cmt[0]-cm[0]) + (cmt[1]-cm[1])*(cmt[1]-cm[1]));
        j = static_cast<int> (rr/size_bin);
        if (j < nx && cmt[2] >= cm[2]){
          ij = tag[i] - id_min;
          bn1[ij] = j+1;
	}
      }
    }
    MPI_Reduce(&bn1[0],&bn2[0],g_num,MPI_INT,MPI_MAX,0,world);  
    if (comm->me == 0){  
      nr_bin = (int *) memory->smalloc(nx*sizeof(int),"statistic_corr:nr_bin");
      for (j = 0; j < nx; j++) nr_bin[j] = 0;
      memory->create(bin_list,nx,g_num,"statistic_corr:bin_list");
      npart = 0;
      for (i = 0; i < g_num; i++)
        if (bn2[i] > 0.5){ 
          ij = bn2[i] - 1;
          bin_list[ij][nr_bin[ij]] = i;
          nr_bin[ij]++;
          npart++;
	}
       
      corr_map = (int *) memory->smalloc(ny*sizeof(int),"statistic_corr:corr_map"); 
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
       
      if (nwind && npart) {
        memory->create(corr,npart,nwind,"statistic_corr:corr");
        memory->create(z_cur,npart,nwind,"statistic_corr:z_cur");
      }
      for (i = 0; i < npart; i++)
        for (j = 0; j < nwind; j++){
          corr[i][j] = 0.0;
          z_cur[i][j] = 0.0;
	}
      if (npart){
        memory->create(z_cr,npart,dump_each,"statistic_corr:z_cr");
        z_avg = (double *) memory->smalloc(npart*sizeof(double),"statistic_corr:z_avg");
        for (j = 0; j < npart; j++) 
          z_avg[j] = 0.0;
      }
      rot_list = (int *) memory->smalloc(nwind*sizeof(int),"statistic_corr:rot_list");   
    }
  }
 
  double zcm1[g_num], zcm2[g_num];
  for (i = 0; i < g_num; i++){
    zcm1[i] = -9999999;
    zcm2[i] = -9999999;
  }
  for (i = 0; i < nlocal; i++){
    mol = molecule[i];
    if (mol && mask[i] & groupbit){
      ij = tag[i] - id_min;
      zcm1[ij] = x[i][2];
    }
  }
  MPI_Reduce(&zcm1[0],&zcm2[0],g_num,MPI_DOUBLE,MPI_MAX,0,world);
  
  st_accum++;
  if (st_accum > dump_each)
    st_accum = dump_each + 1;

  if (comm->me == 0){
    avg_step++;
    mol = (update->ntimestep - st_start - 1)%dump_each;
    k = 0; 
    for (i = 0; i < nx; i++)
      for (j = 0; j < nr_bin[i]; j++){
        l = bin_list[i][j];
	z_cr[k][mol] = zcm2[l];
	z_avg[k] += zcm2[l];
        k++;
      }

    if (st_accum > dump_each && update->ntimestep%nz == 0){
      num_step++;
      for (i = 0; i < nwind; i++)
        rot_list[i] = (mol+1+corr_map[i])%dump_each;
      i_b = rot_list[0];
      k = 0;
      for (i = 0; i < nx; i++)
        for (j = 0; j < nr_bin[i]; j++){ 
          for (l = 0; l < nwind; l++){
            ij = rot_list[l];
            corr[k][l] += z_cr[k][i_b]*z_cr[k][ij];
            z_cur[k][l] += z_cr[k][ij]; 
          }  
          k++;        
	}
    }
  }
}

/* ---------------------------------------------------------------------- */

void StatisticCorr::write_stat(bigint step)
{
  int i, j, k, l;
  double tt,zz;
  double dt = update->dt;
  double **corr_bin, *z_avg_bin;
  char f_name[FILENAME_MAX];

  if (st_accum <= dump_each){
    num_step = 0; 
    return;
  }

  if (comm->me == 0){
    if (nwind) memory->create(corr_bin,nx,nwind,"statistic_corr:corr_bin");
    for (i = 0; i < nx; i++)
      for (j = 0; j < nwind; j++) 
        corr_bin[i][j] = 0.0;

    k = 0;
    for (i = 0; i < nx; i++)
      for (j = 0; j < nr_bin[i]; j++){
        zz = z_avg[k]/avg_step; 
        for (l = 0; l < nwind; l++){
          tt = (corr[k][l] - zz*(z_cur[k][0] + z_cur[k][l]))/num_step + zz*zz;
          corr_bin[i][l] += tt;
          //corr[k][l] = 0.0;
        }
        //z_avg[k] = 0.0;
        k++;        
      }
    for (i = 0; i < nx; i++)
      if (nr_bin[i] > 0)
        for (j = 0; j < nwind; j++) 
          corr_bin[i][j] /= nr_bin[i];
    //num_step = 0;
    //avg_step = 0;

    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    for (j = 0; j < nwind; j++){
      tt = corr_map[j]*dt;
      fprintf(out_stat,"%g ", tt);
      for (i = 0; i < nx; i++)
        fprintf(out_stat,"%15.10lf ", corr_bin[i][j]);
      fprintf(out_stat," \n");    
    }
    fclose(out_stat);

    memory->destroy(corr_bin);
  }  
}

/* ---------------------------------------------------------------------- */

