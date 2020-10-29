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
#include "statistic_kappa.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticKappa::StatisticKappa(LAMMPS *lmp, int narg, char **arg) : Statistic(lmp,narg,arg)
{
  if (!atom->ctensor_flag)
    error->all(FLERR,"Trying to ouput conformation tensor that isn't allocated");
  int i, j, k;
  memory->create(num,nx,ny,nz,"statistic_kappa:num");
  memory->create(kappa11,nx,ny,nz,"statistic_kappa:kappa11"); 
  memory->create(kappa22,nx,ny,nz,"statistic_kappa:kappa22"); 
  memory->create(kappa33,nx,ny,nz,"statistic_kappa:kappa33"); 
  memory->create(kappa12,nx,ny,nz,"statistic_kappa:kappa12"); 
  memory->create(kappa21,nx,ny,nz,"statistic_kappa:kappa21"); 
  memory->create(kappa13,nx,ny,nz,"statistic_kappa:kappa13"); 
  memory->create(kappa31,nx,ny,nz,"statistic_kappa:kappa31"); 
  memory->create(kappa23,nx,ny,nz,"statistic_kappa:kappa23"); 
  memory->create(kappa32,nx,ny,nz,"statistic_kappa:kappa32"); 
  for (i=0;i<nx;i++)
    for (j=0;j<ny;j++)
      for (k=0;k<nz;k++){
        num[i][j][k] = 0.0;
        kappa11[i][j][k] = 0.0;
        kappa22[i][j][k] = 0.0;
        kappa33[i][j][k] = 0.0;
        kappa12[i][j][k] = 0.0;
        kappa21[i][j][k] = 0.0;
        kappa13[i][j][k] = 0.0;
        kappa31[i][j][k] = 0.0;
        kappa23[i][j][k] = 0.0;
        kappa32[i][j][k] = 0.0;
      }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticKappa::~StatisticKappa()
{
  memory->destroy(num);
  memory->destroy(kappa11);
  memory->destroy(kappa22);
  memory->destroy(kappa33);
  memory->destroy(kappa12);
  memory->destroy(kappa21);
  memory->destroy(kappa13);
  memory->destroy(kappa31);
  memory->destroy(kappa23);
  memory->destroy(kappa32);
}

/* ---------------------------------------------------------------------- */

void StatisticKappa::calc_stat()
{
  int *mask = atom->mask;
  double **x = atom->x;
  double ***kappa = atom->kappa;
  double rr;

  for (int l=0;l<atom->nlocal;l++) 
    if (mask[l] & groupbit)    
      if (map_index(x[l][0],x[l][1],x[l][2])){
        num[is][js][ks] += 1.0;
        kappa11[is][js][ks] += kappa[l][0][0];
        kappa22[is][js][ks] += kappa[l][1][1];
        kappa33[is][js][ks] += kappa[l][2][2];
        kappa12[is][js][ks] += kappa[l][0][1];
        kappa21[is][js][ks] += kappa[l][1][0];
        kappa13[is][js][ks] += kappa[l][0][2];
        kappa31[is][js][ks] += kappa[l][2][0];
        kappa23[is][js][ks] += kappa[l][1][2];
        kappa32[is][js][ks] += kappa[l][2][1];
      }
  num_step++;  

}

/* ---------------------------------------------------------------------- */

void StatisticKappa:: write_stat(bigint step)
{
  int i, j, k, l;
  double x, y, z, rr, theta;
  int total = nx*ny*nz;
  double *ntmp, *kappa11tmp, *kappa22tmp, *kappa33tmp, *kappa12tmp, *kappa13tmp, *kappa23tmp, *kappa21tmp, *kappa31tmp, *kappa32tmp, *tmp;
  char f_name[FILENAME_MAX];

  memory->create(ntmp,total,"statistic_kappa:ntmp");
  memory->create(kappa11tmp,total,"statistic_kappa:kappa11tmp");
  memory->create(kappa22tmp,total,"statistic_kappa:kappa22tmp");
  memory->create(kappa33tmp,total,"statistic_kappa:kappa33tmp");
  memory->create(kappa12tmp,total,"statistic_kappa:kappa12tmp");
  memory->create(kappa13tmp,total,"statistic_kappa:kappa13tmp");
  memory->create(kappa23tmp,total,"statistic_kappa:kappa23tmp");
  memory->create(kappa21tmp,total,"statistic_kappa:kappa21tmp");
  memory->create(kappa31tmp,total,"statistic_kappa:kappa31tmp");
  memory->create(kappa32tmp,total,"statistic_kappa:kappa32tmp");
  memory->create(tmp,total,"statistic_kappa:tmp");

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=num[i][j][k]/num_step;
        num[i][j][k] = 0.0;
        ntmp[l] = 0.0;
        kappa11tmp[l] = 0.0;
        kappa22tmp[l] = 0.0;
        kappa33tmp[l] = 0.0;
        kappa12tmp[l] = 0.0;
        kappa13tmp[l] = 0.0;
        kappa23tmp[l] = 0.0;
        kappa21tmp[l] = 0.0;
        kappa31tmp[l] = 0.0;
        kappa32tmp[l] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,ntmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=kappa11[i][j][k]/num_step;
        kappa11[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,kappa11tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=kappa22[i][j][k]/num_step;
        kappa22[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,kappa22tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=kappa33[i][j][k]/num_step;
        kappa33[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,kappa33tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=kappa12[i][j][k]/num_step;
        kappa12[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,kappa12tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=kappa13[i][j][k]/num_step;
        kappa13[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,kappa13tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=kappa23[i][j][k]/num_step;
        kappa23[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,kappa23tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=kappa21[i][j][k]/num_step;
        kappa21[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,kappa21tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=kappa31[i][j][k]/num_step;
        kappa31[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,kappa31tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=kappa32[i][j][k]/num_step;
        kappa32[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,kappa32tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"kappa11\",\"kappa12\",\"kappa13\",\"kappa21\",\"kappa22\",\"kappa23\",\"kappa31\",\"kappa32\",\"kappa33\",\n");
    if (cyl_ind){
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", ny, nz, nx);
      for (i=0; i<nx; i++)
        for (k=0; k<nz; k++)
          for (j=0; j<ny; j++){ 
            l = k*nx*ny + j*nx + i;
            x = xlo + (i+0.5)*dx;
            rr = (j+0.5)*dy;
            theta = k*dz;
            y = ylo + rr*cos(theta);
            z = zlo + rr*sin(theta);
            if (ntmp[l] > 0.0)
              fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf \n",x, y, z, kappa11tmp[l]/ntmp[l], kappa12tmp[l]/ntmp[l],kappa13tmp[l]/ntmp[l], kappa21tmp[l]/ntmp[l], kappa22tmp[l]/ntmp[l], kappa23tmp[l]/ntmp[l], kappa31tmp[l]/ntmp[l], kappa32tmp[l]/ntmp[l], kappa33tmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n",x, y, z);
	  }
    } else{
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", nx, ny, nz);
      l = 0;
      for (k=0; k<nz; k++)
        for (j=0; j<ny; j++)
          for (i=0; i<nx; i++){ 
            x = xlo + (i+0.5)*dx;
            y = ylo + (j+0.5)*dy;
            z = zlo + (k+0.5)*dz;
            if (ntmp[l] > 0.0)
              fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf \n",x, y, z, kappa11tmp[l]/ntmp[l], kappa12tmp[l]/ntmp[l],kappa13tmp[l]/ntmp[l], kappa21tmp[l]/ntmp[l], kappa22tmp[l]/ntmp[l], kappa23tmp[l]/ntmp[l], kappa31tmp[l]/ntmp[l], kappa32tmp[l]/ntmp[l], kappa33tmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n",x, y, z);           
            l++; 
	  }
    }
    fclose(out_stat);
  }    
  memory->destroy(ntmp);
  memory->destroy(kappa11tmp);
  memory->destroy(kappa22tmp);
  memory->destroy(kappa33tmp);
  memory->destroy(kappa12tmp);
  memory->destroy(kappa13tmp);
  memory->destroy(kappa23tmp);
  memory->destroy(kappa21tmp);
  memory->destroy(kappa31tmp);
  memory->destroy(kappa32tmp);
  memory->destroy(tmp);    
}

/* ---------------------------------------------------------------------- */

