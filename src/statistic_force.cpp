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
#include "statistic_force.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticForce::StatisticForce(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  int i, j, k;
  memory->create(num,nx,ny,nz,"statistic_force:num");
  memory->create(fx,nx,ny,nz,"statistic_force:fx"); 
  memory->create(fy,nx,ny,nz,"statistic_force:fy"); 
  memory->create(fz,nx,ny,nz,"statistic_force:fz"); 
  for (i=0;i<nx;i++)
    for (j=0;j<ny;j++)
      for (k=0;k<nz;k++){
        num[i][j][k] = 0.0;
        fx[i][j][k] = 0.0;
        fy[i][j][k] = 0.0;
        fz[i][j][k] = 0.0;
      }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticForce::~StatisticForce()
{
  memory->destroy(num);
  memory->destroy(fx);
  memory->destroy(fy);
  memory->destroy(fz);
}

/* ---------------------------------------------------------------------- */

void StatisticForce::calc_stat()
{
  int *mask = atom->mask;
  double **x = atom->x;
  double **f = atom->f;
  double rr;

  for (int l=0;l<atom->nlocal;l++) 
    if (mask[l] & groupbit)    
      if (map_index(x[l][0],x[l][1],x[l][2])){
        num[is][js][ks] += 1.0;
        fx[is][js][ks] += f[l][0];
        if (cyl_ind && zhi > 0.5){
          rr = sqrt((x[l][1]-ylo)*(x[l][1]-ylo) + (x[l][2]-zlo)*(x[l][2]-zlo));
          if (rr > 0.0){
            fy[is][js][ks] += (f[l][1]*(x[l][1]-ylo) + f[l][2]*(x[l][2]-zlo))/rr;
            fz[is][js][ks] += ((x[l][1]-ylo)*f[l][2] - (x[l][2]-zlo)*f[l][1])/rr;
          }
        } else{
          fy[is][js][ks] += f[l][1];
          fz[is][js][ks] += f[l][2];
        }
      }
  num_step++;  

}

/* ---------------------------------------------------------------------- */

void StatisticForce:: write_stat(bigint step)
{
  int i, j, k, l;
  double x, y, z, rr, theta;
  int total = nx*ny*nz;
  double *ntmp, *fxtmp, *fytmp, *fztmp, *tmp;
  char f_name[FILENAME_MAX];

  ntmp = (double *) memory->smalloc(total*sizeof(double),"statistic_force:ntmp");
  fxtmp = (double *) memory->smalloc(total*sizeof(double),"statistic_force:fxtmp");
  fytmp = (double *) memory->smalloc(total*sizeof(double),"statistic_force:fytmp");
  fztmp = (double *) memory->smalloc(total*sizeof(double),"statistic_force:fztmp");
  tmp = (double *) memory->smalloc(total*sizeof(double),"statistic_force:tmp");

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=num[i][j][k]/num_step;
        num[i][j][k] = 0.0;
        ntmp[l] = 0.0;
        fxtmp[l] = 0.0;
        fytmp[l] = 0.0;
        fztmp[l] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,ntmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=fx[i][j][k]/num_step;
        fx[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,fxtmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=fy[i][j][k]/num_step;
        fy[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,fytmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=fz[i][j][k]/num_step;
        fz[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,fztmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"f_x\",\"f_y\",\"f_z\"  \n");
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
              fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf \n",x, y, z, fxtmp[l]/ntmp[l], fytmp[l]/ntmp[l],fztmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 0.0 0.0 \n",x, y, z);
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
              fprintf(out_stat,"%lf %lf %lf %lf %lf %lf \n",x, y, z, fxtmp[l]/ntmp[l], fytmp[l]/ntmp[l],fztmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 0.0 0.0 \n",x, y, z);            
            l++; 
	  }
    }
    fclose(out_stat);
  }    
  memory->sfree(ntmp);
  memory->sfree(fxtmp);
  memory->sfree(fytmp);
  memory->sfree(fztmp);
  memory->sfree(tmp);    
}

/* ---------------------------------------------------------------------- */

