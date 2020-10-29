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
#include "statistic_dens.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticDens::StatisticDens(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  int i, j, k;
  memory->create(num,nx,ny,nz,"statistic_dens:num");
  memory->create(nmass,nx,ny,nz,"statistic_dens:nmass"); 
  for (i=0;i<nx;i++)
    for (j=0;j<ny;j++)
      for (k=0;k<nz;k++){
        num[i][j][k] = 0.0;
        nmass[i][j][k] = 0.0;
      }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticDens::~StatisticDens()
{
  memory->destroy(num);
  memory->destroy(nmass);
}

/* ---------------------------------------------------------------------- */

void StatisticDens::calc_stat()
{
  int *type = atom->type;
  int *mask = atom->mask;
  double *mass = atom->mass;
  double **x = atom->x;

  for (int l=0; l<atom->nlocal; l++)
    if (mask[l] & groupbit)   
      if (map_index(x[l][0],x[l][1],x[l][2])){
        num[is][js][ks] += 1.0;
        nmass[is][js][ks] += mass[type[l]];
      }
  num_step++;

}

/* ---------------------------------------------------------------------- */

void StatisticDens:: write_stat(bigint step)
{
  int i, j, k, l;
  int total = nx*ny*nz;
  double x, y, z, rr, theta, vol;
  double *ntmp, *mtmp, *tmp;
  char f_name[FILENAME_MAX];

  memory->create(ntmp,total,"statistic_dens:ntmp");
  memory->create(mtmp,total,"statistic_dens:mtmp");
  memory->create(tmp,total,"statistic_dens:tmp");  

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=num[i][j][k]/num_step;
        num[i][j][k] = 0.0;
        ntmp[l] = 0.0;
        mtmp[l] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,ntmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=nmass[i][j][k]/num_step;
        nmass[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,mtmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"number density\",\"density\"  \n");
    if (cyl_ind){
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", ny, nz, nx);
      for (i=0; i<nx; i++)
        for (k=0; k<nz; k++)
          for (j=0; j<ny; j++){
            l = k*nx*ny + j*nx + i; 
            vol = dx*dz*(j+0.5)*dy*dy;
            x = xlo + (i+0.5)*dx;
            rr = (j+0.5)*dy;
            theta = k*dz;
            y = ylo + rr*cos(theta);
            z = zlo + rr*sin(theta);
            fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf \n",x, y, z, ntmp[l]/vol, mtmp[l]/vol);
          }
    } else {
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", nx, ny, nz);
      vol = dx*dy*dz;
      l = 0;
      for (k=0; k<nz; k++)
        for (j=0; j<ny; j++)
          for (i=0; i<nx; i++){
            x = xlo + (i+0.5)*dx;
            y = ylo + (j+0.5)*dy;
            z = zlo + (k+0.5)*dz;
            fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf \n",x, y, z, ntmp[l]/vol, mtmp[l]/vol);
            l++;
	  }
    }
    fclose(out_stat);
  }
  memory->destroy(ntmp);
  memory->destroy(mtmp);
  memory->destroy(tmp);    
}

/* ---------------------------------------------------------------------- */

