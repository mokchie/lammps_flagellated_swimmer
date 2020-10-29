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
#include "statistic_ctensor.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticCtensor::StatisticCtensor(LAMMPS *lmp, int narg, char **arg) : Statistic(lmp,narg,arg)
{
  if (!atom->ctensor_flag)
    error->all(FLERR,"Trying to ouput conformation tensor that isn't allocated");
  int i, j, k;
  memory->create(num,nx,ny,nz,"statistic_ctensor:num");
  memory->create(ctensor11,nx,ny,nz,"statistic_ctensor:ctensor11"); 
  memory->create(ctensor22,nx,ny,nz,"statistic_ctensor:ctensor22"); 
  memory->create(ctensor33,nx,ny,nz,"statistic_ctensor:ctensor33"); 
  memory->create(ctensor12,nx,ny,nz,"statistic_ctensor:ctensor12"); 
  memory->create(ctensor13,nx,ny,nz,"statistic_ctensor:ctensor13"); 
  memory->create(ctensor23,nx,ny,nz,"statistic_ctensor:ctensor23"); 
  for (i=0;i<nx;i++)
    for (j=0;j<ny;j++)
      for (k=0;k<nz;k++){
        num[i][j][k] = 0.0;
        ctensor11[i][j][k] = 0.0;
        ctensor22[i][j][k] = 0.0;
        ctensor33[i][j][k] = 0.0;
        ctensor12[i][j][k] = 0.0;
        ctensor13[i][j][k] = 0.0;
        ctensor23[i][j][k] = 0.0;
      }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticCtensor::~StatisticCtensor()
{
  memory->destroy(num);
  memory->destroy(ctensor11);
  memory->destroy(ctensor22);
  memory->destroy(ctensor33);
  memory->destroy(ctensor12);
  memory->destroy(ctensor13);
  memory->destroy(ctensor23);
}

/* ---------------------------------------------------------------------- */

void StatisticCtensor::calc_stat()
{
  int *mask = atom->mask;
  double **x = atom->x;
  double ***ctensor = atom->ctensor;
  double rr;

  for (int l=0;l<atom->nlocal;l++) 
    if (mask[l] & groupbit)    
      if (map_index(x[l][0],x[l][1],x[l][2])){
        num[is][js][ks] += 1.0;
        ctensor11[is][js][ks] += ctensor[l][0][0];
        ctensor22[is][js][ks] += ctensor[l][1][1];
        ctensor33[is][js][ks] += ctensor[l][2][2];
        ctensor12[is][js][ks] += ctensor[l][0][1];
        ctensor13[is][js][ks] += ctensor[l][0][2];
        ctensor23[is][js][ks] += ctensor[l][1][2];
      }
  num_step++;  

}

/* ---------------------------------------------------------------------- */

void StatisticCtensor:: write_stat(bigint step)
{
  int i, j, k, l;
  double x, y, z, rr, theta;
  int total = nx*ny*nz;
  double *ntmp, *ctensor11tmp, *ctensor22tmp, *ctensor33tmp, *ctensor12tmp, *ctensor13tmp, *ctensor23tmp, *tmp;
  char f_name[FILENAME_MAX];

  memory->create(ntmp,total,"statistic_ctensor:ntmp");
  memory->create(ctensor11tmp,total,"statistic_ctensor:ctensor11tmp");
  memory->create(ctensor22tmp,total,"statistic_ctensor:ctensor22tmp");
  memory->create(ctensor33tmp,total,"statistic_ctensor:ctensor33tmp");
  memory->create(ctensor12tmp,total,"statistic_ctensor:ctensor12tmp");
  memory->create(ctensor13tmp,total,"statistic_ctensor:ctensor13tmp");
  memory->create(ctensor23tmp,total,"statistic_ctensor:ctensor23tmp");
  memory->create(tmp,total,"statistic_ctensor:tmp");

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=num[i][j][k]/num_step;
        num[i][j][k] = 0.0;
        ntmp[l] = 0.0;
        ctensor11tmp[l] = 0.0;
        ctensor22tmp[l] = 0.0;
        ctensor33tmp[l] = 0.0;
        ctensor12tmp[l] = 0.0;
        ctensor13tmp[l] = 0.0;
        ctensor23tmp[l] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,ntmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=ctensor11[i][j][k]/num_step;
        ctensor11[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,ctensor11tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=ctensor22[i][j][k]/num_step;
        ctensor22[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,ctensor22tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=ctensor33[i][j][k]/num_step;
        ctensor33[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,ctensor33tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=ctensor12[i][j][k]/num_step;
        ctensor12[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,ctensor12tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=ctensor13[i][j][k]/num_step;
        ctensor13[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,ctensor13tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=ctensor23[i][j][k]/num_step;
        ctensor23[i][j][k] = 0.0;
  l++;
      }
  MPI_Reduce(tmp,ctensor23tmp,total,MPI_DOUBLE,MPI_SUM,0,world);

  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"ctensor11\",\"ctensor22\",\"ctensor33\",\"ctensor12\",\"ctensor13\",\"ctensor23\"\n");
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
              fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf \n",x, y, z, ctensor11tmp[l]/ntmp[l], ctensor22tmp[l]/ntmp[l],ctensor33tmp[l]/ntmp[l], ctensor12tmp[l]/ntmp[l], ctensor13tmp[l]/ntmp[l], ctensor23tmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 0.0 0.0 0.0 0.0 0.0\n",x, y, z);
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
              fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf\n",x, y, z, ctensor11tmp[l]/ntmp[l], ctensor22tmp[l]/ntmp[l],ctensor33tmp[l]/ntmp[l], ctensor12tmp[l]/ntmp[l], ctensor13tmp[l]/ntmp[l], ctensor23tmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 0.0 0.0 0.0 0.0 0.0\n",x, y, z);            
            l++; 
	  }
    }
    fclose(out_stat);
  }    
  memory->destroy(ntmp);
  memory->destroy(ctensor11tmp);
  memory->destroy(ctensor22tmp);
  memory->destroy(ctensor33tmp);
  memory->destroy(ctensor12tmp);
  memory->destroy(ctensor13tmp);
  memory->destroy(ctensor23tmp);
  memory->destroy(tmp);    
}

/* ---------------------------------------------------------------------- */

