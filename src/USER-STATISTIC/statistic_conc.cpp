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
#include "statistic_conc.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticConc::StatisticConc(LAMMPS *lmp, int narg, char **arg) : Statistic(lmp,narg,arg)
{
  if (!atom->conc_flag)
    error->all(FLERR,"Trying to ouput concentration that isn't allocated");
  int i, j, k, l;
  memory->create(num,nx,ny,nz,"statistic_conc:num");
  memory->create(gconc,nx,ny,nz,atom->individual,"statistic_conc:gconc"); 
  for (i=0;i<nx;i++)
    for (j=0;j<ny;j++)
      for (k=0;k<nz;k++){
        num[i][j][k] = 0.0;
        for (l=0;l<atom->individual;l++){
          gconc[i][j][k][l] = 0.0;
        }
      }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticConc::~StatisticConc()
{
  memory->destroy(num);
  memory->destroy(gconc);
}

/* ---------------------------------------------------------------------- */

void StatisticConc::calc_stat()
{
  int *mask = atom->mask;
  double **x = atom->x;
  double **conc = atom->conc;
  double rr;

  for (int l=0;l<atom->nlocal;l++) 
    if (mask[l] & groupbit)    
      if (map_index(x[l][0],x[l][1],x[l][2])){
        num[is][js][ks] += 1.0;
        for (int kk=0; kk<atom->individual; kk++){
          gconc[is][js][ks][kk] += conc[l][kk];
        }

      }
  num_step++;  
}

/* ---------------------------------------------------------------------- */

void StatisticConc:: write_stat(bigint step)
{
  int i, j, k, l, kk;
  double x, y, z, rr, theta;
  int total =  nx*ny*nz;
  double *ntmp, **gconctmp, *tmp;
  char f_name[FILENAME_MAX];

  memory->create(ntmp,total,"statistic_conc:ntmp");
  memory->create(gconctmp,atom->individual,total,"statistic_conc:gconctmp");
  memory->create(tmp,total,"statistic_conc:tmp");

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=num[i][j][k]/num_step;
        ntmp[l] = 0.0;
        for(kk=0; kk<atom->individual; kk++)
          gconctmp[kk][l] = 0.0;
        num[i][j][k] = 0.0;
	      l++;
      }
  MPI_Reduce(tmp,ntmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  for(kk=0; kk<atom->individual; kk++){
    l = 0;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
        for (i=0; i<nx; i++){
          tmp[l]=gconc[i][j][k][kk]/num_step;
          gconc[i][j][k][kk] = 0.0;
  	l++;
        }
    MPI_Reduce(tmp,gconctmp[kk],total,MPI_DOUBLE,MPI_SUM,0,world);
  }
  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\"");
    for(kk=0; kk<atom->individual; kk++)
      fprintf(out_stat,",\"c%d\"",kk+1);
    fprintf(out_stat,"\n");
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
            fprintf(out_stat,"%lf %lf %lf",x, y, z);
            if (ntmp[l] > 0.0){
              for(kk=0; kk<atom->individual; kk++)
                fprintf(out_stat," %15.10lf",gconctmp[kk][l]/ntmp[l]);
              fprintf(out_stat,"\n");
            }
            else{
              for(kk=0; kk<atom->individual; kk++)
                fprintf(out_stat," %15.10lf",0.0);
              fprintf(out_stat,"\n");
            }
              
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
            fprintf(out_stat,"%lf %lf %lf",x, y, z);
            if (ntmp[l] > 0.0){
              for(kk=0; kk<atom->individual; kk++)
                fprintf(out_stat," %15.10lf",gconctmp[kk][l]/ntmp[l]);
              fprintf(out_stat,"\n");
            }
            else{
              for(kk=0; kk<atom->individual; kk++)
                fprintf(out_stat," %15.10lf",0.0);
              fprintf(out_stat,"\n");
            }
            l++; 
	  }
    }
    fclose(out_stat);
  }    
  memory->destroy(ntmp);
  memory->destroy(gconctmp);
  memory->destroy(tmp);    
}

/* ---------------------------------------------------------------------- */

