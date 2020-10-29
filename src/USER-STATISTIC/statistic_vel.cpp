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
#include "statistic_vel.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticVel::StatisticVel(LAMMPS *lmp, int narg, char **arg) : Statistic(lmp,narg,arg)
{
  int i, j, k;
  memory->create(num,nx,ny,nz,"statistic_vel:num");
  memory->create(vx,nx,ny,nz,"statistic_vel:vx"); 
  memory->create(vy,nx,ny,nz,"statistic_vel:vy"); 
  memory->create(vz,nx,ny,nz,"statistic_vel:vz"); 
  for (i=0;i<nx;i++)
    for (j=0;j<ny;j++)
      for (k=0;k<nz;k++){
        num[i][j][k] = 0.0;
        vx[i][j][k] = 0.0;
        vy[i][j][k] = 0.0;
        vz[i][j][k] = 0.0;
      }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticVel::~StatisticVel()
{
  memory->destroy(num);
  memory->destroy(vx);
  memory->destroy(vy);
  memory->destroy(vz);
}

/* ---------------------------------------------------------------------- */

void StatisticVel::calc_stat()
{
  int *mask = atom->mask;
  double **x = atom->x;
  double **v = atom->v;
  double rr;

  for (int l=0;l<atom->nlocal;l++) 
    if (mask[l] & groupbit)    
      if (map_index(x[l][0],x[l][1],x[l][2])){
        num[is][js][ks] += 1.0;
        vx[is][js][ks] += v[l][0];
        if (cyl_ind && zhi > 0.5){
          rr = sqrt((x[l][1]-ylo)*(x[l][1]-ylo) + (x[l][2]-zlo)*(x[l][2]-zlo));
          if (rr > 0.0){
            vy[is][js][ks] += (v[l][1]*(x[l][1]-ylo) + v[l][2]*(x[l][2]-zlo))/rr;
            vz[is][js][ks] += ((x[l][1]-ylo)*v[l][2] - (x[l][2]-zlo)*v[l][1])/rr;
          }
        } else{
          vy[is][js][ks] += v[l][1];
          vz[is][js][ks] += v[l][2];
          if (comm->le) vy[is][js][ks] += jv*comm->u_le;
        } 
      }
  num_step++;  

}

/* ---------------------------------------------------------------------- */

void StatisticVel:: write_stat(bigint step)
{
  int i, j, k, l;
  double x, y, z, rr, theta;
  int total = nx*ny*nz;
  double *ntmp, *vxtmp, *vytmp, *vztmp, *tmp;
  char f_name[FILENAME_MAX];

  memory->create(ntmp,total,"statistic_vel:ntmp");
  memory->create(vxtmp,total,"statistic_vel:vxtmp");
  memory->create(vytmp,total,"statistic_vel:vytmp");
  memory->create(vztmp,total,"statistic_vel:vztmp");
  memory->create(tmp,total,"statistic_vel:tmp");

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=num[i][j][k]/num_step;
        num[i][j][k] = 0.0;
        ntmp[l] = 0.0;
        vxtmp[l] = 0.0;
        vytmp[l] = 0.0;
        vztmp[l] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,ntmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=vx[i][j][k]/num_step;
        vx[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,vxtmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=vy[i][j][k]/num_step;
        vy[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,vytmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){ 
        tmp[l]=vz[i][j][k]/num_step;
        vz[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,vztmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"v_x\",\"v_y\",\"v_z\"  \n");
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
              fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf \n",x, y, z, vxtmp[l]/ntmp[l], vytmp[l]/ntmp[l],vztmp[l]/ntmp[l]);
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
              fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf \n",x, y, z, vxtmp[l]/ntmp[l], vytmp[l]/ntmp[l],vztmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 0.0 0.0 \n",x, y, z);            
            l++; 
	  }
    }
    fclose(out_stat);
  }    
  memory->destroy(ntmp);
  memory->destroy(vxtmp);
  memory->destroy(vytmp);
  memory->destroy(vztmp);
  memory->destroy(tmp);    
}

/* ---------------------------------------------------------------------- */

