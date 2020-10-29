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
#include "statistic_vol.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"
#include "neighbor.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticVol::StatisticVol(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
    memory->create(dath,2,"statistic_vol:dath");
    memory->create(datt,2,"statistic_vol:datt");
    num_step = 0;
    file_open = 0;
}

/* ---------------------------------------------------------------------- */

StatisticVol::~StatisticVol()
{
  memory->destroy(datt);
  memory->destroy(dath);
}

/* ---------------------------------------------------------------------- */

void StatisticVol::calc_stat()
{
  int i1,i2,i3,n,j,type,kk,l;
  tagint m;
  double d21x,d21y,d21z,d31x,d31y,d31z,d32x,d32y,d32z;
  double nx,ny,nz,nn,mx,my,mz,aa,vv;

  tagint nm = atom->n_mol_max;
  double xx1[3],xx2[3],xx3[3];

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  double *anglelist_area = neighbor->anglelist_area;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nanglelist; n++) {

    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    m = atom->molecule[i1]-1;

    // 2-1 distance
    d21x = x[i2][0] - x[i1][0];
    d21y = x[i2][1] - x[i1][1];
    d21z = x[i2][2] - x[i1][2];

    // 3-1 distance
    d31x = x[i3][0] - x[i1][0];
    d31y = x[i3][1] - x[i1][1];
    d31z = x[i3][2] - x[i1][2];

    // calculate normal
    nx = d21y*d31z - d31y*d21z;
    ny = d31x*d21z - d21x*d31z;
    nz = d21x*d31y - d31x*d21y;
    nn = sqrt(nx*nx + ny*ny + nz*nz);

    // calculate center
    domain->unmap(x[i1],atom->image[i1],xx1);
    domain->unmap(x[i2],atom->image[i2],xx2);
    domain->unmap(x[i3],atom->image[i3],xx3);

    mx =  xx1[0] + xx2[0] + xx3[0];
    my =  xx1[1] + xx2[1] + xx3[1];
    mz =  xx1[2] + xx2[2] + xx3[2];

    // calculate area and volume
    aa = 0.5*nn;
    vv = (nx*mx + ny*my + nz*mz)/18.0;
    dath[m] += aa;
    dath[m+nm] += vv;
  }
  num_step++;  

}

/* ---------------------------------------------------------------------- */

void StatisticVol:: write_stat(bigint step)
{
  char f_name[FILENAME_MAX];
  tagint nm = atom->n_mol_max;
  MPI_Allreduce(&dath[0],&datt[0],2*nm,MPI_DOUBLE,MPI_SUM,world);
  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s.vol",fname);
    if (0 == file_open){
      out_stat=fopen(f_name,"w");
      fprintf(out_stat,"VARIABLES=\"Time\",\"Area\",\"Volume\"\n");
      file_open = 1;
    } else {
      out_stat=fopen(f_name,"a");
    }
    fprintf(out_stat,"%lf %lf %lf\n", step*update->dt, datt[0]/num_step, datt[1]/num_step);
    fclose(out_stat);
  }
}

/* ---------------------------------------------------------------------- */

