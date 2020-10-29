
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
/* ----------------------------------------------------------------------
   Contributing author: Brooke Huisman, Masoud Hoore (FZJ)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
# Statistic_rdf command
# statistic stat-name group-id cyl_ind   nbin cutoff dummy   stat_start calc_each dump_each (c1 c2 c3 c4 c5 c6) fname
		0	1	2	3	4	5	6	7		8    9  10  11 12 13 14  15
  # cylindrical coordinate system (1) has axis in y-direction
------------------------------------------------------------------------- */

#include <cmath>
#include "statistic_rdf.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"
#include "domain.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticRdf::StatisticRdf(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{

  error->warning(FLERR,"statistics_rdf slows down the simulation, take care about calc_each and dump_each values.");

  int i;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  nbin = nx;
  cutoff = ny;

  dr = cutoff/nbin;
  cutoffsq = cutoff*cutoff;

  rdf_time=0;

  Lx = domain->boxhi[0] - domain->boxlo[0];
  Ly = domain->boxhi[1] - domain->boxlo[1];
  Lz = domain->boxhi[2] - domain->boxlo[2];
  Lxhalf= 0.5*Lx;
  Lyhalf= 0.5*Ly;
  Lzhalf= 0.5*Lz;

  nall_tmp = 0;
  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit)
      nall_tmp += 1;

  nall_tmp *= 3; //for x, y, z components

  MPI_Allreduce(&nall_tmp, &nall, 1, MPI_INT, MPI_SUM, world);

  memory->create(rdf, nbin, "statistic_rdf:rdf");

  for(i=0; i<nbin; i++)
    rdf[i] = 0.0;

}

/* ---------------------------------------------------------------------- */

StatisticRdf::~StatisticRdf()
{
  memory->destroy(rdf);
}

/* ---------------------------------------------------------------------- */

void StatisticRdf::calc_stat()
{

  int i, j;

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *rcounts = NULL;
  int *displs = NULL; 
  double *x_tmp = NULL;
  double *x_tmp_all = NULL;
  double distx, disty, distz, rsq;

  memory->create(rcounts,comm->nprocs,"statistic_rdf:rcounts");
  memory->create(displs,comm->nprocs,"statistic_rdf:displs");

  nall_tmp = 0;
  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit)
      nall_tmp++;
  nall_tmp *= 3; //for x, y, z components

  memory->create(x_tmp, nall_tmp, "statistic_rdf:x_tmp");
  memory->create(x_tmp_all, nall, "statistic_rdf:x_tmp_all");

  int ii = 0;
  for (i=0; i<nlocal; i++)
    if (mask[i] & groupbit) {
      x_tmp[ii] = x[i][0];
      x_tmp[ii+1] = x[i][1];
      x_tmp[ii+2] = x[i][2];
      ii += 3;
    }

  //Sends the nall_tmp value from each processor into array "rcounts"
  MPI_Allgather(&nall_tmp, 1, MPI_INT, rcounts, 1, MPI_INT, world); 

  int offset = 0;
  for (i = 0; i < comm->nprocs; i++) {
    displs[i] = offset;
    offset += rcounts[i];
  }

  MPI_Allgatherv(x_tmp, nall_tmp, MPI_DOUBLE, x_tmp_all, rcounts, displs, MPI_DOUBLE, world);


  int me = comm->me;  
  
  if (cyl_ind == 0) { 
    for(i=displs[me]; i<displs[me]+rcounts[me]; i+=3) { //for all particles for processor 'me' (currently in) --split up to processors
      for(j=i+3; j<nall; j+=3)  {
        distx = x_tmp_all[i] - x_tmp_all[j];
        disty = x_tmp_all[i+1] - x_tmp_all[j+1];
        distz = x_tmp_all[i+2] - x_tmp_all[j+2];

        if(distx > Lxhalf) distx = Lx-distx;
        else if(distx < -Lxhalf) distx = distx+Lx;
        if(disty > Lyhalf) disty = Ly-disty;
        else if(disty < -Lyhalf) disty = disty+Ly;
        if(distz > Lzhalf) distz = Lz-distz;
        else if(distz < -Lzhalf) distz = distz+Lz;

        rsq = distx*distx + disty*disty + distz*distz;
        if (rsq < cutoffsq){
          int dum;
          dum = floor(sqrt(rsq)/dr);
          rdf[dum] += 2.0;
        }
      }
    }
  }



  if (cyl_ind == 1) { 
    for(i=displs[me]; i<displs[me]+rcounts[me]; i+=3) { //for all particles for processor 'me' (currently in) --split up to processors
      for(j=i+3; j<nall; j+=3)  {
        distx = x_tmp_all[i] - x_tmp_all[j];
        distz = x_tmp_all[i+2] - x_tmp_all[j+2];

        if(distx > Lxhalf) distx = Lx-distx;
        else if(distx < -Lxhalf) distx = distx+Lx;
        if(distz > Lzhalf) distz = Lz-distz;
        else if(distz < -Lzhalf) distz = distz+Lz;

        rsq = distx*distx + distz*distz;
        if (rsq < cutoffsq){
          int dum;
          dum = floor(sqrt(rsq)/dr);
          rdf[dum] += 2.0;
        }
      }
    }
  }


  memory->destroy(rcounts);
  memory->destroy(displs);
  memory->destroy(x_tmp);
  memory->destroy(x_tmp_all);

  rdf_time++;

}

/* ---------------------------------------------------------------------- */

void StatisticRdf:: write_stat(bigint step)
{
  int i;
  char f_name[FILENAME_MAX];
  double *rdf_tot;

  if (!comm->me){
    memory->create(rdf_tot, nbin, "statistic_rdf:rdf_tot");
  }

  MPI_Reduce(rdf, rdf_tot, nbin, MPI_DOUBLE, MPI_SUM, 0, world);

  if (!comm->me){
    if (cyl_ind == 0) {
      double dum=3.0/(4.0*M_PI*dr*dr*dr*nall*rdf_time);
      for(i=0; i<nbin; i++)
        rdf_tot[i] *= dum/((i+1)*(i+1));
    }
    else if (cyl_ind == 1) { 
      double dum=3.0/(2.0*M_PI*dr*dr*nall*rdf_time*Ly);
      for(i=0; i<nbin; i++)
        rdf_tot[i] *= dum/(i+1);
    }

    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step);

    out_stat=fopen(f_name,"w");
    fclose(out_stat);

    out_stat=fopen(f_name,"a");
    fprintf(out_stat,"ITEM: TIMESTEP\n");
    fprintf(out_stat,"%i \n", step);   

    fprintf(out_stat, "r n_avg \n");

    for(i=0; i<nbin; i++) 
      fprintf(out_stat,"%2.3f %2.3f \n", (i+1)*dr, rdf_tot[i]);  

    fclose(out_stat);
    memory->destroy(rdf_tot);
  }

  for(i=0; i<nbin; i++)
    rdf[i]=0.0;

  rdf_time=0;

}

/* ---------------------------------------------------------------------- */
