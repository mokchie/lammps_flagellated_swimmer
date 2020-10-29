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
#include "statistic_vcorr.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"
#include "update.h"
#include "group.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticVCorr::StatisticVCorr(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  st_accum = 0;
  num_step = 0;
  cyl_ind = 0;
  step_each = 1;

  igroup = group->find(arg[1]);
  if (igroup == -1) error->one(FLERR,"Group ID for Statistic_vcorr does not exist");

  corr_map = rot_list = NULL; 
  corr = NULL;
  vel = NULL;
}

/* ---------------------------------------------------------------------- */

StatisticVCorr::~StatisticVCorr()
{
  if (comm->me == 0){
    memory->sfree(corr_map);
    memory->sfree(rot_list);
    memory->destroy(corr);
    memory->destroy(vel); 
  }
}

/* ---------------------------------------------------------------------- */

void StatisticVCorr::calc_stat()
{
  int i,j,k;
  double v0[3],lgmin,lgmax,dlg;

  if (st_accum == 0){ 
    masstotal = group->mass(igroup);
    if (comm->me == 0){
      time_corr = update->dt*dump_each;
      corr_map = (int *) memory->smalloc(ny*sizeof(int),"statistic_vcorr:corr_map");
      corr_map[0] = 1;
      lgmax = log(time_corr);
      lgmin = log(update->dt);
      dlg = (lgmax - lgmin)/(ny-1);
      nwind = 1;
      for (i = 1; i <= ny-1; i++){
        j = static_cast<int>(floor(exp(lgmin+i*dlg)/update->dt + 0.5));
        if (j > corr_map[nwind-1] && j < dump_each){
          corr_map[nwind] = j;
          nwind++;
        }
      }

      if (nwind) 
        memory->create(corr,nwind,"statistic_vcorr:corr");
      for (i = 0; i < nwind; i++)
        corr[i] = 0.0;
    
      memory->create(vel,dump_each,3,"statistic_vcorr:vel");
      rot_list = (int *) memory->smalloc(nwind*sizeof(int),"statistic_vcorr:rot_list");
    }
  }

  group->vcm(igroup,masstotal,v0);
 
  st_accum++;
  if (st_accum > dump_each)
    st_accum = dump_each + 1;

  if (comm->me == 0){
    k = (update->ntimestep - st_start - 1)%dump_each; 
    for (i = 0; i < 3; i++)
      vel[k][i] = v0[i];

    if (st_accum > dump_each && update->ntimestep%nx == 0){
      num_step++;
      j = (k+1)%dump_each;
      for (i = 0; i < nwind; i++)
        rot_list[i] = (k + 1 + corr_map[i])%dump_each;
      for (i = 0; i < nwind; i++){
        k = rot_list[i];
        dlg = vel[k][0]*vel[j][0] +  vel[k][1]*vel[j][1] + vel[k][2]*vel[j][2]; 
        corr[i] += dlg;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void StatisticVCorr::write_stat(bigint step)
{
  int i;
  double dt = update->dt;
  char f_name[FILENAME_MAX];

  if (st_accum <= dump_each){
    num_step = 0; 
    return;
  }

  if (comm->me == 0){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    for (i = 0; i < nwind; i++)
      fprintf(out_stat,"%g %15.10lf \n", corr_map[i]*dt, corr[i]/num_step);    
    fclose(out_stat);

    //num_step = 0;
  }  
}

/* ---------------------------------------------------------------------- */

