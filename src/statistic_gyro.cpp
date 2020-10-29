/* ----------------------------------------------------------------------
  Kathrin Mueller - 23/11/11 only the one of the group is calculated

  Dmitry Fedosov - 08/12/05  accumulation of statistics
 
  LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

------------------------------------------------------------------------- */

#include "statistic_gyro.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticGyro::StatisticGyro(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  int i,j;
  
  writetest = 0;  
  cyl_ind = 0; 
                  
  init_on = 0;
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticGyro::~StatisticGyro()
{
  if (init_on){
    memory->sfree(rad);
    memory->sfree(c_m);
    memory->sfree(c_mt);
    memory->sfree(mol_list);
  }
}


/* ---------------------------------------------------------------------- */

void StatisticGyro::calc_stat()
{
  int j,k;
  tagint i,l;
  double xx[3];
  double **x = atom->x;
  tagint *n_atoms = atom->mol_size;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint ind_h[atom->n_mol_max];
  tagint mol;
  
  if (init_on == 0){
    nm = atom->n_mol_max;
    rad = (double *) memory->smalloc(3*nm*sizeof(double),"statistic_gyro:rad");
    c_m = (double *) memory->smalloc(3*nm*sizeof(double),"statistic_gyro:c_m");
    c_mt = (double *) memory->smalloc(3*nm*sizeof(double),"statistic_gyro:c_mt");
    mol_list = (tagint *) memory->smalloc(nm*sizeof(tagint),"statistic_gyro:mol_list");
    
    for (i=0; i<nm; ++i){
      ind_h[i] = 0;  
      mol_list[i] = 0; 
    }
    for (j=0; j<nlocal; ++j){
      mol = atom->molecule[j]; 
      if (mol && mask[j] & groupbit)
        ind_h[mol-1] = 1;
    }
    MPI_Allreduce(&ind_h,mol_list,nm,MPI_LMP_TAGINT,MPI_SUM,world);
    nm_h = 0;
    for (i=0; i<nm; ++i)
      if (mol_list[i]){
        mol_list[nm_h] = i;
        nm_h++;
      }

    for(i=0; i<3*nm; ++i)
      rad[i] = 0.0;   

    init_on = 1;   
  }

  for(i=0; i<3*nm; ++i) {   
    c_m[i] = 0.0;
    c_mt[i] = 0.0;
  }

  for (j=0; j<nlocal; ++j){
    mol = atom->molecule[j];     
    if (mol && mask[j] & groupbit){
      domain->unmap(x[j],atom->image[j],xx);
      l = 3*(mol-1);
      for (k=0; k<3; ++k) 
        c_m[l + k] += xx[k];
    }
  }  

  for (i=0; i<nm; ++i){
    k = atom->mol_type[i];
    if (k > -1 && k < atom->n_mol_types)
      for (j=0; j<3; j++) 
        c_m[3*i+j] /= n_atoms[k];
  }
  
  MPI_Allreduce(c_m,c_mt,3*nm,MPI_DOUBLE,MPI_SUM,world);

  for (i=0; i<nlocal; i++){
    mol = atom->molecule[i];   
    if (mol && mask[i] & groupbit){
      domain->unmap(x[i],atom->image[i],xx);
      l = 3*(mol-1);
      for (j=0; j<3; j++) 
        rad[l + j] += (xx[j]-c_mt[l + j])*(xx[j]-c_mt[l + j]);
    }
  }
  
  num_step++;

}

/* ---------------------------------------------------------------------- */

void StatisticGyro:: write_stat(bigint step)
{
  int j,k;
  tagint i,l;
  double *rtmp, *tmp;
  double radd[3],val,x,y;
  char f_name[FILENAME_MAX];

  rtmp = (double *) memory->smalloc(3*nm*sizeof(double),"statistic_gyro:rtmp");
  tmp = (double *) memory->smalloc(3*nm*sizeof(double),"statistic_gyro:tmp");

  for (j=0; j<3; j++)
    radd[j] = 0.0;
  for (i=0; i<3*nm; i++){
    l = static_cast<tagint> (i/3.0);
    k = atom->mol_type[l];
    if (k > -1 && k < atom->n_mol_types)
      tmp[i] = rad[i]/num_step/atom->mol_size[k];
    rad[i] = 0.0;
    rtmp[i] = 0.0; 
  }
  MPI_Reduce(tmp,rtmp,3*nm,MPI_DOUBLE,MPI_SUM,0,world); 

  if (!(comm->me)){
    for (i=0; i<nm_h; i++)
      for (j=0; j<3; j++)
        radd[j] += rtmp[3*i + j]/nm_h;
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    
    if(writetest == 0) {
      out_stat=fopen(f_name, "w");
      fprintf(out_stat,"Step, for all " TAGINT_FORMAT " molecule, then average radius of gyration squared (x,y,z,tot) \n", nm_h);
      writetest = 1;
    }
    else
      out_stat=fopen(f_name, "a");
    
    fprintf(out_stat, BIGINT_FORMAT, step);
    for(i=0; i<nm_h; i++)
      fprintf(out_stat," %lf %lf %lf %lf",rtmp[3*i],rtmp[3*i+1],rtmp[3*i+2],rtmp[3*i]+rtmp[3*i+1]+rtmp[3*i+2] );    
    fprintf(out_stat," %lf %lf %lf %lf \n",radd[0],radd[1],radd[2],radd[0]+radd[1]+radd[2]);


    fclose(out_stat);
  }    
  memory->sfree(rtmp);
  memory->sfree(tmp);
  
  num_step = 0;    
}

/* ---------------------------------------------------------------------- */

