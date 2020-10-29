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

#include "statistic_extension.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

/* ---------------------------------------------------------------------- */
StatisticExtension::StatisticExtension(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  writetest = 0;
  cyl_ind = 0;
  init_on = 0;
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticExtension::~StatisticExtension()
{
  if (init_on){
    memory->sfree(ext);
    memory->sfree(ext_av);
    memory->sfree(c_max);
    memory->sfree(c_min);
    memory->sfree(mol_list);
  }
}


/* ---------------------------------------------------------------------- */

void StatisticExtension::calc_stat()
{
  int j,k,l;
  tagint i, mol, imol;
  double xx[3];
  double **x = atom->x;
  double hmax, hmin;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint n_mol = atom->n_mol_max;

  if (init_on == 0){
    
    tagint *mol_all = new tagint[n_mol];
    tagint *mol_loc = new tagint[n_mol];
     
    for (i=0; i<n_mol; i++){
      mol_all[i] = 0;  
      mol_loc[i] = 0;
    }

    for (i=0; i<nlocal; i++){
      mol = atom->molecule[i];
      if (mol && mask[i] & groupbit){
        
        mol_loc[mol-1] = 1;
      }
    }
    mol_list = (tagint *) memory->smalloc(n_mol*sizeof(tagint),"statistic_extension:mol_list");
    MPI_Allreduce(mol_loc,mol_all,n_mol,MPI_LMP_TAGINT,MPI_SUM,world);
    
    n_stat = 0;
    for (i=0; i<n_mol; i++) {
      if (mol_all[i]){
        mol_list[i] = n_stat;
        n_stat++;
      }
      else
        mol_list[i] = -1;
   }

    ext_av = (double *) memory->smalloc(3*n_stat*sizeof(double),"statistic_extension:ext_av");
    ext = (double *) memory->smalloc(3*n_stat*sizeof(double),"statistic_extension:ext");

    memory->create(c_max,n_stat,3,"statistic_extension:c_max");
    memory->create(c_min,n_stat,3,"statistic_extension:c_min");

    for(i=0; i<3*n_stat; i++)
      ext_av[i] = 0.0;

    init_on = 1;
    delete [] mol_loc;
    delete [] mol_all;
  }

  for(i=0; i<n_stat; i++)
    for(j=0; j<3; j++){ 
    c_max[i][j] = -1.0e250;
    c_min[i][j] = 1.0e250;
  }
  
//calculate minimum and maximum value of coordinates
  for (i=0; i<nlocal; i++){
    mol = atom->molecule[i];
    imol = mol_list[mol-1];
    if (mol && imol >=0){
      domain->unmap(x[i],atom->image[i],xx);
      for (j=0; j<3; j++) {
        c_max[imol][j]  = MAX(c_max[imol][j], xx[j]);
        c_min[imol][j]  = MIN(c_min[imol][j], xx[j]);
      }
    }
  }
  for(imol = 0; imol<n_stat; imol++)
  {
    for(j=0; j<3; j++) {
       hmax = c_max[imol][j];
       hmin = c_min[imol][j];
       MPI_Allreduce(&hmax,&all_max,1,MPI_DOUBLE,MPI_MAX,world);
       MPI_Allreduce(&hmin,&all_min,1,MPI_DOUBLE,MPI_MIN,world);
       ext_av[3*imol+j] += all_max - all_min;
       ext[3*imol+j] = all_max - all_min;
    }
  }
  num_step++;
}

/* ---------------------------------------------------------------------- */

void StatisticExtension:: write_stat(bigint step)
{
  int imol,j;
  char f_name[FILENAME_MAX];
  double num_steppm1 = 1.0/num_step;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s.plt",fname);
    
    if(writetest == 0) {
      out_stat=fopen(f_name, "w");
    }
    else
      out_stat=fopen(f_name, "a");
    
    if(writetest == 0) {
      fprintf(out_stat,"#Step, for all "TAGINT_FORMAT" molecule, maximum extension and average in x, y, z direction\n", n_stat);
      writetest = 1;
    }
    fprintf(out_stat, BIGINT_FORMAT, step);
    for(imol=0; imol<n_stat; imol++){
      fprintf(out_stat," %lf %lf %lf %lf %lf %lf ",ext[3*imol], ext[3*imol+1], ext[3*imol+2], ext_av[3*imol]*num_steppm1, ext_av[3*imol+1]*num_steppm1, ext_av[3*imol+2]*num_steppm1);
      for(j = 0; j<3; j++)
        ext_av[3*imol+j] = 0.0;
    }
    fprintf(out_stat, "\n");
    fclose(out_stat);
  }    

  num_step = 0;
}

/* ---------------------------------------------------------------------- */

