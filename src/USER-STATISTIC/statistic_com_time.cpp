/* ----------------------------------------------------------------------
  Kathrin Mueller - 31/03/15 only the one of the group is calculated
 
  LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

------------------------------------------------------------------------- */

#include "statistic_com_time.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

/* ---------------------------------------------------------------------- */
StatisticCOMTime::StatisticCOMTime(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  writetest = 0;
  init_on = 0;
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticCOMTime::~StatisticCOMTime()
{
  if (init_on){
    memory->sfree(com);
    memory->sfree(com_all);
    memory->sfree(com_vel);
    memory->sfree(com_vel_all);
    memory->sfree(mol_list);
    memory->sfree(cyl_com);

    if (!comm->me){
      memory->destroy(com_av);
      memory->destroy(com_vel_av);
    }
  }
}


/* ---------------------------------------------------------------------- */

void StatisticCOMTime::calc_stat()
{
  int j,k,l;
  tagint i, mol, imol;
  double xx[3];
  double **x = atom->x;
  double **v = atom->v;
  double hmax, hmin;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;		
  tagint n_mol = atom->n_mol_max;	// number of molecules
  tagint *n_atoms = atom->mol_size;	//number of atoms per molecule

  // initialising
  if (init_on == 0){

    tagint *mol_loc = new tagint[n_mol];

    mol_list = (tagint *) memory->smalloc(n_mol*sizeof(tagint),"statistic_com/time:mol_list");

    com = (double *) memory->smalloc(3*n_mol*sizeof(double),"statistic_com/time:com");
    com_vel = (double *) memory->smalloc(3*n_mol*sizeof(double),"statistic_com/time:com_vel");
    com_all = (double *) memory->smalloc(3*n_mol*sizeof(double),"statistic_com/time:com_all");
    com_vel_all = (double *) memory->smalloc(3*n_mol*sizeof(double),"statistic_com/time:com_vel_all");

    cyl_com = (double *) memory->smalloc(3*n_mol*sizeof(double),"statistic_com/time:com");

    for (i=0; i<n_mol; i++){
      mol_loc[i] = 0;
    }

    for (i=0; i<nlocal; i++){
      mol = atom->molecule[i];		// index of molecule the atom i belongs to
      if (mol && mask[i] & groupbit){	
        mol_loc[mol-1] = 1;		// 1 if molecule will be considered
      }
    }

    MPI_Allreduce(mol_loc,mol_list,n_mol,MPI_LMP_TAGINT,MPI_SUM,world);

    n_stat = 0;
    for (i=0; i<n_mol; i++) {
      if (mol_list[i]){
        mol_list[n_stat] = i;		//If molecule is in list, safe the number of this molecule
        n_stat++;			//n_stat is smaller or equal i
      }
   }

    // Arrays for the average
    if (!comm->me){
      memory->create(com_av,n_stat,3,"statistic_com/time:com_av");
      memory->create(com_vel_av,n_stat,3,"statistic_com/time:com_vel_av");

      for(i=0; i<n_stat; i++) {
        for(j=0; j<3; j++) {
         com_av[i][j] = 0.0;
         com_vel_av[i][j] = 0.0;
        }
      }
    }

    init_on = 1;
    delete [] mol_loc;
  } // 

// Set values to zero
    for(i=0; i<3*n_mol; i++) {
       com[i] = 0.0;
       cyl_com[i] = 0.0;
       com_vel[i] = 0.0;
       com_all[i] = 0.0;
       com_vel_all[i] = 0.0;
    }

  
//Averaging the com position and velcoities of the particles of the molecules 
  for (i=0; i<nlocal; i++){
    mol = atom->molecule[i];
    if (mol && mask[i] & groupbit){		//If molecule is in list, than calculate
      mol--;		//molecules start by 1, but arrays by 0
      domain->unmap(x[i],atom->image[i],xx);
      k = atom->mol_type[mol];
      if (k > -1 && k < atom->n_mol_types){
        com_vel[3*mol] += v[i][0]/n_atoms[k];	//vx is the same if cylinder or cartesian coord.
        for (j=0; j<3; j++)
          com[3*mol+j] += xx[j]/n_atoms[k];	//com of atom
      }
    }
  }

  MPI_Allreduce(com,com_all,3*n_mol,MPI_DOUBLE,MPI_SUM,world);

  //Calculation of radial position and pre-calculation for phi
  if (cyl_ind) {
    for (i=0; i<n_mol; i++){
      cyl_com[3*i] = sqrt((com_all[3*i+1]-ylo)*(com_all[3*i+1]-ylo) + (com_all[3*i+2]-zlo)*(com_all[3*i+2]-zlo)); //r
      if (cyl_com[3*i] > 0.0){
        cyl_com[3*i+1] = (com_all[3*i+1]-ylo)/cyl_com[3*i];	// y/r = cos(phi)
        cyl_com[3*i+2] = (com_all[3*i+2]-zlo)/cyl_com[3*i];	// z/r = sin(phi) 
      }
    }
  }

  //calculation of com velocity either in cartesian or in cylindrical coordinates
  for (i=0; i<nlocal; i++){
    mol = atom->molecule[i];
    if (mol && mask[i] & groupbit){		
      mol--;
      k = atom->mol_type[mol];
      if (k > -1 && k < atom->n_mol_types){
        if (cyl_ind) {
          com_vel[3*mol+1] += (v[i][1]*cyl_com[3*mol+1] + v[i][2]*cyl_com[3*mol+2])/n_atoms[k];	// vr = vy*cos(phi)+vz*sin(phi)
          com_vel[3*mol+2] += (v[i][2]*cyl_com[3*mol+1] - v[i][1]*cyl_com[3*mol+2])/n_atoms[k];	// vphi = vz*cos(phi)-vy*sin(phi)
        } else {
          for (j=1; j<3; j++)
            com_vel[3*mol+j] += v[i][j]/n_atoms[k];
        }
      }
    }
  }
  
  MPI_Allreduce(com_vel,com_vel_all,3*n_mol,MPI_DOUBLE,MPI_SUM,world); 

  if (!comm->me){
    
    if (cyl_ind) {
      for(i = 0; i<n_stat; i++)
      {
       // shift_coordinates(com_all[3*mol_list[i]], com_all[3*mol_list[i]+1],com_all[3*mol_list[i]+2]);
        com_av[i][0] += com_all[3*mol_list[i]]; // Cylinder has to be in x-direction
        com_av[i][1] += cyl_com[3*mol_list[i]];	// Radial position

        //Calculation of the angle phi defined in the interval [0,2pi)
        // Is the com located on +y-axis phi is zero; on +z-axis phi is pi/2
        // On -y-axis phi is pi, on -z phi is 3/2 pi
        if(com_all[3*mol_list[i]+2] >= 0) {
          com_av[i][2] += acos(cyl_com[3*mol_list[i]+1]);	//acos(y/r) if z>=0
        } else {
          com_av[i][2] += (2.0*M_PI-acos(cyl_com[3*mol_list[i]+1]));	//2pi-acos(y/r) if z<0
        }

        for(j=0; j<3; j++) {
          com_vel_av[i][j] += com_vel_all[3*mol_list[i]+j];
        }
      }
    } else {
      for(i = 0; i<n_stat; i++)
      {
        //shift_coordinates(com_all[3*mol_list[i]], com_all[3*mol_list[i]+1],com_all[3*mol_list[i]+2]);
        for(j=0; j<3; j++) { 
          com_av[i][j] += com_all[3*mol_list[i]+j];
          com_vel_av[i][j] += com_vel_all[3*mol_list[i]+j];
        }
      }
    }
  }

  num_step++;
}

/* ---------------------------------------------------------------------- */

void StatisticCOMTime::write_stat(bigint step)
{
  int i,j;
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
      fprintf(out_stat,"#Step, for "TAGINT_FORMAT" molecule(s), comx, comy, comz, com_velx, com_vely, com_velz \n", n_stat);
      writetest = 1;
    }
    fprintf(out_stat, BIGINT_FORMAT, step);
    for(i=0; i<n_stat; i++){
      fprintf(out_stat," %lf %lf %lf %lf %lf %lf ",com_av[i][0]*num_steppm1, com_av[i][1]*num_steppm1, com_av[i][2]*num_steppm1, com_vel_av[i][0]*num_steppm1, com_vel_av[i][1]*num_steppm1, com_vel_av[i][2]*num_steppm1);
      for(j = 0; j<3; j++) {
        com_av[i][j] = 0.0;
        com_vel_av[i][j] = 0.0;
      }
    }
    fprintf(out_stat, "\n");
    fclose(out_stat);
  }

  num_step = 0;
}

/* ---------------------------------------------------------------------- */

void StatisticCOMTime::shift_coordinates(double& x, double& y, double& z)
{
  

  if (x<dxlo || x>=dxhi || y<dylo || y>=dyhi || z<dzlo || z>=dzhi){
    if (xper) {
      while (x >= dxhi){
        x -= dxs;
        if (comm->le){
          y -= comm->shift;
          while (y >= dyhi)
            y -= dys;
          while (y < dylo)
            y += dys;
        }
      }
      while (x < dxlo){
        x += dxs;
        if (comm->le){
          y += comm->shift;
          while (y >= dyhi)
            y -= dys;
          while (y < dylo)
            y += dys;
        }
      }
    }
    if (yper) {
      while (y >= dyhi)
        y -= dys;
      while (y < dylo)
        y += dys;
    }
    if (zper) {
      while (z >= dzhi)
        z -= dzs;
      while (z < dzlo)
        z += dzs;
    }
  }
}

/* ---------------------------------------------------------------------- */
