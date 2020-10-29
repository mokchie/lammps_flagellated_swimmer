/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// cosine angle potential

#include <cmath>
#include "angle_area_volume.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define  MOLINC 20

AngleAreaVolume::AngleAreaVolume(LAMMPS *lmp) : Angle(lmp) 
{
  dath = datt = NULL;
  n_mol_limit = MOLINC;
  memory->create(dath,2*n_mol_limit,"angle_area_volume:dath");
  memory->create(datt,2*n_mol_limit,"angle_area_volume:datt");
}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

AngleAreaVolume::~AngleAreaVolume()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(ka);
    memory->destroy(a0);
    memory->destroy(kv);
    memory->destroy(v0);
    memory->destroy(kl);
  }
  memory->destroy(dath);
  memory->destroy(datt);
}

/* ---------------------------------------------------------------------- */

void AngleAreaVolume::compute(int eflag, int vflag)
{
  int i1,i2,i3,n,j,type,kk,l;
  tagint m,m1;
  double d21x,d21y,d21z,d31x,d31y,d31z,d32x,d32y,d32z;
  double nx,ny,nz,nn,mx,my,mz,aa,vv,ar0, halfnn;
  double coefl, coefa, coefv, coefca;
  double eangle;
  double **fa = NULL;
  double **fv = NULL; 
  double xx1[3],xx2[3],xx3[3]; 
  int n_stress = output->n_stress;
  double ff[6];

  memory->create(fa,3,3,"angle_area_volume:fa");
  memory->create(fv,3,3,"angle_area_volume:fv");

  n_mol_max = atom->n_mol_max;
  if (n_mol_max > n_mol_limit){
    n_mol_limit = n_mol_max + MOLINC;
    memory->grow(dath,2*n_mol_limit,"angle_area_volume:dath");
    memory->grow(datt,2*n_mol_limit,"angle_area_volume:datt");    
  }

  eangle = 0.0;

  if(eflag || vflag) ev_setup(eflag, vflag);
  else evflag = 0;
  
  for (n = 0; n < 2*n_mol_max; n++){
    dath[n] = 0.0;
    datt[n] = 0.0;   
  }

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
    dath[m+n_mol_max] += vv;
  }
  MPI_Allreduce(dath,datt,2*n_mol_max,MPI_DOUBLE,MPI_SUM,world);

  if (atom->mol_corresp_ind)
    for (n = 0; n < atom->n_mol_corresp_glob; n++){
      m = atom->mol_corresp_glob[2*n+1] - 1;
      m1 = atom->mol_corresp_glob[2*n] - 1;
      datt[m] = datt[m1];
      datt[m+n_mol_max] = datt[m1+n_mol_max];
    }
 

  for (n = 0; n < nanglelist; n++) {
      
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];
    ar0 = anglelist_area[n];
    m = atom->molecule[i1]-1;

    // 2-1 distance
    d21x = x[i2][0] - x[i1][0];
    d21y = x[i2][1] - x[i1][1];
    d21z = x[i2][2] - x[i1][2];

    // 3-1 distance
    d31x = x[i3][0] - x[i1][0];
    d31y = x[i3][1] - x[i1][1];
    d31z = x[i3][2] - x[i1][2];

    // 3-2 distance
    d32x = x[i3][0] - x[i2][0];
    d32y = x[i3][1] - x[i2][1];
    d32z = x[i3][2] - x[i2][2];
    
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

    // calculate coeffs
    aa = 0.5*nn;
    coefl = 0.5*kl[type]*(ar0-aa)/ar0/nn; 
    coefa = 0.5*ka[type]*(a0[type]-datt[m])/a0[type]/nn;
    coefca = coefl + coefa;        
    coefv = kv[type]*(v0[type]-datt[m+n_mol_max])/v0[type]/18.0;  

    //calculate energy
    if (eflag){
      eangle = 0.5*kl[type]*(ar0-aa)*(ar0-aa)/ar0 + 
               0.5*ka[type]*(a0[type]-datt[m])*(a0[type]-datt[m])/a0[type] + 
               0.5*kv[type]*(v0[type]-datt[m+n_mol_max])*(v0[type]-datt[m+n_mol_max])/v0[type];
    }

    //calculate force for local and global area constraint
    fa[0][0] = coefca*(ny*d32z - nz*d32y);
    fa[0][1] = coefca*(nz*d32x - nx*d32z);    
    fa[0][2] = coefca*(nx*d32y - ny*d32x);
    fa[1][0] = coefca*(nz*d31y - ny*d31z);
    fa[1][1] = coefca*(nx*d31z - nz*d31x);
    fa[1][2] = coefca*(ny*d31x - nx*d31y);
    fa[2][0] = coefca*(ny*d21z - nz*d21y);
    fa[2][1] = coefca*(nz*d21x - nx*d21z);
    fa[2][2] = coefca*(nx*d21y - ny*d21x);
    
    //calculate force for volume constraint
    fv[0][0] = coefv*(nx + d32z*my - d32y*mz);
    fv[0][1] = coefv*(ny - d32z*mx + d32x*mz);    
    fv[0][2] = coefv*(nz + d32y*mx - d32x*my);
    fv[1][0] = coefv*(nx - d31z*my + d31y*mz);
    fv[1][1] = coefv*(ny + d31z*mx - d31x*mz);
    fv[1][2] = coefv*(nz - d31y*mx + d31x*my);
    fv[2][0] = coefv*(nx + d21z*my - d21y*mz);
    fv[2][1] = coefv*(ny - d21z*mx + d21x*mz);
    fv[2][2] = coefv*(nz + d21y*mx - d21x*my);
    
    
    // apply force to each of 3 atoms
    if (newton_bond || i1 < nlocal) {
      f[i1][0] += fa[0][0]+fv[0][0];
      f[i1][1] += fa[0][1]+fv[0][1];
      f[i1][2] += fa[0][2]+fv[0][2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] += fa[1][0]+fv[1][0];
      f[i2][1] += fa[1][1]+fv[1][1];
      f[i2][2] += fa[1][2]+fv[1][2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += fa[2][0]+fv[2][0];
      f[i3][1] += fa[2][1]+fv[2][1];
      f[i3][2] += fa[2][2]+fv[2][2];
    }

    //calculate virial      
    if (evflag) {
      vv = 0.0;
      kk = atom->mol_type[m];       
      if (kk > -1 && kk < atom->n_mol_types) 
        vv = 2.0*datt[m+n_mol_max]*coefv/atom->mol_size[kk];
      ev_tally2(i1,i2,i3,nlocal,newton_bond,eangle,fa,fv,vv,d21x,d21y,d21z,d31x,d31y,d31z,d32x,d32y,d32z);
    }

    if (n_stress){
      vv = 0.0;
      kk = atom->mol_type[m];
      if (kk > -1 && kk < atom->n_mol_types)
        vv = 2.0*datt[m+n_mol_max]*coefv/atom->mol_size[kk]; 
      ff[0] = d21x*fa[1][0] + d31x*fa[2][0] + (d21x*(fv[1][0]-fv[0][0]) + d31x*(fv[2][0]-fv[0][0]) + d32x*(fv[2][0]-fv[1][0]))*THIRD + vv;
      ff[1] = d21y*fa[1][1] + d31y*fa[2][1] + (d21y*(fv[1][1]-fv[0][1]) + d31y*(fv[2][1]-fv[0][1]) + d32y*(fv[2][1]-fv[1][1]))*THIRD + vv;
      ff[2] = d21z*fa[1][2] + d31z*fa[2][2] + (d21z*(fv[1][2]-fv[0][2]) + d31z*(fv[2][2]-fv[0][2]) + d32z*(fv[2][2]-fv[1][2]))*THIRD + vv;
      ff[3] = d21x*fa[1][1] + d31x*fa[2][1] + (d21x*(fv[1][1]-fv[0][1]) + d31x*(fv[2][1]-fv[0][1]) + d32x*(fv[2][1]-fv[1][1]))*THIRD;
      ff[4] = d21x*fa[1][2] + d31x*fa[2][2] + (d21x*(fv[1][2]-fv[0][2]) + d31x*(fv[2][2]-fv[0][2]) + d32x*(fv[2][2]-fv[1][2]))*THIRD;
      ff[5] = d21y*fa[1][2] + d31y*fa[2][2] + (d21y*(fv[1][2]-fv[0][2]) + d31y*(fv[2][2]-fv[0][2]) + d32y*(fv[2][2]-fv[1][2]))*THIRD;
      for (kk = 0; kk < n_stress; kk++){
        l = output->stress_id[kk];
        if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
          output->stat[l]->virial4(i1,i2,i3,ff);
      }
    } 
  }
  memory->destroy(fa);
  memory->destroy(fv);
}

/* ---------------------------------------------------------------------- */

void AngleAreaVolume::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;
  
  int individual = atom->individual;
  if(individual==0)
     error->all(FLERR,"Individual has wrong value or is not set! Using angle area/volume only possible with individual =1");
 
  memory->create(ka,n+1,"angle:ka");
  memory->create(a0,n+1,"angle:a0");
  memory->create(kv,n+1,"angle:kv");
  memory->create(v0,n+1,"angle:v0");
  memory->create(kl,n+1,"angle:kl");

  memory->create(setflag,n+1,"angle:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script for one time
------------------------------------------------------------------------- */

void AngleAreaVolume::coeff(int narg, char **arg)
{

  if (narg != 6) error->all(FLERR, "Incorrect args in angle_coeff command");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nangletypes,ilo,ihi);

  double ka_one = force->numeric(FLERR,arg[1]);
  double a0_one = force->numeric(FLERR,arg[2]);
  double kv_one = force->numeric(FLERR,arg[3]);
  double v0_one = force->numeric(FLERR,arg[4]);
  double kl_one = force->numeric(FLERR,arg[5]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    ka[i] = ka_one;
    a0[i] = a0_one;
    kv[i] = kv_one;
    v0[i] = v0_one;
    kl[i] = kl_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args in angle_coeff command");
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file 
------------------------------------------------------------------------- */

void AngleAreaVolume::write_restart(FILE *fp)
{
  fwrite(&ka[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&a0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&kv[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&v0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&kl[1],sizeof(double),atom->nangletypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void AngleAreaVolume::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0){ 
    fread(&ka[1],sizeof(double),atom->nangletypes,fp);
    fread(&a0[1],sizeof(double),atom->nangletypes,fp);
    fread(&kv[1],sizeof(double),atom->nangletypes,fp);
    fread(&v0[1],sizeof(double),atom->nangletypes,fp);
    fread(&kl[1],sizeof(double),atom->nangletypes,fp);
  }
  MPI_Bcast(&ka[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&a0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&kv[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&v0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&kl[1],atom->nangletypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleAreaVolume::write_data(FILE *fp)
{ 
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp,"%d %g %g %g %g %g\n",i,ka[i],a0[i],kv[i],v0[i],kl[i]);
}

/* ----------------------------------------------------------------------*/
