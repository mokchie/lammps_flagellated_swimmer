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

// harmonic dihedral potential

#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include "dihedral_bend.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

DihedralBend::DihedralBend(LAMMPS *lmp) : Dihedral(lmp) {}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

DihedralBend::~DihedralBend()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
  }
}

/* ---------------------------------------------------------------------- */

void DihedralBend::compute(int eflag, int vflag)
{
  int n,i1,i2,i3,i4,type,kk,l;
  double d21x,d21y,d21z,d31x,d31y,d31z,d32x,d32y,d32z;
  double d34x,d34y,d34z,d24x,d24y,d24z,d14x,d14y,d14z;
  double n1x,n1y,n1z,n2x,n2y,n2z,n1,n2,nn;
  double costheta,sintheta,mx,theta0,ar1,ar2,arb,lsb,gx,gy,gz,gg;
  double alfa,a11,a12,a13,a22,cc,alfh,alfal,chh,coefa1,coefa2;
  double f1[3],f2[3],f3[3],f4[3];
  double edihedral;
  int n_stress = output->n_stress;
  double ff[6];

  if(eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **dihedrallist = neighbor->dihedrallist;
  int ndihedrallist = neighbor->ndihedrallist;
  double *dihedrallist_angle = neighbor->dihedrallist_angle;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < ndihedrallist; n++) {

    i1 = dihedrallist[n][0];
    i2 = dihedrallist[n][1];
    i3 = dihedrallist[n][2];
    i4 = dihedrallist[n][3];
    type = dihedrallist[n][4];
    theta0 = dihedrallist_angle[n];

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

    // 3-4 distance
    d34x = x[i3][0] - x[i4][0];
    d34y = x[i3][1] - x[i4][1];
    d34z = x[i3][2] - x[i4][2];

    // 2-4 distance
    d24x = x[i2][0] - x[i4][0];
    d24y = x[i2][1] - x[i4][1];
    d24z = x[i2][2] - x[i4][2];

    // 1-4 distance
    d14x = x[i1][0] - x[i4][0];
    d14y = x[i1][1] - x[i4][1];
    d14z = x[i1][2] - x[i4][2];
    
    // calculate normals
    n1x = d21y*d31z - d31y*d21z;
    n1y = d31x*d21z - d21x*d31z;
    n1z = d21x*d31y - d31x*d21y;
    n2x = d34y*d24z - d24y*d34z;
    n2y = d24x*d34z - d34x*d24z;
    n2z = d34x*d24y - d24x*d34y;
    n1 = n1x*n1x + n1y*n1y + n1z*n1z;
    n2 = n2x*n2x + n2y*n2y + n2z*n2z;
    ar1 = sqrt(n1);
    ar2 = sqrt(n2);
    nn = ar1*ar2;
    arb = 0.5*(ar1+ar2);
    lsb = d32x*d32x + d32y*d32y + d32z*d32z;
    gx = n1y*n2z - n1z*n2y;
    gy = n1z*n2x - n1x*n2z;
    gz = n1x*n2y - n1y*n2x;
    gg = sqrt(gx*gx + gy*gy + gz*gz);
    if (gg > 0.0){
      gx = gx/gg;
      gy = gy/gg;
      gz = gz/gg;
    } else {
      gx = 0.0;
      gy = 0.0;
      gz = 0.0;
    }

    // cos(theta) and sin(theta) calculation 
    costheta = (n1x*n2x + n1y*n2y + n1z*n2z)/nn;
    if (costheta > 1.0) costheta = 1.0;
    if (costheta < -1.0) costheta = -1.0;
    sintheta = sqrt(1.0-costheta*costheta);
    cc = 1.0;
    mx = (n1x-n2x)*d14x + (n1y-n2y)*d14y + (n1z-n2z)*d14z;
    if (mx < 0) cc = -1.0;
    alfa = -costheta*cos(theta0) - cc*sintheta*sin(theta0);
    mx = 1.0 - alfa;

    // coeffs calculation
    alfh = 0.75*k[type]*lsb/arb;
    a11 = alfh*alfa/n1;
    a12 = alfh*cos(theta0)/nn;
    a22 = alfh*alfa/n2;
    a13 = alfh*sin(theta0)*cc/nn;
    alfal = 1.5*k[type]*mx/arb;
    chh = 0.375*k[type]*lsb*mx/arb/arb;
    coefa1 = chh/ar1;
    coefa2 = chh/ar2;

    // forces calculation
    f1[0] = (a11 + coefa1)*(n1y*d32z - n1z*d32y) + a12*(n2y*d32z - n2z*d32y) + a13*(gx*(n2z*d32z + n2y*d32y) - gy*n2x*d32y - gz*n2x*d32z);
    f1[1] = (a11 + coefa1)*(n1z*d32x - n1x*d32z) + a12*(n2z*d32x - n2x*d32z) + a13*(-gx*n2y*d32x + gy*(n2x*d32x + n2z*d32z) - gz*n2y*d32z);
    f1[2] = (a11 + coefa1)*(n1x*d32y - n1y*d32x) + a12*(n2x*d32y - n2y*d32x) + a13*(-gx*n2z*d32x - gy*n2z*d32y + gz*(n2y*d32y + n2x*d32x));
    f2[0] = (a11 + coefa1)*(n1z*d31y - n1y*d31z) + (a22 + coefa2)*(n2y*d34z - n2z*d34y) + a12*(n2z*d31y - n2y*d31z + n1y*d34z - n1z*d34y) + alfal*d32x + a13*(-gx*(n2z*d31z + n1y*d34y + n2y*d31y + n1z*d34z) + gy*(n2x*d31y + n1x*d34y) + gz*(n1x*d34z + n2x*d31z));
    f2[1] = (a11 + coefa1)*(n1x*d31z - n1z*d31x) + (a22 + coefa2)*(n2z*d34x - n2x*d34z) + a12*(n2x*d31z - n2z*d31x + n1z*d34x - n1x*d34z) + alfal*d32y + a13*(gx*(n1y*d34x + n2y*d31x) - gy*(n2x*d31x + n1z*d34z + n2z*d31z + n1x*d34x) + gz*(n2y*d31z + n1y*d34z));
    f2[2] = (a11 + coefa1)*(n1y*d31x - n1x*d31y) + (a22 + coefa2)*(n2x*d34y - n2y*d34x) + a12*(n2y*d31x - n2x*d31y + n1x*d34y - n1y*d34x) + alfal*d32z + a13*(gx*(n2z*d31x + n1z*d34x) + gy*(n1z*d34y + n2z*d31y) - gz*(n2y*d31y + n1x*d34x + n2x*d31x + n1y*d34y));
    f3[0] = (a11 + coefa1)*(n1y*d21z - n1z*d21y) + (a22 + coefa2)*(n2z*d24y - n2y*d24z) + a12*(n2y*d21z - n2z*d21y + n1z*d24y - n1y*d24z) - alfal*d32x + a13*(gx*(n2z*d21z + n1y*d24y + n2y*d21y + n1z*d24z) - gy*(n2x*d21y + n1x*d24y) - gz*(n1x*d24z + n2x*d21z));
    f3[1] = (a11 + coefa1)*(n1z*d21x - n1x*d21z) + (a22 + coefa2)*(n2x*d24z - n2z*d24x) + a12*(n2z*d21x - n2x*d21z + n1x*d24z - n1z*d24x) - alfal*d32y + a13*(-gx*(n1y*d24x + n2y*d21x) + gy*(n2x*d21x + n1z*d24z + n2z*d21z + n1x*d24x) - gz*(n2y*d21z + n1y*d24z));
    f3[2] = (a11 + coefa1)*(n1x*d21y - n1y*d21x) + (a22 + coefa2)*(n2y*d24x - n2x*d24y) + a12*(n2x*d21y - n2y*d21x + n1y*d24x - n1x*d24y) - alfal*d32z + a13*(-gx*(n2z*d21x + n1z*d24x) - gy*(n1z*d24y + n2z*d21y) + gz*(n2y*d21y + n1x*d24x + n2x*d21x + n1y*d24y));
    f4[0] = (a22 + coefa2)*(n2z*d32y - n2y*d32z) + a12*(n1z*d32y - n1y*d32z) + a13*(gx*(n1y*d32y + n1z*d32z) - gy*n1x*d32y - gz*n1x*d32z);
    f4[1] = (a22 + coefa2)*(n2x*d32z - n2z*d32x) + a12*(n1x*d32z - n1z*d32x) + a13*(-gx*n1y*d32x + gy*(n1z*d32z + n1x*d32x) - gz*n1y*d32z);
    f4[2] = (a22 + coefa2)*(n2y*d32x - n2x*d32y) + a12*(n1y*d32x - n1x*d32y) + a13*(-gx*n1z*d32x - gy*n1z*d32y + gz*(n1x*d32x + n1y*d32y));

    if (eflag)
      edihedral = 0.75*k[type]*lsb*mx/arb;

    // apply force to each of 4 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] += f2[0];
      f[i2][1] += f2[1];
      f[i2][2] += f2[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (newton_bond || i4 < nlocal) {
      f[i4][0] += f4[0];
      f[i4][1] += f4[1];
      f[i4][2] += f4[2];
    }

    if(evflag)
      ev_tally2(i1,i2,i3,i4,nlocal,newton_bond,edihedral,f1,f2,f4,
               d31x,d31y,d31z,d32x,d32y,d32z,d34x,d34y,d34z);

    if (n_stress){
      ff[0] = -d31x*f1[0] - d32x*f2[0] - d34x*f4[0];
      ff[1] = -d31y*f1[1] - d32y*f2[1] - d34y*f4[1];
      ff[2] = -d31z*f1[2] - d32z*f2[2] - d34z*f4[2];
      ff[3] = -d31x*f1[1] - d32x*f2[1] - d34x*f4[1];
      ff[4] = -d31x*f1[2] - d32x*f2[2] - d34x*f4[2];
      ff[5] = -d31y*f1[2] - d32y*f2[2] - d34y*f4[2];
      for (kk = 0; kk < n_stress; kk++){
        l = output->stress_id[kk];
        if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
          output->stat[l]->virial5(i1,i2,i3,i4,ff);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void DihedralBend::allocate()
{
  allocated = 1;
  int n = atom->ndihedraltypes;

  int individual = atom->individual;
  if(individual==0)
    error->all(FLERR,"Individual has wrong value or is not set! Using dihedral bend only possible with individual =1");  

  memory->create(k,n+1,"dihedral:k");

  memory->create(setflag,n+1,"dihedral:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script
------------------------------------------------------------------------- */

void DihedralBend::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Incorrect args in dihedral_coeff command");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(arg[0],atom->ndihedraltypes,ilo,ihi);

  double k_one = force->numeric(FLERR,arg[1]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args in dihedral_coeff command");
}


/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file 
------------------------------------------------------------------------- */

void DihedralBend::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->ndihedraltypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void DihedralBend::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->ndihedraltypes,fp);
  }
  MPI_Bcast(&k[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->ndihedraltypes; i++) setflag[i] = 1;
}
