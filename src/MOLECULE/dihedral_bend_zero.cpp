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

#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include "dihedral_bend_zero.h"
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

/* ---------------------------------------------------------------------- */

DihedralBendZero::DihedralBendZero(LAMMPS *lmp) : Dihedral(lmp) {}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

DihedralBendZero::~DihedralBendZero()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
  }
}

/* ---------------------------------------------------------------------- */

void DihedralBendZero::compute(int eflag, int vflag)
{
  int n,i1,i2,i3,i4,type,kk,l;
  double d21x,d21y,d21z,d31x,d31y,d31z,d32x,d32y,d32z;
  double d34x,d34y,d34z,d24x,d24y,d24z,d14x,d14y,d14z;
  double n1x,n1y,n1z,n2x,n2y,n2z,n1,n2,nn;
  double costheta,mx,ar1,ar2,arb,lsb;
  double alfa,a11,a12,a22,coefa1,coefa2,alfal,cc;
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
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < ndihedrallist; n++) {

    i1 = dihedrallist[n][0];
    i2 = dihedrallist[n][1];
    i3 = dihedrallist[n][2];
    i4 = dihedrallist[n][3];
    type = dihedrallist[n][4];

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

    // cos(theta) calculation 
    costheta = (n1x*n2x + n1y*n2y + n1z*n2z)/nn; 
    if (costheta > 1.0) costheta = 1.0;
    if (costheta < -1.0) costheta = -1.0;
    mx = 1.0 - costheta;    
 
    // coeffs calculation
    alfa = 0.75*k[type]*lsb/arb;
    a11 = -alfa*costheta/n1;
    a12 = alfa/nn;
    a22 = -alfa*costheta/n2;
    alfal = 1.5*k[type]*mx/arb;
    cc = 0.375*k[type]*lsb*mx/arb/arb;
    coefa1 = cc/ar1;
    coefa2 = cc/ar2;

    // forces calculation
    f1[0] = (a11 + coefa1)*(n1y*d32z - n1z*d32y) + a12*(n2y*d32z - n2z*d32y);
    f1[1] = (a11 + coefa1)*(n1z*d32x - n1x*d32z) + a12*(n2z*d32x - n2x*d32z);
    f1[2] = (a11 + coefa1)*(n1x*d32y - n1y*d32x) + a12*(n2x*d32y - n2y*d32x);
    f2[0] = (a11 + coefa1)*(n1z*d31y - n1y*d31z) + (a22 + coefa2)*(n2y*d34z - n2z*d34y) +  
          a12*(n2z*d31y - n2y*d31z + n1y*d34z - n1z*d34y) + alfal*d32x;
    f2[1] = (a11 + coefa1)*(n1x*d31z - n1z*d31x) + (a22 + coefa2)*(n2z*d34x - n2x*d34z) +  
          a12*(n2x*d31z - n2z*d31x + n1z*d34x - n1x*d34z) + alfal*d32y;
    f2[2] = (a11 + coefa1)*(n1y*d31x - n1x*d31y) + (a22 + coefa2)*(n2x*d34y - n2y*d34x) +  
          a12*(n2y*d31x - n2x*d31y + n1x*d34y - n1y*d34x) + alfal*d32z;
    f3[0] = (a11 + coefa1)*(n1y*d21z - n1z*d21y) + (a22 + coefa2)*(n2z*d24y - n2y*d24z) +  
          a12*(n2y*d21z - n2z*d21y + n1z*d24y - n1y*d24z) - alfal*d32x;     
    f3[1] = (a11 + coefa1)*(n1z*d21x - n1x*d21z) + (a22 + coefa2)*(n2x*d24z - n2z*d24x) +  
          a12*(n2z*d21x - n2x*d21z + n1x*d24z - n1z*d24x) - alfal*d32y;
    f3[2] = (a11 + coefa1)*(n1x*d21y - n1y*d21x) + (a22 + coefa2)*(n2y*d24x - n2x*d24y) +  
          a12*(n2x*d21y - n2y*d21x + n1y*d24x - n1x*d24y) - alfal*d32z;
    f4[0] = (a22 + coefa2)*(n2z*d32y - n2y*d32z) + a12*(n1z*d32y - n1y*d32z);
    f4[1] = (a22 + coefa2)*(n2x*d32z - n2z*d32x) + a12*(n1x*d32z - n1z*d32x);
    f4[2] = (a22 + coefa2)*(n2y*d32x - n2x*d32y) + a12*(n1y*d32x - n1x*d32y);

    if (eflag)
      edihedral = alfa*mx;

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

void DihedralBendZero::allocate()
{
  allocated = 1;
  int n = atom->ndihedraltypes;

  memory->create(k,n+1,"dihedral:k");

  memory->create(setflag,n+1,"dihedral:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script
------------------------------------------------------------------------- */

void DihedralBendZero::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Incorrect args in dihedral_coeff command");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->ndihedraltypes,ilo,ihi);

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

void DihedralBendZero::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->ndihedraltypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void DihedralBendZero::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->ndihedraltypes,fp);
  }
  MPI_Bcast(&k[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->ndihedraltypes; i++) setflag[i] = 1;
}
