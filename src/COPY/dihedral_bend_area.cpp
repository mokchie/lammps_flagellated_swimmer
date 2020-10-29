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

// bend/dual dihedral potential with spontaneous curvature

#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include "dihedral_bend_area.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001
#define NV    10

/* ---------------------------------------------------------------------- 
Syntax: dihedral_style bend/area
        dihedral_coeff ID kappa

kappa: bending rigidity
-------------------------------------------------------------------------*/

/* ---------------------------------------------------------------------- */

DihedralBendArea::DihedralBendArea(LAMMPS *lmp) : Dihedral(lmp)
{
  k = A_ij = n_sum = n_i = z_i = nr_ij = NULL;
  n_ij = r_ij = r_ik = r_jk = r_il = r_jl = dA_ij = R_ij = M_ij = fh = NULL;
}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

DihedralBendArea::~DihedralBendArea()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(A_ij);
    memory->destroy(n_sum);
    memory->destroy(n_i);
    memory->destroy(z_i);
    memory->destroy(nr_ij);
    memory->destroy(n_ij);
    memory->destroy(r_ij);
    memory->destroy(r_ik);
    memory->destroy(r_jk);
    memory->destroy(r_il);
    memory->destroy(r_jl);
    memory->destroy(dA_ij);
    memory->destroy(R_ij);
    memory->destroy(M_ij);
    memory->destroy(fh);
  }
}

/* ---------------------------------------------------------------------- */

void DihedralBendArea::compute(int eflag, int vflag)
{
  int i0,i1,i2,i3,type,i_st,i_fin,at,vt;
  int i,l,m,n;
  double H0,H_i,A_i,D_i,nz_i,cc1,cc2;
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

  i_st = i_fin = 0;
  while (i_fin < ndihedrallist){
    
    i1 = dihedrallist[i_fin][1];
    if (i_st == i_fin){
      at = i1;
      i_fin++;
      continue;
    } else {
      if (i1 == at) {
	i_fin++;
	if (i_fin < ndihedrallist) continue;
      }
    }

    type = dihedrallist[i_st][4];
    H0 = dihedrallist_angle[i_st];
    vt = i_fin - i_st;

    for (m = 0; m < 3; m++){
       n_sum[m] = 0.0;
       z_i[m] = 0.0; 
    }
    
    l = 0;
    A_i = 0.0;
    for (n = i_st; n < i_fin; n++) {
      i0 = dihedrallist[n][0];
      i1 = dihedrallist[n][1];
      i2 = dihedrallist[n][2];
      i3 = dihedrallist[n][3];

      r_ij[l][3] = n_ij[l][3] = 0.0;
      for (m = 0; m < 3; m++){
        r_ij[l][m] = x[i1][m] - x[i2][m];
	r_ij[l][3] += r_ij[l][m]*r_ij[l][m];
	r_ik[l][m] = x[i1][m] - x[i0][m];
	r_jk[l][m] = x[i2][m] - x[i0][m];
      }
      
      n_ij[l][0] = THIRD*(r_jk[l][1]*r_ik[l][2] - r_ik[l][1]*r_jk[l][2]);
      n_ij[l][1] = THIRD*(r_ik[l][0]*r_jk[l][2] - r_jk[l][0]*r_ik[l][2]);
      n_ij[l][2] = THIRD*(r_jk[l][0]*r_ik[l][1] - r_ik[l][0]*r_jk[l][1]);

      for (m = 0; m < 3; m++){
	n_ij[l][3] += n_ij[l][m]*n_ij[l][m];
	n_sum[m] += n_ij[l][m];
      }
      n_ij[l][4] = sqrt(n_ij[l][3]);
      A_ij[l] = n_ij[l][4];

      l++;
    }

    for (l = 0; l < vt; l++){
      i = (l+1)%vt;
      A_ij[l] += n_ij[i][4];
      A_i += A_ij[l];
      for (m = 0; m < 3; m++){
	r_il[l][m] = r_ij[i][m];
	r_jl[l][m] = -r_jk[i][m];
	z_i[m] += A_ij[l]*r_ij[l][m]/r_ij[l][3];
	n_ij[l][m] /= n_ij[i][4];
      }
    }
      
    A_i *= 0.25;
    n_sum[3] = 0.0;
    for (m = 0; m < 3; m++)
      n_sum[3] += n_sum[m]*n_sum[m];
    n_sum[4] = sqrt(n_sum[3]);

    D_i = nz_i = 0.0;
    for (m = 0; m < 3; m++){
      n_i[m] = n_sum[m]/n_sum[4];
      D_i += n_i[m]*z_i[m];
      nz_i += n_sum[m]*z_i[m]; 
    }
    nz_i /= n_sum[3];
    H_i = D_i/A_i;
    cc1 = -0.125*k[type]*(H0*H0 - H_i*H_i);
    cc2 = -k[type]*(H_i - H0); 

    l = 0;
    for (n = i_st; n < i_fin; n++) {
      i0 = dihedrallist[n][0];
      i1 = dihedrallist[n][1];
      i2 = dihedrallist[n][2];
      i3 = dihedrallist[n][3];
       
      i = (l+1)%vt;
      dA_ij[0][0] = r_ij[l][2]*n_ij[l][1] - r_ij[l][1]*n_ij[l][2];
      dA_ij[0][1] = r_ij[l][0]*n_ij[l][2] - r_ij[l][2]*n_ij[l][0];
      dA_ij[0][2] = r_ij[l][1]*n_ij[l][0] - r_ij[l][0]*n_ij[l][1];
      dA_ij[1][0] = r_jk[l][2]*n_ij[l][1] - r_jk[l][1]*n_ij[l][2] - r_jl[l][2]*n_ij[i][1] + r_jl[l][1]*n_ij[i][2];
      dA_ij[1][1] = r_jk[l][0]*n_ij[l][2] - r_jk[l][2]*n_ij[l][0] - r_jl[l][0]*n_ij[i][2] + r_jl[l][2]*n_ij[i][0];
      dA_ij[1][2] = r_jk[l][1]*n_ij[l][0] - r_jk[l][0]*n_ij[l][1] - r_jl[l][1]*n_ij[i][0] + r_jl[l][0]*n_ij[i][1];
      dA_ij[2][0] = r_ik[l][1]*n_ij[l][2] - r_ik[l][2]*n_ij[l][1] - r_il[l][1]*n_ij[i][2] + r_il[l][2]*n_ij[i][1];
      dA_ij[2][1] = r_ik[l][2]*n_ij[l][0] - r_ik[l][0]*n_ij[l][2] - r_il[l][2]*n_ij[i][0] + r_il[l][0]*n_ij[i][2];
      dA_ij[2][2] = r_ik[l][0]*n_ij[l][1] - r_ik[l][1]*n_ij[l][0] - r_il[l][0]*n_ij[i][1] + r_il[l][1]*n_ij[i][0];
      dA_ij[3][0] = -r_ij[l][2]*n_ij[i][1] + r_ij[l][1]*n_ij[i][2];
      dA_ij[3][1] = -r_ij[l][0]*n_ij[i][2] + r_ij[l][2]*n_ij[i][0];
      dA_ij[3][2] = -r_ij[l][1]*n_ij[i][0] + r_ij[l][0]*n_ij[i][1];

      nr_ij[l] = 0.0;
      for (m = 0; m < 3; m++)
        nr_ij[l] += n_i[m]*r_ij[l][m];

      for (m = 0; m < 3; m++){
        M_ij[0][m] = dA_ij[0][m]*nr_ij[l]/r_ij[l][3];
        M_ij[1][m] = (dA_ij[1][m]*nr_ij[l] - 2.0*A_ij[l]*r_ij[l][m]*nr_ij[l]/r_ij[l][3] + A_ij[l]*n_i[m])/r_ij[l][3];
	M_ij[2][m] = (dA_ij[2][m]*nr_ij[l] + 2.0*A_ij[l]*r_ij[l][m]*nr_ij[l]/r_ij[l][3] - A_ij[l]*n_i[m])/r_ij[l][3];
        M_ij[3][m] = dA_ij[3][m]*nr_ij[l]/r_ij[l][3];

        R_ij[3][m] = 0.0;     
      }	
	
      R_ij[0][0] = (r_ij[l][2]*z_i[1] - r_ij[l][1]*z_i[2] - nz_i*(r_ij[l][2]*n_sum[1] - r_ij[l][1]*n_sum[2]))/n_sum[4];
      R_ij[0][1] = (r_ij[l][0]*z_i[2] - r_ij[l][2]*z_i[0] - nz_i*(r_ij[l][0]*n_sum[2] - r_ij[l][2]*n_sum[0]))/n_sum[4];
      R_ij[0][2] = (r_ij[l][1]*z_i[0] - r_ij[l][0]*z_i[1] - nz_i*(r_ij[l][1]*n_sum[0] - r_ij[l][0]*n_sum[1]))/n_sum[4];
      R_ij[1][0] = (r_jk[l][2]*z_i[1] - r_jk[l][1]*z_i[2] - nz_i*(r_jk[l][2]*n_sum[1] - r_jk[l][1]*n_sum[2]))/n_sum[4];
      R_ij[1][1] = (r_jk[l][0]*z_i[2] - r_jk[l][2]*z_i[0] - nz_i*(r_jk[l][0]*n_sum[2] - r_jk[l][2]*n_sum[0]))/n_sum[4];
      R_ij[1][2] = (r_jk[l][1]*z_i[0] - r_jk[l][0]*z_i[1] - nz_i*(r_jk[l][1]*n_sum[0] - r_jk[l][0]*n_sum[1]))/n_sum[4];
      R_ij[2][0] = (r_ik[l][1]*z_i[2] - r_ik[l][2]*z_i[1] - nz_i*(r_ik[l][1]*n_sum[2] - r_ik[l][2]*n_sum[1]))/n_sum[4];
      R_ij[2][1] = (r_ik[l][2]*z_i[0] - r_ik[l][0]*z_i[2] - nz_i*(r_ik[l][2]*n_sum[0] - r_ik[l][0]*n_sum[2]))/n_sum[4];
      R_ij[2][2] = (r_ik[l][0]*z_i[1] - r_ik[l][1]*z_i[0] - nz_i*(r_ik[l][0]*n_sum[1] - r_ik[l][1]*n_sum[0]))/n_sum[4];

      // apply force to each of 4 atoms

      for (i = 0; i < 4; i++)
        for (m = 0; m < 3; m++)
          fh[i][m] = cc1*dA_ij[i][m] + cc2*(R_ij[i][m] + M_ij[i][m]); 

      if (newton_bond || i0 < nlocal)
        for (m = 0; m < 3; m++)
          f[i0][m] += fh[0][m];
      
      if (newton_bond || i1 < nlocal) 
        for (m = 0; m < 3; m++)
          f[i1][m] += fh[1][m];

      if (newton_bond || i2 < nlocal) 
        for (m = 0; m < 3; m++)
          f[i2][m] += fh[2][m];

      if (newton_bond || i3 < nlocal) 
        for (m = 0; m < 3; m++)
          f[i3][m] += fh[3][m]; 

      if (n_stress){
        ff[0] = -r_ik[l][0]*fh[0][0] - r_ij[l][0]*fh[2][0] - r_il[l][0]*fh[3][0];
        ff[1] = -r_ik[l][1]*fh[0][1] - r_ij[l][1]*fh[2][1] - r_il[l][1]*fh[3][1];
        ff[2] = -r_ik[l][2]*fh[0][2] - r_ij[l][2]*fh[2][2] - r_il[l][2]*fh[3][2];
        ff[3] = -r_ik[l][0]*fh[0][1] - r_ij[l][0]*fh[2][1] - r_il[l][0]*fh[3][1];
        ff[4] = -r_ik[l][0]*fh[0][2] - r_ij[l][0]*fh[2][2] - r_il[l][0]*fh[3][2];
        ff[5] = -r_ik[l][1]*fh[0][2] - r_ij[l][1]*fh[2][2] - r_il[l][1]*fh[3][2];	
        for (i = 0; i < n_stress; i++){
          m = output->stress_id[i];
          if ((output->next_stat_calc[m] == update->ntimestep) && (output->last_stat_calc[m] != update->ntimestep))
            output->stat[m]->virial5(i0,i1,i2,i3,ff);
        }
      }

      l++;
    }

    if (eflag){
      edihedral = 0.5*k[type]*A_i*(H_i - H0)*(H_i - H0);
      energy += edihedral;
    }

    i_st = i_fin;
  }
}

/* ---------------------------------------------------------------------- */

void DihedralBendArea::allocate()
{
  allocated = 1;
  int n = atom->ndihedraltypes;

  int individual = atom->individual;
  if (individual == 0)
    error->all(FLERR,"Individual has wrong value or is not set! Using dihedral bend/area only possible with individual = 1");  

  memory->create(k,n+1,"dihedral:k");

  memory->create(setflag,n+1,"dihedral:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;

  memory->create(A_ij,NV,"dihedral:c_ij");
  memory->create(n_sum,5,"dihedral:n_sum");
  memory->create(n_i,3,"dihedral:n_i");
  memory->create(z_i,3,"dihedral:z_i");
  memory->create(nr_ij,NV,"dihedral:nr_ij");
  memory->create(n_ij,NV,5,"dihedral:n_ij");
  memory->create(r_ij,NV,4,"dihedral:r_ij");
  memory->create(r_ik,NV,3,"dihedral:r_ik");
  memory->create(r_jk,NV,3,"dihedral:r_jk");
  memory->create(r_il,NV,3,"dihedral:r_il");
  memory->create(r_jl,NV,3,"dihedral:r_jl");
  memory->create(dA_ij,4,3,"dihedral:dc_ij");
  memory->create(R_ij,4,3,"dihedral:A_ij");
  memory->create(M_ij,4,3,"dihedral:M_ij");
  memory->create(fh,4,3,"dihedral:fh");
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script
------------------------------------------------------------------------- */

void DihedralBendArea::coeff(int narg, char **arg)
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

void DihedralBendArea::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->ndihedraltypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void DihedralBendArea::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->ndihedraltypes,fp);
  }
  MPI_Bcast(&k[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->ndihedraltypes; i++) setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */
