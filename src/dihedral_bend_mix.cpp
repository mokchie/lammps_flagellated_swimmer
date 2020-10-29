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

#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include "dihedral_bend_mix.h"
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

#define NV    10

/* ---------------------------------------------------------------------- 
Syntax: dihedral_style bend/mix
        dihedral_coeff ID kappa

kappa: bending rigidity
-------------------------------------------------------------------------*/

/* ---------------------------------------------------------------------- */

DihedralBendMix::DihedralBendMix(LAMMPS *lmp) : Dihedral(lmp)
{
  k = ck = cj = z_i = dot_k = dot_j = dot_i = s_k = s_j = n_sum = n_i = nr_ij = nr_ik = NULL;
  n_ij = r_ij = r_ik = r_jk = dck = dcj = R_ij = B_ij = M_ij = fh = NULL;
}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

DihedralBendMix::~DihedralBendMix()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(ck);
    memory->destroy(cj);
    memory->destroy(z_i);
    memory->destroy(dot_k);
    memory->destroy(dot_j);
    memory->destroy(dot_i);
    memory->destroy(s_k);
    memory->destroy(s_j);
    memory->destroy(n_sum);
    memory->destroy(n_i);
    memory->destroy(nr_ij);
    memory->destroy(nr_ik);
    memory->destroy(n_ij);
    memory->destroy(r_ij);
    memory->destroy(r_ik);
    memory->destroy(r_jk);
    memory->destroy(dck);
    memory->destroy(dcj);
    memory->destroy(R_ij);
    memory->destroy(B_ij);
    memory->destroy(M_ij);
    memory->destroy(fh);
  }
}

/* ---------------------------------------------------------------------- */

void DihedralBendMix::compute(int eflag, int vflag)
{
  int i0,i1,i2,i3,type,i_st,i_fin,at,vt;
  int i,l,m,n;
  double H0,H_i,A_i,D_i,co_k,co_j,nz_i,cc1,cc2;
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

      r_ij[l][3] = r_ik[l][3] = r_jk[l][3] = 0.0;
      dot_k[l] = dot_j[l] = dot_i[l] = 0.0;
      for (m = 0; m < 3; m++){
        r_ij[l][m] = x[i1][m] - x[i2][m];
	r_ij[l][3] += r_ij[l][m]*r_ij[l][m];
	r_ik[l][m] = x[i1][m] - x[i0][m];
	r_ik[l][3] += r_ik[l][m]*r_ik[l][m];
	r_jk[l][m] = x[i2][m] - x[i0][m];
	r_jk[l][3] += r_jk[l][m]*r_jk[l][m];	
        dot_k[l] += r_jk[l][m]*r_ik[l][m];
	dot_j[l] -= r_ij[l][m]*r_jk[l][m];
	dot_i[l] += r_ij[l][m]*r_ik[l][m];
      }      
      s_k[l] = sqrt(r_ik[l][3]*r_jk[l][3] - dot_k[l]*dot_k[l]);
      s_j[l] = sqrt(r_ij[l][3]*r_jk[l][3] - dot_j[l]*dot_j[l]);
      ck[l] = 0.5*dot_k[l]/s_k[l];
      cj[l] = 0.5*dot_j[l]/s_j[l];

      n_ij[l][0] = r_jk[l][1]*r_ik[l][2] - r_ik[l][1]*r_jk[l][2];
      n_ij[l][1] = r_ik[l][0]*r_jk[l][2] - r_jk[l][0]*r_ik[l][2];
      n_ij[l][2] = r_jk[l][0]*r_ik[l][1] - r_ik[l][0]*r_jk[l][1];

      if (dot_i[l] < 0.0 || dot_k[l] < 0.0 || dot_j[l] < 0.0){
      //if (0){
	n_ij[l][5] = 0.125;
	if (dot_i[l] < 0.0)
	  n_ij[l][5] *= 2.0;
	
	n_ij[l][3] = 0.0;
	for (m = 0; m < 3; m++)
          n_ij[l][3] += n_ij[l][m]*n_ij[l][m];
	n_ij[l][4] = sqrt(n_ij[l][3]);

	A_i += n_ij[l][5]*n_ij[l][4];
      } else {
        n_ij[l][5] = -1.0;
	A_i += 0.25*(r_ij[l][3]*ck[l] + r_ik[l][3]*cj[l]);
      }

      for (m = 0; m < 3; m++){
	n_sum[m] += n_ij[l][m];
        z_i[m] += ck[l]*r_ij[l][m] + cj[l]*r_ik[l][m];
      }
      
      l++;
    }

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

    for (l = 0; l < vt; l++){
      nr_ij[l] = 0.0;
      nr_ik[l] = 0.0;
      for (m = 0; m < 3; m++){
        nr_ij[l] += n_i[m]*r_ij[l][m];
        nr_ik[l] += n_i[m]*r_ik[l][m];
      }
      if (n_ij[l][5] > 0.0)
	for (m = 0; m < 3; m++)
	  n_ij[l][m] *= n_ij[l][5]/n_ij[l][4];  
    }
       
    cc1 = -0.5*k[type]*(H0*H0 - H_i*H_i);
    cc2 = -k[type]*(H_i - H0);
        
    l = 0;
    for (n = i_st; n < i_fin; n++) {
      i0 = dihedrallist[n][0];
      i1 = dihedrallist[n][1];
      i2 = dihedrallist[n][2];
      i3 = dihedrallist[n][3];     
 
      co_k = 0.5/s_k[l]/s_k[l]/s_k[l];
      co_j = 0.5/s_j[l]/s_j[l]/s_j[l];
      
      for (m = 0; m < 3; m++){
        dck[0][m] = co_k*(dot_k[l]*(r_ik[l][m]*r_jk[l][3] + r_jk[l][m]*r_ik[l][3]) - r_ik[l][3]*r_jk[l][3]*(r_jk[l][m] + r_ik[l][m])); 
        dck[1][m] = co_k*(r_ik[l][3]*r_jk[l][3]*r_jk[l][m] - dot_k[l]*r_ik[l][m]*r_jk[l][3]); 
	dck[2][m] = co_k*(r_ik[l][3]*r_jk[l][3]*r_ik[l][m] - dot_k[l]*r_jk[l][m]*r_ik[l][3]);

	dcj[0][m] = co_j*(r_ij[l][3]*r_jk[l][3]*r_ij[l][m] + dot_j[l]*r_jk[l][m]*r_ij[l][3]);
        dcj[1][m] = -co_j*(r_ij[l][3]*r_jk[l][3]*r_jk[l][m] + dot_j[l]*r_ij[l][m]*r_jk[l][3]); 
	dcj[2][m] = co_j*(dot_j[l]*(r_ij[l][m]*r_jk[l][3] - r_jk[l][m]*r_ij[l][3]) - r_ij[l][3]*r_jk[l][3]*(r_ij[l][m] - r_jk[l][m]));  

        M_ij[0][m] = dck[0][m]*nr_ij[l] + dcj[0][m]*nr_ik[l] - cj[l]*n_i[m];
        M_ij[1][m] = dck[1][m]*nr_ij[l] + dcj[1][m]*nr_ik[l] + (ck[l] + cj[l])*n_i[m];
        M_ij[2][m] = dck[2][m]*nr_ij[l] + dcj[2][m]*nr_ik[l] - ck[l]*n_i[m];  
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

      if (n_ij[l][5] > 0.0){
        B_ij[0][0] = r_ij[l][2]*n_ij[l][1] - r_ij[l][1]*n_ij[l][2];
        B_ij[0][1] = r_ij[l][0]*n_ij[l][2] - r_ij[l][2]*n_ij[l][0];
        B_ij[0][2] = r_ij[l][1]*n_ij[l][0] - r_ij[l][0]*n_ij[l][1];
        B_ij[1][0] = r_jk[l][2]*n_ij[l][1] - r_jk[l][1]*n_ij[l][2];
        B_ij[1][1] = r_jk[l][0]*n_ij[l][2] - r_jk[l][2]*n_ij[l][0];
        B_ij[1][2] = r_jk[l][1]*n_ij[l][0] - r_jk[l][0]*n_ij[l][1];
        B_ij[2][0] = r_ik[l][1]*n_ij[l][2] - r_ik[l][2]*n_ij[l][1];
        B_ij[2][1] = r_ik[l][2]*n_ij[l][0] - r_ik[l][0]*n_ij[l][2];
        B_ij[2][2] = r_ik[l][0]*n_ij[l][1] - r_ik[l][1]*n_ij[l][0];	
      } else{
        for (m = 0; m < 3; m++){
      	  B_ij[0][m] = 0.25*(dck[0][m]*r_ij[l][3] + dcj[0][m]*r_ik[l][3]) - 0.5*cj[l]*r_ik[l][m];
          B_ij[1][m] = 0.25*(dck[1][m]*r_ij[l][3] + dcj[1][m]*r_ik[l][3]) + 0.5*(ck[l]*r_ij[l][m] + cj[l]*r_ik[l][m]);
          B_ij[2][m] = 0.25*(dck[2][m]*r_ij[l][3] + dcj[2][m]*r_ik[l][3]) - 0.5*ck[l]*r_ij[l][m];
	}
      }

      // apply force to each of 4 atoms

      for (i = 0; i < 3; i++)
        for (m = 0; m < 3; m++)
          fh[i][m] = cc1*B_ij[i][m] + cc2*(R_ij[i][m] + M_ij[i][m]); 

      if (newton_bond || i0 < nlocal)
        for (m = 0; m < 3; m++)
          f[i0][m] += fh[0][m];
      
      if (newton_bond || i1 < nlocal) 
        for (m = 0; m < 3; m++)
          f[i1][m] += fh[1][m];

      if (newton_bond || i2 < nlocal) 
        for (m = 0; m < 3; m++)
          f[i2][m] += fh[2][m];

      if (n_stress){
        ff[0] = -r_ik[l][0]*fh[0][0] - r_ij[l][0]*fh[2][0];
        ff[1] = -r_ik[l][1]*fh[0][1] - r_ij[l][1]*fh[2][1];
        ff[2] = -r_ik[l][2]*fh[0][2] - r_ij[l][2]*fh[2][2];
        ff[3] = -r_ik[l][0]*fh[0][1] - r_ij[l][0]*fh[2][1];
        ff[4] = -r_ik[l][0]*fh[0][2] - r_ij[l][0]*fh[2][2];
        ff[5] = -r_ik[l][1]*fh[0][2] - r_ij[l][1]*fh[2][2];	
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

void DihedralBendMix::allocate()
{
  allocated = 1;
  int n = atom->ndihedraltypes;

  int individual = atom->individual;
  if (individual == 0)
    error->all(FLERR,"Individual has wrong value or is not set! Using dihedral bend/mix only possible with individual = 1");  

  memory->create(k,n+1,"dihedral:k");

  memory->create(setflag,n+1,"dihedral:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;

  memory->create(ck,NV,"dihedral:ck");
  memory->create(cj,NV,"dihedral:cj");
  memory->create(z_i,3,"dihedral:z_i");
  memory->create(dot_k,NV,"dihedral:dot_k");
  memory->create(dot_j,NV,"dihedral:dot_j");
  memory->create(dot_i,NV,"dihedral:dot_i");
  memory->create(s_k,NV,"dihedral:s_k");
  memory->create(s_j,NV,"dihedral:s_j");
  memory->create(n_sum,5,"dihedral:n_sum");
  memory->create(n_i,3,"dihedral:n_i");
  memory->create(nr_ij,NV,"dihedral:nr_ij");
  memory->create(nr_ik,NV,"dihedral:nr_ik");
  memory->create(n_ij,NV,6,"dihedral:n_ij");
  memory->create(r_ij,NV,4,"dihedral:r_ij");
  memory->create(r_ik,NV,4,"dihedral:r_ik");
  memory->create(r_jk,NV,4,"dihedral:r_jk");
  memory->create(dck,3,3,"dihedral:dck");
  memory->create(dcj,3,3,"dihedral:dcj");
  memory->create(R_ij,3,3,"dihedral:R_ij");
  memory->create(B_ij,3,3,"dihedral:B_ij");
  memory->create(M_ij,3,3,"dihedral:M_ij");
  memory->create(fh,3,3,"dihedral:fh");
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script
------------------------------------------------------------------------- */

void DihedralBendMix::coeff(int narg, char **arg)
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

void DihedralBendMix::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->ndihedraltypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void DihedralBendMix::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->ndihedraltypes,fp);
  }
  MPI_Bcast(&k[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->ndihedraltypes; i++) setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */
