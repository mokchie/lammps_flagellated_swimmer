/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Kurt Smith (U Pittsburgh)
                        Dmitry Fedosov 
                        C Mo
------------------------------------------------------------------------- */
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pair_sdpdve_2d.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "force.h"
#include "group.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "random_mars.h"
#include "MersenneTwister.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-10
#define EPS 1.0e-3 // EPS is the criteria we used to decide whether two eigenvals are equal
#define EPSR 1.0e-4 // EPSR is the criteria we used to decide whether an eigenvector must be renormalized
#define DR 0.01

/* ---------------------------------------------------------------------- */

PairSDPDVE2D::PairSDPDVE2D(LAMMPS *lmp) : Pair(lmp)
{
  random = NULL;
  mtrand = NULL;
  weight1 = weight2 = NULL;
  // weight1 is the tabulated kernel function w(r)
  // weight2 is the tabulation of -w'(r)/r
  set_weight_ind = 1;
  dr_inv = 1.0/DR;
  init_fix = 1;
  list_fix_bc = NULL;

  comm_forward = 10; 
  comm_reverse = 10; // These two numbers both should be 1+9=10 because besides rho we also need to sum up the vel gradient tensor
}

/* ---------------------------------------------------------------------- */

PairSDPDVE2D::~PairSDPDVE2D()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(eta);
    memory->destroy(rho0);
  }

  if (random) delete random;

#ifdef MTRAND
  if (mtrand) delete mtrand;
#endif
  memory->destroy(weight1);
  memory->destroy(weight2);
  memory->destroy(list_fix_bc);
}

/* ---------------------------------------------------------------------- */

void PairSDPDVE2D::compute(int eflag, int vflag)
{
  int i,j,k,l,ll,ii,jj,kk,inum,jnum,itype,jtype;
  double sum;
  double PI_i[2][2], PI_j[2][2],fcon[2],kappa_eigen[2][2],H[2][2],randn[2];
  double xtmp,ytmp,ztmp,evdwl,fpair;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double rsq,r,rinv,dot,wd,wdf,factor_dpd;
  double p_i,p_j,gam,sig,fnc[3],omx,omy,omz;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double r20_7 = 20.0/7.0;
  double norm;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  if (init_fix){
    num_fix_bc = 0; 
    for (i = 0; i < modify->nfix; i++){
      if (strcmp(modify->fix[i]->id,"inflow") == 0) num_fix_bc++;
      if (strcmp(modify->fix[i]->id,"inflow/periodic") == 0) num_fix_bc++; 
      if (strcmp(modify->fix[i]->id,"outflow") == 0) num_fix_bc++;
    }
    if (num_fix_bc){
      memory->create(list_fix_bc,num_fix_bc+1,"pair:list_fix_bc");
      num_fix_bc = 0; 
      for (i = 0; i < modify->nfix; i++){
        if (strcmp(modify->fix[i]->id,"inflow") == 0) list_fix_bc[num_fix_bc++] = i;
        if (strcmp(modify->fix[i]->id,"inflow/periodic") == 0) list_fix_bc[num_fix_bc++] = i;
        if (strcmp(modify->fix[i]->id,"outflow") == 0) list_fix_bc[num_fix_bc++] = i;
      }       
    }
    init_fix = 0;
  }

  if (set_weight_ind){
    set_weight();
    set_weight_ind = 0;
  }

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *rho = atom->rho;
  double ***kappa = atom->kappa;
  double ***ctensor = atom->ctensor;
  double **ctensor_eigenval = atom->ctensor_eigenval;
  double ***ctensor_eigenvec = atom->ctensor_eigenvec; 
  double **f_ctensor_eigenval = atom->f_ctensor_eigenval;
  double ***f_ctensor_eigenvec = atom->f_ctensor_eigenvec;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);
  int n_stress = output->n_stress;
  int ind_stress;
  double ff[6];

  for (i = 0; i < nall; i++) {
    rho[i] = 0.0;
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        kappa[i][j][k] = 0.0;
      }
    }
  }

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (i=0; i<nall; i++) {
    for (j=0; j<2; j++){
      for (k=j; k<2; k++){
        sum = 0.0;
        for (l=0; l<2; l++){
          sum += ctensor_eigenval[i][l] * ctensor_eigenvec[i][l][j] * ctensor_eigenvec[i][l][k];
        }
        ctensor[i][j][k] = sum;
        if(k>j){
          ctensor[i][k][j] = sum;
        }
      }
    }
  }


  // loop over the neighbors to accumulate the local density and velocity gradient tensor
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = 0.0;
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = 0.0;
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    if (mask[i] & groupbit){
      rho[i] += weight1[itype][itype][0];
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        if (mask[j] & groupbit){
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = 0.0;
          delvx = vxtmp - v[j][0];
          delvy = vytmp - v[j][1];
          delvz = 0.0;
          rsq = delx*delx + dely*dely;
          jtype = type[j];

          if (rsq < cutsq[itype][jtype]) {
            r = sqrt(rsq);
            k = static_cast<int> (r*dr_inv);
            wd = (weight1[itype][jtype][k+1]-weight1[itype][jtype][k])*(r*dr_inv - k) + weight1[itype][jtype][k];
            wdf = (weight2[itype][jtype][k+1]-weight2[itype][jtype][k])*(r*dr_inv - k) + weight2[itype][jtype][k];
            rho[i] += wd;
            kappa[i][0][0] += wdf*delx*delvx;
            kappa[i][0][1] += wdf*delx*delvy;
            kappa[i][1][0] += wdf*dely*delvx;
            kappa[i][1][1] += wdf*dely*delvy;
            if (newton_pair || j < nlocal){
              rho[j] += wd; 
              kappa[j][0][0] += wdf*delx*delvx;
              kappa[j][0][1] += wdf*delx*delvy;
              kappa[j][1][0] += wdf*dely*delvx;
              kappa[j][1][1] += wdf*dely*delvy;
            }
	        }
	      }
      }
    }
  }

  // communicate and sum densities & gradient tensor
  comm->reverse_comm_pair(this);
  comm->forward_comm_pair(this);
  for (i = 0; i < nall; i++)
    if (mask[i] & groupbit_rho)
      rho[i] = rho_reset;
  // kappa needs to be divded by the local density to get the real velocity gradient tensor
  for (i = 0; i < nall; i++){
    for (j = 0; j < 2; j++){
      for (k = 0; k < 2; k++){
        kappa[i][j][k] /= rho[i];
      }
    }
  }

  // renormalize the eigenvectors if needed
  for(i=0; i<nlocal; i++){
    for(j=0; j<2; j++){
      norm = 0.0;
      for(k=0; k<2; k++){
        norm += ctensor_eigenvec[i][j][k]*ctensor_eigenvec[i][j][k];
      }
      norm = sqrt(norm);
      if(fabs(norm-1.0)>EPSR){
      for(k=0; k<2; k++)
        ctensor_eigenvec[i][j][k] /= norm;
      }
    }
  }


  if (num_fix_bc)
    for (i = 0; i < num_fix_bc; i++)
      modify->fix[list_fix_bc[i]]->reset_target(rho_reset);

  // loop over neighbors of my atoms to calculate forces
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = 0.0;
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = 0.0;
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    if (mask[i] & groupbit){
      wd = rho[i]/rho0[itype][itype];
      p_i = wd;
      for (k = 0; k < g_exp-1; k++)
        p_i *= wd;
      p_i = p0*p_i + bb;
      for (k=0; k<2; k++){
        for (l=k; l<2; l++){
          if (l==k)
            PI_i[k][l] = p_i + Np*rho[i]*temperature*(1-ctensor[i][k][l]);
          else {
            PI_i[k][l] = - Np*rho[i]*temperature*ctensor[i][k][l];
            PI_i[l][k] = PI_i[k][l];
          }

        }
      }

      if (n_stress)
        for (k = 0; k < n_stress; k++){
          l = output->stress_id[k];
          if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
            output->stat[l]->virial1(i);
        }
   
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        factor_dpd = special_lj[sbmask(j)];
        j &= NEIGHMASK;
        
        if (mask[j] & groupbit){
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = 0.0;
          rsq = delx*delx + dely*dely;
          jtype = type[j];

          if (rsq < cutsq[itype][jtype]) {
            wd = rho[j]/rho0[jtype][jtype];
            p_j = wd;
            for (k = 0; k < g_exp-1; k++)
              p_j *= wd;
            p_j = p0*p_j + bb;
            for (k=0; k<2; k++){
              for (l=k; l<2; l++){
                if (l==k)
                  PI_j[k][l] = p_j + Np*rho[j]*temperature*(1-ctensor[j][k][l]);
                else {
                  PI_j[k][l] = - Np*rho[j]*temperature*ctensor[j][k][l];
                  PI_j[l][k] = PI_j[k][l];
                }

              }
            }


            r = sqrt(rsq);
            if (r < EPSILON) continue;     // r can be 0.0 in SDPDVE2D systems
            rinv = 1.0/r;
            delvx = vxtmp - v[j][0];
            delvy = vytmp - v[j][1];
            delvz = 0.0;
            dot = (delx*delvx + dely*delvy)*rinv;
            omx = 0.0;
            omy = 0.0;
            omz = omega[i][2] + omega[j][2];  
              
            k = static_cast<int> (r*dr_inv);
            wd = (weight2[itype][jtype][k+1]-weight2[itype][jtype][k])*(r*dr_inv - k) + weight2[itype][jtype][k];

            //generate_wrr();
#ifdef MTRAND
            wrr[0] = sqrt(0.5*(delx*delx/rsq + 3.0))*(2.0*mtrand->rand()-1.0);
            wrr[1] = sqrt(0.5*(dely*dely/rsq + 3.0))*(2.0*mtrand->rand()-1.0);
            wrr[2] = sqrt(0.5*(delz*delz/rsq + 3.0))*(2.0*mtrand->rand()-1.0);
            wrr[3] = 2.0*mtrand->rand()-1.0;
#else
            wrr[0] = sqrt(0.5*(delx*delx/rsq + 3.0))*(2.0*random->uniform()-1.0);
            wrr[1] = sqrt(0.5*(dely*dely/rsq + 3.0))*(2.0*random->uniform()-1.0);
            wrr[2] = sqrt(0.5*(delz*delz/rsq + 3.0))*(2.0*random->uniform()-1.0);
            wrr[3] = 2.0*random->uniform()-1.0;
#endif

            gam = r20_7*eta[itype][jtype]*wd/rho[i]/rho[j];
            sig = 2.0*sqrt(temperature*gam)*dtinvsqrt;

            fpair = -gam*dot*rinv;  // drag force 
            fpair += sig*wrr[3]*rinv;  // random force
            fpair *= factor_dpd;

            //fcon is the force from the reversible part of the dynamics equation
            for (k=0; k<2; k++){
              fcon[k] = ((PI_i[k][0]/rho[i]/rho[i] + PI_j[k][0]/rho[j]/rho[j]) * delx + (PI_i[k][1]/rho[i]/rho[i] + PI_j[k][1]/rho[j]/rho[j]) * dely) * wd; 
            }
            

            fnc[0] = factor_dpd*fcon[0] + factor_dpd*(sig*wrr[0] - delvx*gam - 0.5*gam*(dely*omz - delz*omy));
            fnc[1] = factor_dpd*fcon[1] + factor_dpd*(sig*wrr[1] - delvy*gam - 0.5*gam*(delz*omx - delx*omz));
	          fnc[2] = 0.0;

            f[i][0] += delx*fpair + fnc[0];
    	      f[i][1] += dely*fpair + fnc[1];
	          f[i][2] += 0.0;
            torque[i][0] += 0.0;
            torque[i][1] += 0.0;
            torque[i][2] += 0.5*(dely*fnc[0] - delx*fnc[1]);
            if (newton_pair || j < nlocal) {
              f[j][0] -= delx*fpair + fnc[0];
    	        f[j][1] -= dely*fpair + fnc[1];
	            f[j][2] -= 0.0;
              torque[j][0] += 0.0;
              torque[j][1] += 0.0;
              torque[j][2] += 0.5*(dely*fnc[0] - delx*fnc[1]);
            }

            /*if (eflag) {
              evdwl = 0.5*a0[itype][jtype]*cut[itype][jtype] * wd*wd; //old from DPD 
              evdwl *= factor_dpd;
	      }*/

            if (n_stress){
              ind_stress = 0;
              ff[0] = delx*(delx*fpair + fnc[0]);
              ff[1] = dely*(dely*fpair + fnc[1]);
              ff[2] = delz*(delz*fpair + fnc[2]);
              ff[3] = delx*(dely*fpair + fnc[1]);
              ff[4] = delx*(delz*fpair + fnc[2]);
              ff[5] = dely*(delz*fpair + fnc[2]);
              //if (itype != jtype) ind_stress = 1;
              for (k = 0; k < n_stress; k++){
                l = output->stress_id[k];
                if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
                  output->stat[l]->virial2(j,ff,ind_stress);
              }
            }

            if (evflag) ev_tally(i,j,nlocal,newton_pair,
                                 evdwl,0.0,fpair,delx,dely,delz);
	        }
	      }
      }
    }
  }
  
  for (i=0; i<nlocal; i++){
    for (j=0; j<2; j++){
      for (k=0; k<2; k++){

        kappa_eigen[j][k] = 0.0;
        for (jj=0; jj<2; jj++){
          for (kk=0; kk<2; kk++){
            kappa_eigen[j][k] += ctensor_eigenvec[i][j][jj]*kappa[i][jj][kk]*ctensor_eigenvec[i][k][kk];
          }
        }
      }
    }

    for (j=0; j<2; j++){
      for (k=j; k<2; k++){
        H[j][k] = 0.0;
        if(j != k){
          if(fabs(ctensor_eigenval[i][j] - ctensor_eigenval[i][k])<EPS)
          {
              H[j][k] = 0.0;
          }
          else
            H[j][k] = (ctensor_eigenval[i][j]*kappa_eigen[j][k] + ctensor_eigenval[i][k]*kappa_eigen[k][j]) / (ctensor_eigenval[i][j] - ctensor_eigenval[i][k]);
          H[k][j] = -H[j][k];
        }

      }
    }

    for (j=0; j<2; j++) {
      randn[j] = random->gaussian(); //Note: it must be gaussian, uniform distribution won't work
      f_ctensor_eigenval[i][j] = 2.0*ctensor_eigenval[i][j]*kappa_eigen[j][j] + (1.0-ctensor_eigenval[i][j])/tau + 2.0/tau/Np + sqrt(4.0*ctensor_eigenval[i][j]/tau/Np)*randn[j]*dtinvsqrt;
      f_ctensor_eigenvec[i][j][0] = 0.0;
      f_ctensor_eigenvec[i][j][1] = 0.0;
      for (k=0; k<2; k++){
        f_ctensor_eigenvec[i][j][0] += H[j][k]*ctensor_eigenvec[i][k][0];
        f_ctensor_eigenvec[i][j][1] += H[j][k]*ctensor_eigenvec[i][k][1];
      }
    }
    f_ctensor_eigenval[i][2] = 0.0;
    f_ctensor_eigenvec[i][0][2] = 0.0;
    f_ctensor_eigenvec[i][1][2] = 0.0;
    f_ctensor_eigenvec[i][2][2] = 0.0;
    f_ctensor_eigenvec[i][2][1] = 0.0;
    f_ctensor_eigenvec[i][2][0] = 0.0;

  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSDPDVE2D::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cut,n+1,n+1,"pair:cut");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++){
      setflag[i][j] = 0;
      cut[i][j] = 0.0;
  }

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(eta,n+1,n+1,"pair:eta");
  memory->create(rho0,n+1,n+1,"pair:rho0");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSDPDVE2D::settings(int narg, char **arg)
{
  long unsigned int seed2;
  int grph;
  /* 
  narg should be 11, 2 additional arguments have been added: 
  tau: relaxation time, its datatype is double;
  Np: the number of polymers in one particle, its datatype is int;
  */
  if (narg != 11) error->all(FLERR,"Illegal pair_style command");

  p0 = force->numeric(FLERR,arg[0]);
  bb = force->numeric(FLERR,arg[1]);
  g_exp = force->inumeric(FLERR,arg[2]);
  temperature = force->numeric(FLERR,arg[3]);
  grph = group->find(arg[4]);
  if (grph == -1) error->all(FLERR,"Could not find group ID in sdpdve/2d");
  groupbit = group->bitmask[grph];
  cut_global = force->numeric(FLERR,arg[5]);
  seed = force->inumeric(FLERR,arg[6]);
  grph = group->find(arg[7]);
  if (grph == -1) error->all(FLERR,"Could not find group_rho ID in sdpdve/2d");
  groupbit_rho = group->bitmask[grph];
  rho_reset = force->numeric(FLERR,arg[8]);
  tau = force->numeric(FLERR,arg[9]);
  Np = force->inumeric(FLERR,arg[10]);

  // initialize Marsaglia RNG with processor-unique seed

  if (seed <= 0) error->all(FLERR,"Illegal pair_style command");
  seed +=  comm->me;

#ifdef MTRAND
  if (mtrand) delete mtrand;
  seed2 = static_cast<long unsigned int>(seed);
  mtrand = new MTRand(&seed2);
  //mtrand = new MTRand();
#else
  if (random) delete random;
  random = new RanMars(lmp,seed);
#endif

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSDPDVE2D::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double rho0_one = force->numeric(FLERR,arg[2]);
  double eta_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      rho0[i][j] = rho0_one;
      eta[i][j] = eta_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSDPDVE2D::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair sdpdve/2d requires ghost atoms store velocity");

  // if newton off, forces between atoms ij will be double computed
  // using different random numbers

  if (force->newton_pair == 0 && comm->me == 0) error->warning(FLERR,
      "Pair sdpdve/2d needs newton pair on for momentum conservation");

  neighbor->request(this,instance_me);
  set_weight_ind = 1;
}

/* ---------------------------------------------------------------------- 
   init for one type pair i,j and corresponding j,i */

double PairSDPDVE2D::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  cut[j][i] = cut[i][j];
  rho0[j][i] = rho0[i][j];
  eta[j][i] = eta[i][j];
  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSDPDVE2D::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&rho0[i][j],sizeof(double),1,fp);
        fwrite(&eta[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSDPDVE2D::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&rho0[i][j],sizeof(double),1,fp);
          fread(&eta[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&rho0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&eta[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSDPDVE2D::write_restart_settings(FILE *fp)
{
  fwrite(&p0,sizeof(double),1,fp);
  fwrite(&bb,sizeof(double),1,fp);
  fwrite(&g_exp,sizeof(int),1,fp);
  fwrite(&temperature,sizeof(double),1,fp);
  fwrite(&groupbit,sizeof(int),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&seed,sizeof(int),1,fp);
  fwrite(&groupbit_rho,sizeof(int),1,fp);
  fwrite(&rho_reset,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tau,sizeof(double),1,fp);
  fwrite(&Np,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSDPDVE2D::read_restart_settings(FILE *fp)
{
  long unsigned int seed2;

  if (comm->me == 0) {
    fread(&p0,sizeof(double),1,fp);
    fread(&bb,sizeof(double),1,fp);
    fread(&g_exp,sizeof(int),1,fp);
    fread(&temperature,sizeof(double),1,fp);
    fread(&groupbit,sizeof(int),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&seed,sizeof(int),1,fp);
    fread(&groupbit_rho,sizeof(int),1,fp);
    fread(&rho_reset,sizeof(double),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tau,sizeof(double),1,fp);
    fread(&Np,sizeof(int),1,fp);
  }
  MPI_Bcast(&p0,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&bb,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&g_exp,1,MPI_INT,0,world);
  MPI_Bcast(&temperature,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&groupbit,1,MPI_INT,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&seed,1,MPI_INT,0,world);
  MPI_Bcast(&groupbit_rho,1,MPI_INT,0,world);
  MPI_Bcast(&rho_reset,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tau,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&Np,1,MPI_INT,0,world);

  // initialize Marsaglia RNG with processor-unique seed
  // same seed that pair_style command initially specified

  seed +=  comm->me;
#ifdef MTRAND
  if (mtrand) delete mtrand;
  seed2 = static_cast<long unsigned int>(seed);
  mtrand = new MTRand(&seed2);
  //mtrand = new MTRand();
#else
  if (random) delete random;
  random = new RanMars(lmp,seed);
#endif
}

/* ---------------------------------------------------------------------- */

double PairSDPDVE2D::single(int i, int j, int itype, int jtype, double rsq,
                       double factor_coul, double factor_dpd, double &fforce)
{
  int k;
  double r,wd,phi,p_i,p_j;
  double *rho = atom->rho;

  r = sqrt(rsq);
  if (r < EPSILON) {
    fforce = 0.0;
    return 0.0;
  }

  wd = rho[i]/rho0[itype][itype];
  p_i = wd;
  for (k = 0; k < g_exp-1; k++)
    p_i *= wd;
  p_i = p0*p_i + bb;
  //p_i = p0*pow(rho[i]/rho0[itype][itype],g_exp) + bb;

  wd = rho[j]/rho0[jtype][jtype];
  p_j = wd;
  for (k = 0; k < g_exp-1; k++)
    p_j *= wd;
  p_j = p0*p_j + bb;
  //p_j = p0*pow(rho[j]/rho0[jtype][jtype],g_exp) + bb;


  k = static_cast<int> (r*dr_inv);
  wd = (weight2[itype][jtype][k+1]-weight2[itype][jtype][k])*(r*dr_inv - k) + weight2[itype][jtype][k];
  fforce = factor_dpd*(p_i/rho[i]/rho[i] + p_j/rho[j]/rho[j])*wd; // conservative force

  phi = 0.0;
  return factor_dpd*phi;
}



void PairSDPDVE2D::set_weight()
{
  //using the quintic spline kernel
  int i,j,k,l;
  int n = atom->ntypes;
  double rr,omega0,h;
 
  if (weight1) { 
    memory->destroy(weight1);
    weight1 = NULL;
  }
  if (weight2) { 
    memory->destroy(weight2);
    weight2 = NULL;
  }

  if (domain->dimension == 3) {
      omega0 = 3.0 / (359.0 * M_PI);
    } else{
      omega0 = 7.0 / (478.0 * M_PI);
  }


  rr = -1.0;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++) 
      rr = MAX(rr,cut[i][j]);


  nw_max = static_cast<int> (rr*dr_inv) + 2;
  if (nw_max < 1 || nw_max > 600) error->all(FLERR,"Non-positive or too large value for nw_max to initialize weight arrays: PairSDPDVE2D::set_weight.");
  memory->create(weight1,n+1,n+1,nw_max,"pair:weight1");
  memory->create(weight2,n+1,n+1,nw_max,"pair:weight2");

  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++){
      k = static_cast<int> (cut[i][j]*dr_inv) + 1;
      if (k > nw_max) error->all(FLERR,"Error in PairSDPDVE2D::set_weight - k > nw_max");

      if (cut[i][j] > 0.0){
        h = cut[i][j]/3.0;
        for (l = 0; l < nw_max; l++){
          rr = l*DR/h;
          if (rr > 3.0){
            weight1[i][j][l] = 0.0;
            weight2[i][j][l] = 0.0;
          } else{
            if (domain->dimension == 3) {
              if(rr<=1.0){
                weight1[i][j][l] = omega0/h/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 6.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr) + 15.0*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 30.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr) + 75.0*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr))/rr;
              } else if(rr<=2.0){
                weight1[i][j][l] = omega0/h/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 6.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 30.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr))/rr;
              } else {
                weight1[i][j][l] = omega0/h/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr))/rr;
              }
            } else{
              if(rr<=1.0){
                weight1[i][j][l] = omega0/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 6.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr) + 15.0*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 30.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr) + 75.0*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr))/rr;
              } else if(rr<=2.0){
                weight1[i][j][l] = omega0/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 6.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 30.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr))/rr;
              } else {
                weight1[i][j][l] = omega0/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr))/rr;
              }

            }
          }
          weight1[j][i][l] = weight1[i][j][l];
          weight2[j][i][l] = weight2[i][j][l];
        } 
        weight2[i][j][0] = weight2[i][j][1];
        weight2[j][i][0] = weight2[j][i][1];
      } else{
        for (l = 0; l < nw_max; l++){ 
          weight1[i][j][l] = 0.0;
          weight2[i][j][l] = 0.0;
          weight1[j][i][l] = 0.0;
          weight2[j][i][l] = 0.0;
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

// void PairSDPDVE2D::set_weight()
// {
//   int i,j,k,l;
//   int n = atom->ntypes;
//   double rr,al1,al2;
 
//   if (weight1) { 
//     memory->destroy(weight1);
//     weight1 = NULL;
//   }
//   if (weight2) { 
//     memory->destroy(weight2);
//     weight2 = NULL;
//   }

//   if (domain->dimension == 3) {
//     al1 = 105.0/16.0/M_PI;  
//     al2 = 315.0/4.0/M_PI;
//   } else{
//     al1 = 5.0/M_PI;  
//     al2 = 60.0/M_PI; 
//   }

//   rr = -1.0;
//   for (i = 1; i <= n; i++)
//     for (j = i; j <= n; j++) 
//       rr = MAX(rr,cut[i][j]);

//   nw_max = static_cast<int> (rr*dr_inv) + 2;
//   if (nw_max < 1 || nw_max > 600) error->all(FLERR,"Non-positive or too large value for nw_max to initialize weight arrays: PairSDPDVE2D::set_weight.");
//   memory->create(weight1,n+1,n+1,nw_max,"pair:weight1");
//   memory->create(weight2,n+1,n+1,nw_max,"pair:weight2");

//   for (i = 1; i <= n; i++)
//     for (j = i; j <= n; j++){
//       k = static_cast<int> (cut[i][j]*dr_inv) + 1;
//       if (k > nw_max) error->all(FLERR,"Error in PairSDPDVE2D::set_weight - k > nw_max");

//       if (cut[i][j] > 0.0){
//         for (l = 0; l < nw_max; l++){
//           rr = l*DR/cut[i][j];
//           if (rr > 1.0){
//             weight1[i][j][l] = 0.0;
//             weight2[i][j][l] = 0.0;
//           } else{
//             if (domain->dimension == 3) {
//               weight1[i][j][l] = al1*(1.0 + 3.0*rr)*(1.0-rr)*(1.0-rr)*(1.0-rr)/cutsq[i][j]/cut[i][j];
//               weight2[i][j][l] = al2*(1.0-rr)*(1.0-rr)/cutsq[i][j]/cutsq[i][j]/cut[i][j];
//             } else{
//               weight1[i][j][l] = al1*(1.0 + 3.0*rr)*(1.0-rr)*(1.0-rr)*(1.0-rr)/cutsq[i][j];
//               weight2[i][j][l] = al2*(1.0-rr)*(1.0-rr)/cutsq[i][j]/cutsq[i][j];
//             }
//           }
//           weight1[j][i][l] = weight1[i][j][l];
//           weight2[j][i][l] = weight2[i][j][l];
//         } 
//       } else{
//         for (l = 0; l < nw_max; l++){ 
//           weight1[i][j][l] = 0.0;
//           weight2[i][j][l] = 0.0;
//           weight1[j][i][l] = 0.0;
//           weight2[j][i][l] = 0.0;
//         }
//       }
//     }
// }
/* ---------------------------------------------------------------------- */

void PairSDPDVE2D::generate_wrr()
{
  int i;
  double ww[3][3];
  double v1, v2, factor, ss;

  for (i=0; i<5; i++){
    ss = 100.0;
    while ( ss > 1.0 ){
#ifdef MTRAND
      v1 = 2.0 * mtrand->rand() - 1.0;
      v2 = 2.0 * mtrand->rand() - 1.0;
#else
      v1 = 2.0 * random->uniform() - 1.0;
      v2 = 2.0 * random->uniform() - 1.0;
#endif
      ss = v1*v1 + v2*v2;
    }
    factor = sqrt(-2.0 * log(ss)/ss);
    if (i < 3){
      ww[i][0] = factor*v1;
      ww[i][1] = factor*v2;
    }
    else if (i == 3){
      ww[0][2] = factor*v1;
      ww[1][2] = factor*v2;
    }
    else
      ww[2][2] = factor*v1;
  }
  wrr[3] = (ww[0][0]+ww[1][1]+ww[2][2])/3.0;
  wrr[0] = (ww[0][0]-wrr[3])*delx + 0.5*(ww[0][1]+ww[1][0])*dely + 0.5*(ww[0][2]+ww[2][0])*delz;
  wrr[1] = 0.5*(ww[1][0]+ww[0][1])*delx + (ww[1][1]-wrr[3])*dely + 0.5*(ww[1][2]+ww[2][1])*delz;
  wrr[2] = 0.5*(ww[2][0]+ww[0][2])*delx + 0.5*(ww[2][1]+ww[1][2])*dely + (ww[2][2]-wrr[3])*delz;
}

/* ---------------------------------------------------------------------- */

int PairSDPDVE2D::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;
  double *rho = atom->rho;
  double ***kappa = atom->kappa;

  m = 0;
  for (i = 0; i < n; i ++) {
    j = list[i];
    buf[m++] = rho[j];
    buf[m++] = kappa[j][0][0];
    buf[m++] = kappa[j][0][1];
    buf[m++] = kappa[j][0][2];
    buf[m++] = kappa[j][1][0];
    buf[m++] = kappa[j][1][1];
    buf[m++] = kappa[j][1][2];
    buf[m++] = kappa[j][2][0];
    buf[m++] = kappa[j][2][1];
    buf[m++] = kappa[j][2][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairSDPDVE2D::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  double *rho = atom->rho;
  double ***kappa = atom->kappa;

  m = 0;
  last = first + n ;
  for (i = first; i < last; i++) {
    rho[i] = buf[m++];
    kappa[i][0][0] = buf[m++];
    kappa[i][0][1] = buf[m++];
    kappa[i][0][2] = buf[m++];
    kappa[i][1][0] = buf[m++];
    kappa[i][1][1] = buf[m++];
    kappa[i][1][2] = buf[m++];
    kappa[i][2][0] = buf[m++];
    kappa[i][2][1] = buf[m++];
    kappa[i][2][2] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int PairSDPDVE2D::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  double *rho = atom->rho;
  double ***kappa = atom->kappa;
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = rho[i];
    buf[m++] = kappa[i][0][0];
    buf[m++] = kappa[i][0][1];
    buf[m++] = kappa[i][0][2];
    buf[m++] = kappa[i][1][0];
    buf[m++] = kappa[i][1][1];
    buf[m++] = kappa[i][1][2];
    buf[m++] = kappa[i][2][0];
    buf[m++] = kappa[i][2][1];
    buf[m++] = kappa[i][2][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairSDPDVE2D::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  double *rho = atom->rho;
  double ***kappa = atom->kappa;
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    rho[j] += buf[m++];
    kappa[j][0][0] += buf[m++];
    kappa[j][0][1] += buf[m++];
    kappa[j][0][2] += buf[m++];
    kappa[j][1][0] += buf[m++];
    kappa[j][1][1] += buf[m++];
    kappa[j][1][2] += buf[m++];
    kappa[j][2][0] += buf[m++];
    kappa[j][2][1] += buf[m++];
    kappa[j][2][2] += buf[m++];
  }
}

/* ----------------------------------------------------------------------*/
