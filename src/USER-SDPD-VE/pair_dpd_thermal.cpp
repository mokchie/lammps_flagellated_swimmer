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
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "pair_dpd_thermal.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "random_mars.h"
#include "MersenneTwister.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-10
#define DR 0.01

/* ---------------------------------------------------------------------- */

PairDPDThermal::PairDPDThermal(LAMMPS *lmp) : Pair(lmp)
{
  random = NULL;
  mtrand = NULL;
  weight = NULL;
  set_weight_ind = 1;
  dr_inv = 1.0/DR;
}

/* ---------------------------------------------------------------------- */

PairDPDThermal::~PairDPDThermal()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(a0);
    memory->destroy(kappa0);
    memory->destroy(sigma);
    memory->destroy(w_exp);
  }

  if (random) delete random;

#ifdef MTRAND
  if (mtrand) delete mtrand;
#endif
  memory->destroy(weight);
}

/* ---------------------------------------------------------------------- */

void PairDPDThermal::compute(int eflag, int vflag)
{
  int i,j,k,l,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair,eij;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double q1, q2, q3, gamma, kappa, alpha, engh;
  double rsq,r,rinv,dot,wd,wdg,wdt,randnum,randnum1,factor_dpd;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  if (set_weight_ind){
    set_weight();
    set_weight_ind = 0;
  }

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *e = atom->e;
  double *de = atom->de;
  double *pte1 = atom->pte1;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);
  double shift = sqrt(3.0);
  int n_stress = output->n_stress;
  int ind_stress;
  double ff[6];

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

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

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
        rinv = 1.0/r;
        delvx = vxtmp - v[j][0];
        delvy = vytmp - v[j][1];
        delvz = vztmp - v[j][2];
        dot = (delx*delvx + dely*delvy + delz*delvz)*rinv;
        wd = 1.0 - r/cut[itype][jtype];
#ifdef MTRAND
        randnum = shift*(2.0*mtrand->rand()-1.0);
        randnum1 = shift*(2.0*mtrand->rand()-1.0);
#else
        randnum = shift*(2.0*random->uniform()-1.0);
        randnum1 = shift*(2.0*random->uniform()-1.0);
#endif
        k = static_cast<int> (r*dr_inv);
        wdg = (weight[itype][jtype][k+1]-weight[itype][jtype][k])*(r*dr_inv - k) + weight[itype][jtype][k];
        wdt = wdg;

        eij = 0.5*(e[i]+e[j]);
        gamma = 0.5*sigma[itype][jtype]*sigma[itype][jtype]*eij/e[i]/e[j];
        fpair = a0[itype][jtype]*eij*wd;
        fpair -= gamma*wdg*wdg*dot; 
        fpair += sigma[itype][jtype]*wdg*randnum*dtinvsqrt;
        fpair *= factor_dpd*rinv;
        kappa = c_v*c_v*kappa0[itype][jtype]*eij*eij;
        alpha = sqrt(2.0*kappa);
        q1 = kappa*wdt*wdt*(1.0/e[i] - 1.0/e[j]) + alpha*wdt*dtinvsqrt*randnum1;
        //q2 = wd*a0[itype][jtype]*eij*dot + 0.5*wdg*dot*(wdg*gamma*dot - sigma[itype][jtype]*randnum*dtinvsqrt);
        //q2 = 0.5*wdg*dot*(wdg*gamma*dot - sigma[itype][jtype]*randnum*dtinvsqrt);  
        //q3 = - 0.5*wdg*wdg*sigma[itype][jtype]*sigma[itype][jtype];
        engh = 0.25*factor_dpd*a0[itype][jtype]*eij*cut[itype][jtype]*wd*wd;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        //de[i] += (q1 + q2 + q3/mass[itype])/c_v;
        de[i] += q1/c_v;
        pte1[i] += engh;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
          //de[j] += (q2 - q1 + q3/mass[jtype])/c_v;
          de[j] -= q1/c_v;
          pte1[j] += engh;
        }

        if (eflag) evdwl = 2.0*engh;

        if (n_stress){
          ind_stress = 0;
          ff[0] = delx*delx*fpair;
          ff[1] = dely*dely*fpair;
          ff[2] = delz*delz*fpair;
          ff[3] = delx*dely*fpair;
          ff[4] = delx*delz*fpair;
          ff[5] = dely*delz*fpair;
          if (itype != jtype) ind_stress = 1;
          if (itype == 2 && jtype == 2) ind_stress = 2;
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

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairDPDThermal::allocate()
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

  memory->create(a0,n+1,n+1,"pair:a0");
  memory->create(kappa0,n+1,n+1,"pair:kappa0");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(w_exp,n+1,n+1,"pair:w_exp");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDPDThermal::settings(int narg, char **arg)
{
  long unsigned int seed2;

  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  c_v = force->numeric(FLERR,arg[0]);
  cut_global = force->numeric(FLERR,arg[1]);
  seed = force->inumeric(FLERR,arg[2]);
  atom->c_v = c_v;

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

void PairDPDThermal::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 7) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double a0_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);
  double kappa0_one = force->numeric(FLERR,arg[4]);
  double w_exp_one = force->numeric(FLERR,arg[5]);

  double cut_one = cut_global;
  if (narg == 7) cut_one = force->numeric(FLERR,arg[6]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a0[i][j] = a0_one;
      sigma[i][j] = sigma_one;
      kappa0[i][j] = kappa0_one;
      w_exp[i][j] = w_exp_one;
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

void PairDPDThermal::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair dpd/thermal requires ghost atoms store velocity");

  // if newton off, forces between atoms ij will be double computed
  // using different random numbers

  if (force->newton_pair == 0 && comm->me == 0) error->warning(FLERR,
      "Pair dpd/thermal needs newton pair on for momentum conservation");

  neighbor->request(this,instance_me);
  set_weight_ind = 1;
  atom->c_v = c_v;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDPDThermal::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  cut[j][i] = cut[i][j];
  a0[j][i] = a0[i][j];
  sigma[j][i] = sigma[i][j];
  kappa0[j][i] = kappa0[i][j];
  w_exp[j][i] = w_exp[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPDThermal::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a0[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&kappa0[i][j],sizeof(double),1,fp);
        fwrite(&w_exp[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPDThermal::read_restart(FILE *fp)
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
          fread(&a0[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&kappa0[i][j],sizeof(double),1,fp);
          fread(&w_exp[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&a0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&kappa0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&w_exp[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPDThermal::write_restart_settings(FILE *fp)
{
  fwrite(&c_v,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&seed,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPDThermal::read_restart_settings(FILE *fp)
{
  long unsigned int seed2;

  if (comm->me == 0) {
    fread(&c_v,sizeof(double),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&seed,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&c_v,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&seed,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);

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

double PairDPDThermal::single(int i, int j, int itype, int jtype, double rsq,
                       double factor_coul, double factor_dpd, double &fforce)
{
  double r,rinv,wd,phi;

  r = sqrt(rsq);
  if (r < EPSILON) {
    fforce = 0.0;
    return 0.0;
  }

  rinv = 1.0/r;
  wd = 1.0 - r/cut[itype][jtype];
  fforce = a0[itype][jtype]*wd * factor_dpd*rinv;

  phi = 0.5*a0[itype][jtype]*cut[itype][jtype] * wd*wd;
  return factor_dpd*phi;
}

/* ---------------------------------------------------------------------- */

void PairDPDThermal::set_weight()
{
  int i,j,k,l;
  int n = atom->ntypes;
  double rr,wd;
 
  if (weight) { 
    memory->destroy(weight);
    weight = NULL;
  }

  rr = -1.0;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++) 
      rr = MAX(rr,cut[i][j]);

  nw_max = static_cast<int> (rr*dr_inv) + 2;
  if (nw_max < 1 || nw_max > 600) error->all(FLERR,"Non-positive or too large value for nw_max to initialize weight arrays: PairDPD::set_weight.");
  memory->create(weight,n+1,n+1,nw_max,"pair:weight");
 
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++){
      k = static_cast<int> (cut[i][j]*dr_inv) + 1;
      if (k > nw_max) error->all(FLERR,"Error in PairDPD::set_weight - k > nw_max");
      for (l = 0; l < nw_max; l++){
        rr = l*DR;
        if (rr > cut[i][j])
          wd = 0.0;
        else  
          wd = 1.0 - rr/cut[i][j];
        weight[i][j][l] = pow(wd,w_exp[i][j]);
        weight[j][i][l] = weight[i][j][l]; 
      }
    }
}

/* ---------------------------------------------------------------------- */
