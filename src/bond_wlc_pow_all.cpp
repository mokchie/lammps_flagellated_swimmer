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

// WLC bond potential

#include <cstdlib>
#include <cmath>
#include <cstring>
#include "bond_wlc_pow_all.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondWLC_POW_ALL::BondWLC_POW_ALL(LAMMPS *lmp) : Bond(lmp) {}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

BondWLC_POW_ALL::~BondWLC_POW_ALL()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(temp);
    memory->destroy(r0);
    memory->destroy(mu_targ);
    memory->destroy(qp);
    memory->destroy(spring_scale);
  }
}

/* ---------------------------------------------------------------------- */

void BondWLC_POW_ALL::compute(int eflag, int vflag)
{
  int i1,i2,n,type,k,l;
  double rr,ra,rlogarg,kph,l0,lmax,mu,lambda, ebond, fbond,rrs,lh;
  double delx, dely, delz, rsq;
  int n_stress = output->n_stress;
  double ff[6];

  ebond = 0.0;
  if(eflag || vflag) ev_setup(eflag, vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double *bondlist_length = neighbor->bondlist_length;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  for (n = 0; n < nbondlist; n++) {
    
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    l0 = bondlist_length[n]*spring_scale[type];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    // force from log term
    rsq = delx*delx + dely*dely + delz*delz;
    ra = sqrt(rsq);
    lmax = l0*r0[type];
    rr = 1.0/r0[type];
    rrs = rr/spring_scale[type];
    lh = l0/spring_scale[type];
    kph = pow(l0,qp[type])*temp[type]*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr);
    mu = 0.25*sqrt(3.0)*(temp[type]*(-0.25/(1.0-rrs)/(1.0-rrs) + 0.25 + 0.5*rrs/(1.0-rrs)/(1.0-rrs)/(1.0-rrs))/lh + kph*(qp[type]+1.0)/pow(lh,qp[type]+1.0)) + sqrt(3.0)*(temp[type]*(0.25/(1.0-rrs)/(1.0-rrs) - 0.25 + rrs)/lh - kph/pow(lh,qp[type]+1.0));
    lambda = mu/mu_targ[type];
    kph = kph*mu_targ[type]/mu;
    rr = ra/lmax; 
    rlogarg = pow(ra,qp[type]+1.0);

    if (rr >= 1.0) {
      char warning[128];
      sprintf(warning,"WLC bond too long: " BIGINT_FORMAT " "
              TAGINT_FORMAT " " TAGINT_FORMAT " %g",
              update->ntimestep,atom->tag[i1],atom->tag[i2],rr);
      error->warning(FLERR, warning, 0);
    }   

    fbond = - temp[type]*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr)/lambda/ra + kph/rlogarg;

    // force & energy
    if(eflag){
      ebond = 0.25*temp[type]*lmax*(3.0*rr*rr-2.0*rr*rr*rr)/(1.0-rr)/lambda;
      if (qp[type] == 1.0)
        ebond -= kph*log(ra);
      else
        ebond += kph/(qp[type]-1.0)/pow(ra,qp[type]-1.0);
    }
    
    // apply force to each of 2 atoms
    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;
    }

    // virial contribution

    if (n_stress){
      ff[0] = delx*delx*fbond;
      ff[1] = dely*dely*fbond;
      ff[2] = delz*delz*fbond;
      ff[3] = delx*dely*fbond;
      ff[4] = delx*delz*fbond;
      ff[5] = dely*delz*fbond;
      for (k = 0; k < n_stress; k++){
        l = output->stress_id[k];
        if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
          output->stat[l]->virial3(i1,i2,ff);
      }
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondWLC_POW_ALL::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  int individual = atom->individual;
  if (individual == 0)
    error->all(FLERR,"Individual has wrong value or is not set! Using bond wlc/pow/all only possible with individual =1");  

  memory->create(temp,n+1,"bond:temp");
  memory->create(r0,n+1,"bond:r0");
  memory->create(mu_targ,n+1,"bond:mu_targ");
  memory->create(qp,n+1,"bond:qp");
  memory->create(spring_scale,n+1,"bond:spring_scale");

  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script for one or more types
------------------------------------------------------------------------- */

void BondWLC_POW_ALL::coeff(int narg, char **arg)
{
  if (narg != 6) error->all(FLERR,"Incorrect args in bond_coeff command");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  double temp_one = force->numeric(FLERR,arg[1]);
  double r0_one = force->numeric(FLERR,arg[2]);
  double mu_one = force->numeric(FLERR,arg[3]);
  double qp_one = force->numeric(FLERR,arg[4]);
  double spring_scale_one = force->numeric(FLERR,arg[5]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    temp[i] = temp_one;
    r0[i] = r0_one;
    mu_targ[i] = mu_one;
    qp[i] = qp_one;
    spring_scale[i] = spring_scale_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args in bond_coeff command");
}

/* ---------------------------------------------------------------------- */

void BondWLC_POW_ALL::init_style()
{
  if (!allocated) error->all(FLERR,"Bond coeffs are not set");
  //for (int i = 1; i <= atom->nbondtypes; i++)
    //if (setflag[i] == 0) error->all(FLERR,"All bond coeffs are not set"); 
}

/* ---------------------------------------------------------------------- */

double BondWLC_POW_ALL::equilibrium_distance(int i)
{
  return r0[i];
}


/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file 
------------------------------------------------------------------------- */

void BondWLC_POW_ALL::write_restart(FILE *fp)
{
  fwrite(&temp[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&mu_targ[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&qp[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&spring_scale[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void BondWLC_POW_ALL::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&temp[1],sizeof(double),atom->nbondtypes,fp);
    fread(&r0[1],sizeof(double),atom->nbondtypes,fp);
    fread(&mu_targ[1],sizeof(double),atom->nbondtypes,fp);
    fread(&qp[1],sizeof(double),atom->nbondtypes,fp);
    fread(&spring_scale[1],sizeof(double),atom->nbondtypes,fp);  
  }
  MPI_Bcast(&temp[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&mu_targ[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&qp[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&spring_scale[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondWLC_POW_ALL::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %g\n",i,temp[i],r0[i],mu_targ[i],qp[i],spring_scale[i]);
}

/* ----------------------------------------------------------------------*/

double BondWLC_POW_ALL::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  char warning[128];

  double rr = rsq;

  if (rr >= 1.0) {
    sprintf(warning,"WLC single bond too long: " BIGINT_FORMAT " %g",update->ntimestep,rr);
    error->warning(FLERR,warning,0);
  }

  fforce = 0.0;
  return 0.0;
}

/* ----------------------------------------------------------------------*/

