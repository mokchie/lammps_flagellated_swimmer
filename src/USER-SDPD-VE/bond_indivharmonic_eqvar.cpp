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

#include <cmath>
#include <cstdlib>
#include "bond_indivharmonic_eqvar.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#define EPS 1e-6

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondIndivHarmonicEqvar::BondIndivHarmonicEqvar(LAMMPS *lmp) : Bond(lmp) {}

/* ---------------------------------------------------------------------- */

BondIndivHarmonicEqvar::~BondIndivHarmonicEqvar()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(ra);
    memory->destroy(omega);
    memory->destroy(tau);
  }
}

/* ---------------------------------------------------------------------- */

void BondIndivHarmonicEqvar::compute(int eflag, int vflag)
{
  int i1,i2,n,type,m,i;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,dr,rk;
  double phi;

  ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double *bondlist_length = neighbor->bondlist_length;
  double **bond_phase = atom->bond_phase;
  int *num_bond = atom->num_bond; 
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  double l0;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    m=-1;
    for (i=0; i<num_bond[i1]; i++){
      if (atom->tag[i2] == bond_atom[i1][i]){
        m=i;
        break;
      }
    }
    if (m>=0) phi = bond_phase[i1][m];
    else {
      for (i=0; i<num_bond[i2]; i++){
        if (atom->tag[i1] == bond_atom[i2][i]){
          m=i;
          break;
        }
      }
      if (m>=0) phi = bond_phase[i2][m];
    }

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    if (tau[type]>=EPS)
      l0 = bondlist_length[n] + (1.0-exp(-update->ntimestep*update->dt/tau[type])) * (r0[type] + ra[type] * sin(omega[type]*update->ntimestep*update->dt + phi));
    else
      l0 = bondlist_length[n] + r0[type] + ra[type] * sin(omega[type]*update->ntimestep*update->dt + phi);
    dr = r - l0;
    rk = k[type] * dr;
    // printf("indivHarmonic: equilibrium bond %d (particles %d and %d), is %f\n", n, i1, i2, l0);
    // force & energy
    if (r > 0.0) fbond = -2.0*rk/r;
    else fbond = 0.0;

    if (eflag) ebond = rk*dr;

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

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondIndivHarmonicEqvar::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k,n+1,"bond:k");
  memory->create(r0,n+1,"bond:r0");
  memory->create(ra,n+1,"bond:ra");
  memory->create(omega,n+1,"bond:omega");
  memory->create(tau,n+1,"bond:tau");

  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondIndivHarmonicEqvar::coeff(int narg, char **arg)
{
  if (narg < 4 && narg > 6) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  double k_one = force->numeric(FLERR,arg[1]);  
  double ra_one = force->numeric(FLERR,arg[2]);
  double omega_one = force->numeric(FLERR,arg[3]);
  double r0_one = 0.0;
  double tau_one = 0.0;
  if (narg>=5)
	  r0_one = force->numeric(FLERR,arg[4]);
  if (narg>=6)
    tau_one = force->numeric(FLERR,arg[5]);
  

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    ra[i] = ra_one;
    r0[i] = r0_one;
    omega[i] = omega_one;
    tau[i] = tau_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondIndivHarmonicEqvar::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondIndivHarmonicEqvar::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&ra[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&omega[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);  
  fwrite(&tau[1],sizeof(double),atom->nbondtypes,fp);

}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondIndivHarmonicEqvar::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->nbondtypes,fp);
    fread(&ra[1],sizeof(double),atom->nbondtypes,fp);
    fread(&omega[1],sizeof(double),atom->nbondtypes,fp);
    fread(&r0[1],sizeof(double),atom->nbondtypes,fp);    
    fread(&tau[1],sizeof(double),atom->nbondtypes,fp); 
  }
  MPI_Bcast(&k[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&ra[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&omega[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&tau[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondIndivHarmonicEqvar::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %g\n",i,k[i],ra[i],omega[i],r0[i],tau[i]);
}

/* ---------------------------------------------------------------------- */

double BondIndivHarmonicEqvar::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - ra[type]-r0[type];
  double rk = k[type] * dr;
  fforce = 0;
  if (r > 0.0) fforce = -2.0*rk/r;
  return rk*dr;
}