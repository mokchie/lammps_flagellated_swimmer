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
#include <cstring>
#include "bond_expon.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondExpon::BondExpon(LAMMPS *lmp) : Bond(lmp)
{
  reinitflag = 1;
}

/* ---------------------------------------------------------------------- */

BondExpon::~BondExpon()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(rc);
    memory->destroy(rmax);
    memory->destroy(fmax);
  }
}

/* ---------------------------------------------------------------------- */

void BondExpon::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,dr,drm,drinv,drminv;

  ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    dr = rc[type] - r;
    drm = rmax[type] - r;
    drinv = 1.0/dr;
    drminv = 1.0/drm;

    // force & energy

    if (r < rc[type])
      fbond = 0.0;
    else if (r >= rmax[type])
      fbond = -fmax[type];
    else {
      fbond = -k[type] * exp(drinv) * drinv*drinv * drminv*drminv * (rc[type]*rc[type] - 2.0*rc[type]*r + rmax[type] + rsq - r) / r;
      if (fbond < -fmax[type])
        fbond = -fmax[type];
    }

    if (eflag) ebond = k[type]*drminv*exp(drinv);

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

void BondExpon::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k,n+1,"bond:k");
  memory->create(rc,n+1,"bond:rc");
  memory->create(rmax,n+1,"bond:rmax");
  memory->create(fmax,n+1,"bond:fmax");

  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondExpon::coeff(int narg, char **arg)
{
  if (narg != 5) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  double k_one = force->numeric(FLERR,arg[1]);
  double rc_one = force->numeric(FLERR,arg[2]);
  double rmax_one = force->numeric(FLERR,arg[3]);
  double fmax_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    rc[i] = rc_one;
    rmax[i] = rmax_one;
    fmax[i] = fmax_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondExpon::equilibrium_distance(int i)
{
  return rc[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondExpon::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&rc[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&rmax[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&fmax[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondExpon::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->nbondtypes,fp);
    fread(&rc[1],sizeof(double),atom->nbondtypes,fp);
    fread(&rmax[1],sizeof(double),atom->nbondtypes,fp);
    fread(&fmax[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&k[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&rc[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&rmax[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&fmax[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondExpon::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",i,k[i],rc[i],rmax[i],fmax[i]);
}

/* ---------------------------------------------------------------------- */

double BondExpon::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - rc[type];
  double rk = k[type] * dr;
  fforce = 0;
  if (r > 0.0) fforce = -2.0*rk/r;
  return rk*dr;
}

/* ----------------------------------------------------------------------
    Return ptr to internal members upon request.
------------------------------------------------------------------------ */
void *BondExpon::extract( char *str, int &dim )
{
  dim = 1;
  if( strcmp(str,"kappa")==0) return (void*) k;
  if( strcmp(str,"rc")==0) return (void*) rc;
  return NULL;
}


