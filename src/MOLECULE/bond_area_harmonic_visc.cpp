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

// 2D area bond potential

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "bond_area_harmonic_visc.h"
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
#include "random_mars.h"
#include "MersenneTwister.h"

using namespace LAMMPS_NS;

#define  MOLINC 20

/* ---------------------------------------------------------------------- */

BondArea_Harmonic_VISC::BondArea_Harmonic_VISC(LAMMPS *lmp) : Bond(lmp) 
{
  random = NULL;
  mtrand = NULL;
  dath = datt = NULL;
  n_mol_limit = MOLINC;
  memory->create(dath,2*n_mol_limit,"bond_area_harmonic_visc:dath");
  memory->create(datt,2*n_mol_limit,"bond_area_harmonic_visc:datt");  
}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

BondArea_Harmonic_VISC::~BondArea_Harmonic_VISC()
{
  if (allocated) {
    memory->sfree(setflag);
    memory->sfree(ks);
    memory->sfree(ka);
    memory->sfree(kl);
    memory->sfree(circum);
    memory->sfree(area);
    memory->sfree(temp);
    memory->sfree(gamc);
    memory->sfree(gamt);
    memory->sfree(sigc);
    memory->sfree(sigt);
  }
  memory->destroy(dath);
  memory->destroy(datt);

  if (random) delete random;

#ifdef MTRAND
  if (mtrand) delete mtrand;
#endif
}

/* ---------------------------------------------------------------------- */

void BondArea_Harmonic_VISC::compute(int eflag, int vflag)
{
  int i1,i2,n,type,factor,k,l;
  tagint m, m1;
  double rsq,r,rfactor,l0,aa;
  double dvx, dvy, dvz;
  double coefa,vv;
  double xx1[3],xx2[3];
  double fbond, ebond, fr[3];
  double **xx = NULL;
  double **fa = NULL;
  int n_stress = output->n_stress;
  double ff[6];
  memory->create(fa,2,2,"bond_area_harmonic_visc:fa");
  memory->create(xx,2,3,"bond_area_harmonic_visc:fa");

  n_mol_max = atom->n_mol_max;
  if (n_mol_max > n_mol_limit){
    n_mol_limit = n_mol_max + MOLINC;
    memory->grow(dath,2*n_mol_limit,"bond_area_harmonic_visc:dath");
    memory->grow(datt,2*n_mol_limit,"bond_area_harmonic_visc:datt");
  }
  
  ebond = 0.0;
  eng_vdwl = 0.0;
  if(eflag || vflag) ev_setup(eflag, vflag);
  else evflag = 0;

  for (n = 0; n < 2*n_mol_max; n++){
    dath[n] = 0.0;
    datt[n] = 0.0;   
  }

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double *bondlist_length = neighbor->bondlist_length; 
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    m = atom->molecule[i1]-1;

    domain->unmap(x[i1],atom->image[i1],xx1);
    domain->unmap(x[i2],atom->image[i2],xx2);    

    aa = 0.5*(xx1[0]*xx2[1] - xx2[0]*xx1[1]);
    dath[m+n_mol_max] += aa;
    dath[m] += sqrt((xx2[0]-xx1[0])*(xx2[0]-xx1[0]) + (xx2[1]-xx1[1])*(xx2[1]-xx1[1]));
  }
  MPI_Allreduce(dath,datt,2*n_mol_max,MPI_DOUBLE,MPI_SUM,world);

  if (atom->mol_corresp_ind)
    for (n = 0; n < atom->n_mol_corresp_glob; n++){
      m = atom->mol_corresp_glob[2*n+1] - 1;
      m1 = atom->mol_corresp_glob[2*n] - 1;
      datt[m] = datt[m1];
      datt[m+n_mol_max] = datt[m1+n_mol_max];
    }  

  for (n = 0; n < nbondlist; n++) {
    
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    l0 = bondlist_length[n];
    m = atom->molecule[i1]-1;

    if (newton_bond) factor = 2;
    else {
      factor = 0;
      if (i1 < nlocal) factor++;
      if (i2 < nlocal) factor++;
    }
    rfactor = 0.5 * factor;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    dvx = v[i1][0] - v[i2][0];
    dvy = v[i1][1] - v[i2][1];
    dvz = v[i1][2] - v[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);  

    // force & energy
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

    fbond = - (ks[type]*(r - l0) + kl[type]*(datt[m] - circum[type]))/r;

    if (eflag)
      ebond = 0.5*ks[type]*(r-l0)*(r-l0);

    domain->unmap(x[i1],atom->image[i1],xx[0]);
    domain->unmap(x[i2],atom->image[i2],xx[1]);

    coefa = 0.5*ka[type]*(datt[m + n_mol_max]-area[type]);
    fa[0][0] = -coefa*xx[1][1];
    fa[0][1] = coefa*xx[1][0];
    fa[1][0] = -coefa*xx[0][1];
    fa[1][1] = coefa*xx[0][0];

    fr[0] = sigt[type]*wrr[0] - gamt[type]*dvx;
    fr[1] = sigt[type]*wrr[1] - gamt[type]*dvy;
    fr[2] = sigt[type]*wrr[2] - gamt[type]*dvz;
    //fr[0] = sigt[type]*wrr[0]/ra - gamt[type]*dvx;  // use with generate_wrr();
    //fr[1] = sigt[type]*wrr[1]/ra - gamt[type]*dvy;  // use with generate_wrr();
    //fr[2] = sigt[type]*wrr[2]/ra - gamt[type]*dvz;  // use with generate_wrr();

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond + fa[0][0] + fr[0];
      f[i1][1] += dely*fbond + fa[0][1] + fr[1];
      f[i1][2] += delz*fbond + fr[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond + fa[1][0] + fr[0];
      f[i2][1] -= dely*fbond + fa[1][1] + fr[1];
      f[i2][2] -= delz*fbond + fr[2];
    }

    if (evflag) {
      vv = 0.0;
      k = atom->mol_type[m];
      if (k > -1 && k < atom->n_mol_types)
        vv = coefa*datt[m + n_mol_max]/2.0/atom->mol_size[k];
      ev_tally3(i1,i2,nlocal,newton_bond,ebond,fa,fr,fbond,vv,xx,delx,dely,delz);
    }

    // virial contribution
    
    if (n_stress){
      vv = 0.0;
      k = atom->mol_type[m];
      if (k > -1 && k < atom->n_mol_types)      
        vv = coefa*datt[m + n_mol_max]/2.0/atom->mol_size[k];
      ff[0] = delx*(delx*fbond + fr[0]) - 0.5*(fa[1][0]+fa[0][0])*(xx[1][0]-xx[0][0]) - vv;
      ff[1] = dely*(dely*fbond + fr[1]) - 0.5*(fa[1][1]+fa[0][1])*(xx[1][1]-xx[0][1]) - vv;
      ff[2] = delz*(delz*fbond + fr[2]);
      ff[3] = delx*(dely*fbond + fr[1]) - 0.5*(fa[1][1]+fa[0][1])*(xx[1][0]-xx[0][0]);
      ff[4] = delx*(delz*fbond + fr[2]);
      ff[5] = dely*(delz*fbond + fr[2]);
      for (k = 0; k < n_stress; k++){
        l = output->stress_id[k];
        if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
          output->stat[l]->virial3(i1,i2,ff);
      }
    }
  }
  memory->destroy(fa);
  memory->destroy(xx);
}

/* ---------------------------------------------------------------------- */

void BondArea_Harmonic_VISC::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;
  int seed = 23;
  long unsigned int seed2;

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

  int individual = atom->individual;
  if(individual == 0)
    error->all(FLERR,"Individual has wrong value or is not set! Using bond area/harmonic/visc is only possible with individual = 1.");

  ks = (double *) memory->smalloc((n+1)*sizeof(double),"bond:ks");
  ka = (double *) memory->smalloc((n+1)*sizeof(double),"bond:ka");
  kl = (double *) memory->smalloc((n+1)*sizeof(double),"bond:kl");
  circum = (double *) memory->smalloc((n+1)*sizeof(double),"bond:circum");
  area = (double *) memory->smalloc((n+1)*sizeof(double),"bond:area");
  temp = (double *) memory->smalloc((n+1)*sizeof(double),"bond:temp");
  gamc = (double *) memory->smalloc((n+1)*sizeof(double),"bond:gamc");
  gamt = (double *) memory->smalloc((n+1)*sizeof(double),"bond:gamt");
  sigc = (double *) memory->smalloc((n+1)*sizeof(double),"bond:sigc");
  sigt = (double *) memory->smalloc((n+1)*sizeof(double),"bond:sigt");

  setflag = (int *) memory->smalloc((n+1)*sizeof(int),"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script
------------------------------------------------------------------------- */

void BondArea_Harmonic_VISC::coeff(int narg, char **arg)
{
  if (narg != 9) error->all(FLERR,"Incorrect args in bond_coeff command");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  double ks_one = force->numeric(FLERR,arg[1]);
  double kl_one = force->numeric(FLERR,arg[2]);
  double circum_one = force->numeric(FLERR,arg[3]);
  double ka_one = force->numeric(FLERR,arg[4]);
  double area_one = force->numeric(FLERR,arg[5]);
  double temp_one = force->numeric(FLERR,arg[6]);
  double gamc_one = force->numeric(FLERR,arg[7]);
  double gamt_one = force->numeric(FLERR,arg[8]);

  if(gamt_one > 3*gamc_one)
    error->all(FLERR,"Gamma_t > 3*Gamma_c");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    ks[i] = ks_one;
    kl[i] = kl_one;
    circum[i] = circum_one;
    ka[i] = ka_one;
    area[i] = area_one;
    temp[i] = temp_one;
    gamc[i] = gamc_one;
    gamt[i] = gamt_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args in bond_coeff command");
}

/* ---------------------------------------------------------------------- */
  
void BondArea_Harmonic_VISC::init_style()
{ 
  double sdtt = sqrt(update->dt);

  if (!allocated) error->all(FLERR,"Bond coeffs are not set");
  for (int i = 1; i <= atom->nbondtypes; i++){
    if (setflag[i] == 0) error->all(FLERR,"All bond coeffs are not set");
    if (gamt[i] > 3.0*gamc[i]) error->all(FLERR,"Gamma_t > 3*Gamma_c");
    sigc[i] = sqrt(2.0*temp[i]*(3.0*gamc[i]-gamt[i]))/sdtt;
    sigt[i] = 2.0*sqrt(gamt[i]*temp[i])/sdtt;
  } 
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length 
------------------------------------------------------------------------- */

double BondArea_Harmonic_VISC::equilibrium_distance(int i)
{
  return 0.0;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file 
------------------------------------------------------------------------- */

void BondArea_Harmonic_VISC::write_restart(FILE *fp)
{
  fwrite(&ks[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&kl[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&circum[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&ka[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&area[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&temp[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&gamc[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&gamt[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void BondArea_Harmonic_VISC::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&ks[1],sizeof(double),atom->nbondtypes,fp);
    fread(&kl[1],sizeof(double),atom->nbondtypes,fp);
    fread(&circum[1],sizeof(double),atom->nbondtypes,fp);
    fread(&ka[1],sizeof(double),atom->nbondtypes,fp);
    fread(&area[1],sizeof(double),atom->nbondtypes,fp);
    fread(&temp[1],sizeof(double),atom->nbondtypes,fp);
    fread(&gamc[1],sizeof(double),atom->nbondtypes,fp);
    fread(&gamt[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&ks[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&kl[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&circum[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&ka[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&area[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&temp[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&gamc[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&gamt[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondArea_Harmonic_VISC::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g %g %g\n",i,ks[i],kl[i],circum[i],ka[i],area[i],temp[i],gamc[i],gamt[i]);
}

/* ---------------------------------------------------------------------- */

double BondArea_Harmonic_VISC::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  fforce = 0.0;
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void BondArea_Harmonic_VISC::generate_wrr()
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

/* ----------------------------------------------------------------------*/
