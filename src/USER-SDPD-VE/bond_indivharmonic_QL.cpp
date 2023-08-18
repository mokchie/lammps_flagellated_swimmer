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
   Contributing author: Chaojie Mo (BUAA)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include "bond_indivharmonic_QL.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "random_mars.h"
#include "math_const.h"
#define EPS 1e-6

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

BondIndivHarmonicQL::BondIndivHarmonicQL(LAMMPS *lmp) : Bond(lmp) {
  random = NULL;
  mtrand = NULL;
  k = NULL;
  omega = NULL;
  r0min = NULL;
  r0max = NULL;
  ra = NULL;
  tau = NULL;
  alpha = NULL;
  gamma = NULL;
  epsilon = NULL;
  Nl = NULL;
  r0 = NULL;
  xtarget = NULL;
  ytarget = NULL;
  ztarget = NULL;  
  dist0 = NULL;
  dist = NULL;
  T_t = NULL;
  Qmatrix = NULL;
  reward = NULL;
  sa = NULL;
  Qallocated = false;
  first_learn = NULL;
  QL_on = NULL;
  Q0 = NULL;
  follow_btype = NULL;
}

/* ---------------------------------------------------------------------- */

BondIndivHarmonicQL::~BondIndivHarmonicQL()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(r0min);
    memory->destroy(r0max);
    memory->destroy(ra);
    memory->destroy(omega);
    memory->destroy(alpha);
    memory->destroy(gamma);
    memory->destroy(epsilon);
    memory->destroy(xtarget);
    memory->destroy(ytarget);
    memory->destroy(ztarget);
    memory->destroy(Q0);
    memory->destroy(Nl);
    memory->destroy(follow_btype);
    memory->destroy(tau);
    memory->destroy(dist0);
    memory->destroy(dist);
    memory->destroy(T_t);
    memory->destroy(reward);
    memory->destroy(sa);
    memory->destroy(first_learn);
    memory->destroy(QL_on);
  }
  if (Qallocated) memory->destroy(Qmatrix);
  if (random) delete random;

#ifdef MTRAND
  if (mtrand) delete mtrand;
#endif  
}

/* ---------------------------------------------------------------------- */

void BondIndivHarmonicQL::compute(int eflag, int vflag)
{
  int nbt = atom->nbondtypes;
  if (!Qallocated){
    int Nlmax = *std::max_element(Nl+1,Nl+nbt+1);
    memory->create(Qmatrix,nbt+1,Nlmax,3,"bond:Qmatrix");
    for (int i=1; i<nbt+1; i++){
      for (int j=0; j<Nl[i]; j++){
        for (int k=0; k<3; k++){
          Qmatrix[i][j][k] = Q0[i];
        }
      }
    }
    Qallocated = true;
  }
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
  int ii,imin1,imin1_all,imin2,imin2_all;
  int current_state;
  double qmax,rnd;
  int iqmax,irand;
  double xx1[3],xx2[3],xx1_all[3],xx2_all[3];
  for (i=1; i<nbt+1; i++){
    if (QL_on[i]){
      T_t[i]-=update->dt;
      if (T_t[i]<=0){
        //printf("r0[%d]=%f\n",i,r0[i]);
        //printf("r0[%d]=%f\n",follow_btype[i],r0[follow_btype[i]]);
        T_t[i]=fabs(2*MY_PI/omega[i]);
        imin1 = INT_MAX;
        imin2 = INT_MAX;
        for (n=0; n<nbondlist; n++){
          type = bondlist[n][2];
          if (type==i){
            i1 = bondlist[n][0];
            i2 = bondlist[n][1];
            ii = MIN(atom->tag[i1],atom->tag[i2]);
            imin1 = MIN(imin1,ii);
          }
          if (type==follow_btype[i]){
            i1 = bondlist[n][0];
            i2 = bondlist[n][1];
            ii = MIN(atom->tag[i1],atom->tag[i2]);
            imin2 = MIN(imin2,ii);
          }        
        }

        MPI_Allreduce(&imin1,&imin1_all,1,MPI_INT,MPI_MIN,world);
        MPI_Allreduce(&imin2,&imin2_all,1,MPI_INT,MPI_MIN,world);

        if (imin1==imin1_all){
          ii=atom->map(imin1);
          domain->unmap(x[ii],atom->image[ii],xx1);
        }
        else{
          xx1[0] = 0.0;
          xx1[1] = 0.0;
          xx1[2] = 0.0;
        }
        if (imin2==imin2_all){
          ii=atom->map(imin2);
          domain->unmap(x[ii],atom->image[ii],xx2);
        }
        else{
          xx2[0] = 0.0;
          xx2[1] = 0.0;
          xx2[2] = 0.0;
        }
        MPI_Allreduce(&xx1[0],&xx1_all[0],3,MPI_DOUBLE,MPI_SUM,world);
        MPI_Allreduce(&xx2[0],&xx2_all[0],3,MPI_DOUBLE,MPI_SUM,world);
        //printf("xx[0]=%f,xtarget=%f\n",(xx1_all[0]+xx2_all[0])/2,xtarget[i]);
        delx = (xx1_all[0]+xx2_all[0])/2 - xtarget[i];
        dely = (xx1_all[1]+xx2_all[1])/2 - ytarget[i];
        delz = (xx1_all[2]+xx2_all[2])/2 - ztarget[i];
        dist[i] = sqrt(delx*delx + dely*dely + delz*delz);
        if (comm->me==0) printf("dist[i]: %f\n",dist[i]);
        if (first_learn[i]){
          first_learn[i] = false;
          dist0[i] = dist[i];
          reward[i] = 0.0;
          sa[i][0] = int(round((r0[i]-r0min[i])/((r0max[i]-r0min[i])/(Nl[i]-1))));
          sa[i][1] = 1; // action = 1 means no action
        }
        else{
  #ifdef MTRAND    
          rnd = mtrand->rand(); 
  #else
          rnd = random->uniform();
  #endif
          MPI_Bcast(&rnd,1,MPI_DOUBLE,0,world);        
          //printf("rnd=%f\n",rnd);
          reward[i] = dist0[i]-dist[i];
          if (comm->me==0) printf("reward: %f\n",reward[i]);
          dist0[i] = dist[i];
          current_state = sa[i][0]+sa[i][1]-1;
          if (current_state == 0) {
            iqmax = std::distance(Qmatrix[i][current_state],std::max_element(Qmatrix[i][current_state]+1,Qmatrix[i][current_state]+3));
            if(rnd<0.5) irand = 1; else irand = 2;
          }
          else if (current_state == Nl[i]-1) {
            iqmax = std::distance(Qmatrix[i][current_state],std::max_element(Qmatrix[i][current_state],Qmatrix[i][current_state]+2));
            if(rnd<0.5) irand = 0; else irand = 1;
          }
          else {
            iqmax = std::distance(Qmatrix[i][current_state],std::max_element(Qmatrix[i][current_state],Qmatrix[i][current_state]+3));
            if(rnd<1.0/3.0) irand = 0; else if (rnd<2.0/3.0) irand = 1; else irand = 2;
          }
          qmax = Qmatrix[i][current_state][iqmax];
          if (comm->me==0) {
            printf("Q[%d] = [%f,%f,%f]\n",current_state,Qmatrix[i][current_state][0],Qmatrix[i][current_state][1],Qmatrix[i][current_state][2]);
            printf("state:%d optimal action:%d\n",current_state,iqmax-1);
            printf("alpha=%f,reward[i]=%f,gamma=%f,qmax=%f,Q[s][a]=%f\n",alpha[i],reward[i],gamma[i],qmax,Qmatrix[i][sa[i][0]][sa[i][1]]);
          }
          Qmatrix[i][sa[i][0]][sa[i][1]] += alpha[i] * (reward[i] + gamma[i]*qmax - Qmatrix[i][sa[i][0]][sa[i][1]]);
          sa[i][0] = current_state; //update the state;
  #ifdef MTRAND    
          rnd = mtrand->rand(); 
  #else
          rnd = random->uniform();
  #endif
          MPI_Bcast(&rnd,1,MPI_DOUBLE,0,world);
          //printf("rnd=%f\n",rnd);        
          if(rnd>epsilon[i]){
            //sa[i][0] += iqmax-1;
            sa[i][1] = iqmax;
            r0[i] = r0min[i] + (sa[i][0]+sa[i][1]-1) * ((r0max[i]-r0min[i])/(Nl[i]-1));
          } else {
            //sa[i][0] += irand-1;
            sa[i][1] = irand;
            r0[i] = r0min[i] + (sa[i][0]+sa[i][1]-1) * ((r0max[i]-r0min[i])/(Nl[i]-1));
          }
          if (comm->me==0) printf("realistic action:%d\n",sa[i][1]-1);
        }
      }    
    }
  }
  for (i=1; i<nbt+1; i++){
    if (!QL_on[i] && follow_btype[i]>0 && QL_on[follow_btype[i]])
        r0[i] = -r0[follow_btype[i]];
  }

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
    
    if (m>=0) {
      phi = bond_phase[i1][m];

    }
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

void BondIndivHarmonicQL::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;
  int seed;
  long unsigned int seed2;
  seed = comm->me+10000;

  memory->create(k,n+1,"bond:k");
  memory->create(r0,n+1,"bond:r0");
  memory->create(r0min,n+1,"bond:r0min");
  memory->create(r0max,n+1,"bond:r0max");
  memory->create(ra,n+1,"bond:ra");
  memory->create(omega,n+1,"bond:omega");
  memory->create(alpha,n+1,"bond:alpha");
  memory->create(gamma,n+1,"bond:gamma");
  memory->create(epsilon,n+1,"bond:epsilon");
  memory->create(xtarget,n+1,"bond:xtarget");
  memory->create(ytarget,n+1,"bond:ytarget");
  memory->create(ztarget,n+1,"bond:ztarget");  
  memory->create(Q0,n+1,"bond:Q0");
  memory->create(follow_btype,n+1,"bond:follow_btype");
  memory->create(Nl,n+1,"bond:Nl");
  memory->create(dist0,n+1,"bond:dist0");
  memory->create(dist,n+1,"bond:dist");
  memory->create(T_t,n+1,"bond:T_t");
  memory->create(reward,n+1,"bond:reward");
  memory->create(sa,n+1,2,"bond:sa");
  memory->create(first_learn,n+1,"bond:first_learn");
  memory->create(QL_on,n+1,"bond:QL_on");
  memory->create(tau,n+1,"bond:tau");
  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
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

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondIndivHarmonicQL::coeff(int narg, char **arg)
{
  if (narg < 14 && narg > 17) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  double k_one = force->numeric(FLERR,arg[1]);  
  double ra_one = force->numeric(FLERR,arg[2]);
  double omega_one = force->numeric(FLERR,arg[3]);
  double r0min_one = force->numeric(FLERR,arg[4]);
  double r0max_one = force->numeric(FLERR,arg[5]);
  int Nl_one = force->inumeric(FLERR,arg[6]);
  double alpha_one = force->numeric(FLERR,arg[7]);
  if (alpha_one>1.0 || alpha_one<0.0)
    error->all(FLERR,"Incorrect args for bond coefficients: alpha outside [0,1]");  
  double gamma_one = force->numeric(FLERR,arg[8]);
  if (gamma_one>1.0 || gamma_one<0.0)
    error->all(FLERR,"Incorrect args for bond coefficients: gamma outside [0,1]");  
  double epsilon_one = force->numeric(FLERR,arg[9]);
  if (epsilon_one>1.0 || epsilon_one<0.0)
    error->all(FLERR,"Incorrect args for bond coefficients: epsilon outside [0,1]");  
  double xt_one = force->numeric(FLERR,arg[10]);
  double yt_one = force->numeric(FLERR,arg[11]);
  double zt_one = force->numeric(FLERR,arg[12]);
  double Q0_one = force->numeric(FLERR,arg[13]);
  double tau_one = 0.0;
  if (narg>=15)
    tau_one = force->numeric(FLERR,arg[14]);
  int follow_btype_one = 0;
  if (narg>=16)
    follow_btype_one = force->inumeric(FLERR,arg[15]);
  int QL_on_one = 0;
  if (narg>=17){
    if (strcmp(arg[16],"yes") == 0) QL_on_one = 1;
    else {
      if (strcmp(arg[16],"no") == 0) QL_on_one = 0;
      else error->all(FLERR,"Illegal bond coeff command!"); 
    }
  }
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    ra[i] = ra_one;
    omega[i] = omega_one;
    r0min[i] = r0min_one;
    r0max[i] = r0max_one;
    Nl[i] = Nl_one;
    r0[i] = r0min_one + int((Nl_one-1)/2)*((r0max_one-r0min_one)/(Nl_one-1));
    
    alpha[i] = alpha_one;
    gamma[i] = gamma_one;
    epsilon[i] = epsilon_one;
    xtarget[i] = xt_one;
    ytarget[i] = yt_one;
    ztarget[i] = zt_one;
    T_t[i] = fabs(2*MY_PI/omega_one);
    first_learn[i] = true;    
    Q0[i] = Q0_one;
    tau[i] = tau_one;
    follow_btype[i] = follow_btype_one;
    QL_on[i] = QL_on_one;

    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondIndivHarmonicQL::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondIndivHarmonicQL::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&ra[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&omega[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0min[1],sizeof(double),atom->nbondtypes,fp);  
  fwrite(&r0max[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&Nl[1],sizeof(int),atom->nbondtypes,fp); 
  fwrite(&alpha[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&gamma[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&epsilon[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&xtarget[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&ytarget[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&ztarget[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&Q0[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&tau[1],sizeof(double),atom->nbondtypes,fp); 
  fwrite(&follow_btype[1],sizeof(int),atom->nbondtypes,fp); 
  fwrite(&QL_on[1],sizeof(int),atom->nbondtypes,fp); 
  fwrite(&tau[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&reward[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&sa[1][0],sizeof(int),atom->nbondtypes*2,fp);  
  fwrite(&dist0[1],sizeof(double),atom->nbondtypes,fp);  
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondIndivHarmonicQL::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->nbondtypes,fp);
    fread(&ra[1],sizeof(double),atom->nbondtypes,fp);
    fread(&omega[1],sizeof(double),atom->nbondtypes,fp);
    fread(&r0min[1],sizeof(double),atom->nbondtypes,fp);    
    fread(&r0max[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&r0[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&Nl[1],sizeof(int),atom->nbondtypes,fp); 
    fread(&alpha[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&gamma[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&epsilon[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&xtarget[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&ytarget[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&ztarget[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&Q0[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&tau[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&follow_btype[1],sizeof(int),atom->nbondtypes,fp); 
    fread(&QL_on[1],sizeof(int),atom->nbondtypes,fp); 
    fread(&reward[1],sizeof(double),atom->nbondtypes,fp); 
    fread(&sa[1],sizeof(int),atom->nbondtypes*2,fp); 
    fread(&dist0[1],sizeof(double),atom->nbondtypes,fp); 
  }
  MPI_Bcast(&k[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&ra[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&omega[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0min[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&r0max[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&Nl[1],atom->nbondtypes,MPI_INT,0,world); 
  MPI_Bcast(&alpha[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&gamma[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&epsilon[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&xtarget[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&ytarget[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&ztarget[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&Q0[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&tau[1],atom->nbondtypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&follow_btype[1],atom->nbondtypes,MPI_INT,0,world); 
  MPI_Bcast(&QL_on[1],atom->nbondtypes,MPI_INT,0,world); 
  MPI_Bcast(&reward[1],atom->nbondtypes,MPI_DOUBLE,0,world); 
  MPI_Bcast(&sa[1],atom->nbondtypes*2,MPI_INT,0,world); 
  MPI_Bcast(&dist0[1],atom->nbondtypes,MPI_DOUBLE,0,world); 


  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondIndivHarmonicQL::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %d %g %g %g %g %g %g %g %g %g %d %d\n",i,k[i],ra[i],omega[i],r0min[i],r0max[i],Nl[i],r0[i],alpha[i],gamma[i],epsilon[i],xtarget[i], ytarget[i], ztarget[i], Q0[i],tau[i],follow_btype[i],QL_on[i]);
}

/* ---------------------------------------------------------------------- */

double BondIndivHarmonicQL::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - ra[type]-r0[type];
  double rk = k[type] * dr;
  fforce = 0;
  if (r > 0.0) fforce = -2.0*rk/r;
  return rk*dr;
}
void BondIndivHarmonicQL::write_Qmatrix(FILE *fp)
{ if (Qallocated) {
    int n=0;
    for (int i=1; i <= atom->nbondtypes; i++){
      if (QL_on[i]) n+=1;
    }
    fprintf(fp,"%lld %d\n",update->ntimestep,n);
    for (int i=1; i <= atom->nbondtypes; i++){
      if (QL_on[i]){
        fprintf(fp,"%d %d %g\n",i,Nl[i],r0[i]);
        for (int j=0; j<Nl[i]; j++){
          fprintf(fp,"%d %d %g %g %g\n",i,j,Qmatrix[i][j][0],Qmatrix[i][j][1],Qmatrix[i][j][2]);
        }
      }
    }
  }
}