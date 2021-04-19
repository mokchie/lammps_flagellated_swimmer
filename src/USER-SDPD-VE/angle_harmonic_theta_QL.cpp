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
   Contributing author: Naveen Michaud-Agrawal (Johns Hopkins U)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include "angle_harmonic_theta_QL.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001
#define EPS 1e-6


/* ---------------------------------------------------------------------- */

AngleHarmonicThetaQL::AngleHarmonicThetaQL(LAMMPS *lmp) : Angle(lmp)
{
  random = NULL;
  mtrand = NULL;
  k = NULL;
  b = NULL;
  omega = NULL;
  theta0max = NULL;
  skew = NULL;
  tau = NULL;
  alpha = NULL;
  gamma = NULL;
  epsilon = NULL;
  Nl = NULL;
  theta0 = NULL;
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
  Q0 = NULL;
  //imin = NULL;
}

/* ---------------------------------------------------------------------- */

AngleHarmonicThetaQL::~AngleHarmonicThetaQL()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(b);
    memory->destroy(omega); 
    memory->destroy(theta0max);
    memory->destroy(skew);
    memory->destroy(tau);
    memory->destroy(alpha);    
    memory->destroy(gamma);
    memory->destroy(epsilon);   
    memory->destroy(Nl); 
    memory->destroy(theta0);
    memory->destroy(xtarget);
    memory->destroy(ytarget);    
    memory->destroy(ztarget);  
    memory->destroy(dist0);      
    memory->destroy(dist);    
    memory->destroy(T_t);    
    memory->destroy(reward);
    memory->destroy(sa);
    memory->destroy(first_learn);
    memory->destroy(Q0);
    //memory->destroy(imin);
  }
  if (Qallocated) memory->destroy(Qmatrix);
  if (random) delete random;

#ifdef MTRAND
  if (mtrand) delete mtrand;
#endif
}

/* ---------------------------------------------------------------------- */

void AngleHarmonicThetaQL::compute(int eflag, int vflag)
{
  int nat = atom->nangletypes;
  if (!Qallocated){
    int Nlmax = *std::max_element(Nl+1,Nl+nat+1);
    memory->create(Qmatrix,nat+1,Nlmax,3,"angle:Qmatrix");
    for (int i=1; i<nat+1; i++){
      for (int j=0; j<Nl[i]; j++){
        for (int k=0; k<3; k++){
          Qmatrix[i][j][k] = Q0[i];
        }
      }
    }
    Qallocated = true;
  }

  int i1,i2,i3,n,type,nc,m=-1,i;
  double bi;
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double eangle,f1[3],f3[3];
  double dtheta,theta,thetaeq,tk,cross,phi;
  double rsq1,rsq2,r1,r2,c,s,a,a11,a12,a22;

  eangle = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  double *anglelist_area = neighbor->anglelist_area;
  double **angle_area = atom->angle_area;
  double **angle_area2 = atom->angle_area2;
  int *num_angle = atom->num_angle;
  int **angle_atom1 = atom->angle_atom1;
  int **angle_atom2 = atom->angle_atom2;
  int **angle_atom3 = atom->angle_atom3;
  int ii,imin,imin_all;
  int current_state;
  double qmax,rnd,dist_one;
  int iqmax,irand;
  double xx[3];
  //update_theta0();
  //printf("core=%d, omega=%f\n",comm->me,omg_t[1]);

  for (i=1; i<nat+1; i++){
    T_t[i]-=update->dt;
    if (T_t[i]<=0){
      T_t[i]=fabs(2*MY_PI/omega[i]);
      imin = INT_MAX;
      for (n=0; n<nanglelist; n++){
        type = anglelist[n][3];
        if (type==i){
          i1 = anglelist[n][0];
          i2 = anglelist[n][1];
          i3 = anglelist[n][2];
          ii = MIN(MIN(atom->tag[i1],atom->tag[i2]),atom->tag[i3]);
          imin = MIN(imin,ii);
        }
      }
      MPI_Allreduce(&imin,&imin_all,1,MPI_INT,MPI_MIN,world);

      if (imin==imin_all){
        ii=atom->map(imin);
        domain->unmap(x[ii],atom->image[ii],xx);
        delx1 = xx[0] - xtarget[i];
        dely1 = xx[1] - ytarget[i];
        delz1 = xx[2] - ztarget[i];
        dist_one = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);
      } else dist_one=0.0;
      printf("process %d, dist_one=%f\n",comm->me,dist_one);
      MPI_Allreduce(&dist_one,&dist[i],1,MPI_DOUBLE,MPI_SUM,world);
      if (first_learn[i]){
        first_learn[i] = false;
        dist0[i] = dist[i];
        reward[i] = 0.0;
        sa[i][0] = int(round((theta0[i]+theta0max[i])/(2*theta0max[i]/(Nl[i]-1))));
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
        if (comm->me==0) printf("Q[%d] = [%f,%f,%f]\n",current_state,Qmatrix[i][current_state][0],Qmatrix[i][current_state][1],Qmatrix[i][current_state][2]);
        if (comm->me==0) printf("state:%d optimal action:%d\n",current_state,iqmax-1);
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
          theta0[i] = -theta0max[i] + (sa[i][0]+sa[i][1]-1) * (2*theta0max[i]/(Nl[i]-1));
        } else {
          //sa[i][0] += irand-1;
          sa[i][1] = irand;
          theta0[i] = -theta0max[i] + (sa[i][0]+sa[i][1]-1) * (2*theta0max[i]/(Nl[i]-1));
        }
        if (comm->me==0) printf("realistic action:%d\n",sa[i][1]-1);
      }
    }    
  }


  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];
    //thx = anglelist_area[n];

    // 1st bond

    delx1 = x[i1][0] - x[i2][0];
    dely1 = x[i1][1] - x[i2][1];
    delz1 = x[i1][2] - x[i2][2];

    rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
    r1 = sqrt(rsq1);

    // 2nd bond

    delx2 = x[i3][0] - x[i2][0];
    dely2 = x[i3][1] - x[i2][1];
    delz2 = x[i3][2] - x[i2][2];

    rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    r2 = sqrt(rsq2);

    // angle (cos and sin)

    c = delx1*delx2 + dely1*dely2 + delz1*delz2;
    c /= r1*r2;

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    s = sqrt(1.0 - c*c);
    if (s < SMALL) s = SMALL;
    s = 1.0/s;

    // force & energy
    theta = MY_PI-acos(c);
    cross = delx1*dely2-delx2*dely1;
    if (cross > 0.0)
      theta = -theta;

    for (i=0; i<num_angle[i2]; i++){
      if (atom->tag[i1]==angle_atom1[i2][i] && atom->tag[i2]==angle_atom2[i2][i] && atom->tag[i3]==angle_atom3[i2][i]){
        m=i;
        break;
      }
    }
    if (m<0){
      error->all(FLERR,"unable to find the angle");
    }
    if(atom->individual>=2)
      bi = angle_area2[i2][m]/180.0*MY_PI;
    else
      bi = b[type];
    angle_area[i2][m] += omega[type]*update->dt;
    phi = angle_area[i2][m];
    nc = (int) (floor(phi/MY_PI*2.0));
    if (nc%2==1) nc--;
    if (phi < (nc+1)*MY_PI/2.0+skew[type])
      phi = shiftmap(phi, nc*MY_PI/2.0, (nc+1)*MY_PI/2.0+skew[type], nc*MY_PI/2.0, (nc+1)*MY_PI/2.0);
    else
      phi = shiftmap(phi, (nc+1)*MY_PI/2.0+skew[type], (nc+2)*MY_PI/2.0, (nc+1)*MY_PI/2.0, (nc+2)*MY_PI/2.0);

    if (tau[type]>=EPS)
      thetaeq = theta0[type] + (1.0-exp(-update->ntimestep*update->dt/tau[type])) * bi*sin(phi);
    else
      thetaeq = theta0[type] + bi*sin(phi);
    thetaeq -= 2.0*MY_PI * round(thetaeq/2.0/MY_PI);

    dtheta = theta - thetaeq;
    dtheta -= 2.0*MY_PI * round(dtheta/2.0/MY_PI);

    tk = k[type] * dtheta;


    if (eflag) eangle = tk*dtheta;
    if(cross > 0.0)
      a = -2.0 * tk *s;
    else
      a = 2.0 * tk * s;
    a11 = a*c / rsq1;
    a12 = -a / (r1*r2);
    a22 = a*c / rsq2;

    f1[0] = a11*delx1 + a12*delx2;
    f1[1] = a11*dely1 + a12*dely2;
    f1[2] = a11*delz1 + a12*delz2;
    f3[0] = a22*delx2 + a12*delx1;
    f3[1] = a22*dely2 + a12*dely1;
    f3[2] = a22*delz2 + a12*delz1;

    // apply force to each of 3 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= f1[0] + f3[0];
      f[i2][1] -= f1[1] + f3[1];
      f[i2][2] -= f1[2] + f3[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (evflag) ev_tally(i1,i2,i3,nlocal,newton_bond,eangle,f1,f3,
                         delx1,dely1,delz1,delx2,dely2,delz2);
  }
}

/* ---------------------------------------------------------------------- */

void AngleHarmonicThetaQL::allocate()
{
  allocated = 1;  
  int seed;
  long unsigned int seed2;
  seed = comm->me+10000;

  int n = atom->nangletypes;

  memory->create(k,n+1,"angle:k");
  memory->create(b,n+1,"angle:b");
  memory->create(omega,n+1,"angle:omega");
  memory->create(theta0max,n+1,"angle:theta0max");
  memory->create(skew,n+1,"angle:skew");
  memory->create(setflag,n+1,"angle:setflag"); 
  memory->create(tau,n+1,"angle:tau");  
  memory->create(alpha,n+1,"angle:alpha");
  memory->create(gamma,n+1,"angle:gamma");  
  memory->create(epsilon,n+1,"angle:epsilon");
  memory->create(Nl,n+1,"angle:Nl");
  memory->create(theta0,n+1,"angle:theta0");
  memory->create(xtarget,n+1,"angle:xtarget");  
  memory->create(ytarget,n+1,"angle:ytarget");
  memory->create(ztarget,n+1,"angle:ztarget");
  memory->create(dist0,n+1,"angle:dist0");  
  memory->create(dist,n+1,"angle:dist");
  memory->create(T_t,n+1,"angle:T_t");
  memory->create(reward,n+1,"angle:reward");
  memory->create(sa,n+1,2,"angle:sa");
  memory->create(first_learn,n+1,"angle:first_learn");
  memory->create(Q0,n+1,"angle:Q0");  
  //memory->create(imin,n+1,"angle:imin"); 
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

void AngleHarmonicThetaQL::coeff(int narg, char **arg)
{  
  if (narg < 14) error->all(FLERR,"Incorrect args for angle coefficients");
  if (!allocated) allocate();


  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nangletypes,ilo,ihi);

  double k_one = force->numeric(FLERR,arg[1]);
  double b_one = force->numeric(FLERR,arg[2]);
  double omega_one = force->numeric(FLERR,arg[3]);
  double theta0max_one = force->numeric(FLERR,arg[4]);
  int Nl_one = force->inumeric(FLERR,arg[5]);
  double skew_one = force->numeric(FLERR,arg[6]);
  if (skew_one>90.0 || skew_one<=-90.0)
    error->all(FLERR,"Incorrect args for angle coefficients: skew angle too large");
  double alpha_one = force->numeric(FLERR,arg[7]);
  if (alpha_one>1.0 || alpha_one<0.0)
    error->all(FLERR,"Incorrect args for angle coefficients: alpha outside [0,1]");
  double gamma_one = force->numeric(FLERR,arg[8]);
  if (gamma_one>1.0 || gamma_one<0.0)
    error->all(FLERR,"Incorrect args for angle coefficients: gamma outside [0,1]");
  double epsilon_one = force->numeric(FLERR,arg[9]);
  if (epsilon_one>1.0 || epsilon_one<0.0)
    error->all(FLERR,"Incorrect args for angle coefficients: epsilon outside [0,1]");
  double xtarget_one = force->numeric(FLERR,arg[10]);
  double ytarget_one = force->numeric(FLERR,arg[11]);
  double ztarget_one = force->numeric(FLERR,arg[12]);  
  double Q0_one = force->numeric(FLERR,arg[13]);

  double tau_one = 0.0;
  if (narg>=15)
    tau_one = force->numeric(FLERR,arg[14]);


  // convert theta0 and b from degrees to radians

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    b[i] = b_one/180.0 * MY_PI;
    omega[i] = omega_one;    
    theta0max[i] = theta0max_one/180.0 * MY_PI;
    theta0[i] = 0.0;
    Nl[i] = Nl_one;
    skew[i] = skew_one/180.0 * MY_PI;
    alpha[i] = alpha_one;
    gamma[i] = gamma_one;
    epsilon[i] = epsilon_one;
    xtarget[i] = xtarget_one;
    ytarget[i] = ytarget_one;    
    ztarget[i] = ztarget_one;    
    tau[i] = tau_one;    
    T_t[i] = fabs(2*MY_PI/omega_one);
    first_learn[i] = true;
    Q0[i] = Q0_one;
    //reward[i] = 0.0;
    //sa[i][0] = 0;
    //sa[i][1] = 0;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleHarmonicThetaQL::equilibrium_angle(int i)
{
  return theta0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void AngleHarmonicThetaQL::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&b[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&omega[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&theta0max[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&theta0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&Nl[1],sizeof(int),atom->nangletypes,fp);
  fwrite(&skew[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&alpha[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&gamma[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&epsilon[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&xtarget[1],sizeof(double),atom->nangletypes,fp);   
  fwrite(&ytarget[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&ztarget[1],sizeof(double),atom->nangletypes,fp);   
  fwrite(&tau[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&reward[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&sa[1][0],sizeof(int),atom->nangletypes*2,fp);  
  fwrite(&dist0[1],sizeof(double),atom->nangletypes,fp);    

}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleHarmonicThetaQL::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->nangletypes,fp);
    fread(&b[1],sizeof(double),atom->nangletypes,fp);
    fread(&omega[1],sizeof(double),atom->nangletypes,fp);
    fread(&theta0max[1],sizeof(double),atom->nangletypes,fp);
    fread(&theta0[1],sizeof(double),atom->nangletypes,fp);
    fread(&Nl[1],sizeof(int),atom->nangletypes,fp);
    fread(&skew[1],sizeof(double),atom->nangletypes,fp);
    fread(&alpha[1],sizeof(double),atom->nangletypes,fp);
    fread(&gamma[1],sizeof(double),atom->nangletypes,fp);    
    fread(&epsilon[1],sizeof(double),atom->nangletypes,fp);    
    fread(&xtarget[1],sizeof(double),atom->nangletypes,fp);
    fread(&ytarget[1],sizeof(double),atom->nangletypes,fp);
    fread(&ztarget[1],sizeof(double),atom->nangletypes,fp);
    fread(&tau[1],sizeof(double),atom->nangletypes,fp);
    fread(&reward[1],sizeof(double),atom->nangletypes,fp);    
    fread(&sa[1][0],sizeof(int),atom->nangletypes*2,fp);    
    fread(&dist0[1],sizeof(double),atom->nangletypes,fp);
  }
  MPI_Bcast(&k[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&b[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&omega[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&theta0max[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&theta0[1],atom->nangletypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&Nl[1],atom->nangletypes,MPI_INT,0,world);
  MPI_Bcast(&skew[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&alpha[1],atom->nangletypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&gamma[1],atom->nangletypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&epsilon[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&xtarget[1],atom->nangletypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&ytarget[1],atom->nangletypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&ztarget[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&tau[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&reward[1],atom->nangletypes,MPI_DOUBLE,0,world);  
  MPI_Bcast(&sa[1][0],atom->nangletypes*2,MPI_INT,0,world);  
  MPI_Bcast(&dist0[1],atom->nangletypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleHarmonicThetaQL::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp,"%d %g %g %g %g %d %g %g %g %g %g %g %g %g\n",i,k[i],b[i]/MY_PI*180.0,omega[i],theta0max[i]/MY_PI*180.0,Nl[i],skew[i]/MY_PI*180.0,alpha[i],gamma[i],epsilon[i],xtarget[i],ytarget[i],ztarget[i],tau[i]);
}

/* ---------------------------------------------------------------------- */

double AngleHarmonicThetaQL::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;

  double delx1 = x[i1][0] - x[i2][0];
  double dely1 = x[i1][1] - x[i2][1];
  double delz1 = x[i1][2] - x[i2][2];
  domain->minimum_image(delx1,dely1,delz1);
  double r1 = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);

  double delx2 = x[i3][0] - x[i2][0];
  double dely2 = x[i3][1] - x[i2][1];
  double delz2 = x[i3][2] - x[i2][2];
  domain->minimum_image(delx2,dely2,delz2);
  double r2 = sqrt(delx2*delx2 + dely2*dely2 + delz2*delz2);

  double c = delx1*delx2 + dely1*dely2 + delz1*delz2;
  c /= r1*r2;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;

  //double dtheta = acos(c) - theta0[type];
  //double tk = k[type] * dtheta;

  double theta = acos(-c);
  if (delx1*dely2-delx2*dely1 > 0)
    theta = -theta;
  double thetaeq = theta0[type] + b[type]*sin(omega[type]*update->ntimestep*update->dt);
  thetaeq = thetaeq - 2.0*MY_PI * round(thetaeq/2.0/MY_PI);
  double dtheta = theta - thetaeq;
  double tk = k[type] * dtheta;
  return tk*dtheta;
}
// void AngleHarmonicThetaQL::update_omega(){
//   int n = atom->nangletypes;
//   double rg;
//   for (int i=1; i<=n; i++){
//     T_t[i]-=update->dt;
//     if (T_t[i]<0){
// #ifdef MTRAND
//       rg = mtrand->gaussian();
// #else
//       rg = random->gaussian();
// #endif
//       omg_t[i] = -exp(rg*omega_sigma[i] + omega[i]);
//       T_t[i] = fabs(2.0*MY_PI/omg_t[i]);
//     }
//   }
//   MPI_Bcast(&omg_t[1],atom->nangletypes,MPI_DOUBLE,0,world);
//   MPI_Bcast(&T_t[1],atom->nangletypes,MPI_DOUBLE,0,world);

// }
double AngleHarmonicThetaQL::shiftmap(double x, double a0, double b0, double a, double b)
{
  if (!(x>=a0 && x<=b0 && fabs(b0-a0)>=SMALL))
    error->all(FLERR,"Passing inappropriate arguments to shiftmap()");
  return (x-a0)/(b0-a0)*(b-a)+a;
}

