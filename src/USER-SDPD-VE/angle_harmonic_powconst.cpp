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
#include <cstring>
#include "angle_harmonic_powconst.h"
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

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001
#define EPS 1e-6


/* ---------------------------------------------------------------------- */

AngleHarmonicPowconst::AngleHarmonicPowconst(LAMMPS *lmp) : Angle(lmp)
{
  k = NULL;
  b = NULL;
  omega = NULL;
  theta0 = NULL;
  skew = NULL;
  pw = NULL;
  start = NULL;
  icompute = NULL;
  pr = NULL;
  fraction = NULL;
  tau = NULL;
}

/* ---------------------------------------------------------------------- */

AngleHarmonicPowconst::~AngleHarmonicPowconst()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(b);
    memory->destroy(omega);
    memory->destroy(theta0);
    memory->destroy(skew);
    memory->destroy(pw);
    memory->destroy(start);
    memory->destroy(icompute);
    memory->destroy(pr);
    memory->destroy(fraction);
    memory->destroy(tau);
  }
}

/* ---------------------------------------------------------------------- */

void AngleHarmonicPowconst::compute(int eflag, int vflag)
{
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
  update_omega();
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

void AngleHarmonicPowconst::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;

  memory->create(k,n+1,"angle:k");
  memory->create(b,n+1,"angle:b");
  memory->create(omega,n+1,"angle:omega");
  memory->create(theta0,n+1,"angle:theta0");
  memory->create(skew,n+1,"angle:skew");
  memory->create(pw,n+1,"angle:pw");
  memory->create(start,n+1,"angle:start");
  memory->create(icompute,n+1,"angle:icompute");
  memory->create(setflag,n+1,"angle:setflag");
  memory->create(pr,n+1,"angle:pr");
  memory->create(fraction,n+1,"angle:fraction");  
  memory->create(tau,n+1,"angle:tau");  

  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void AngleHarmonicPowconst::coeff(int narg, char **arg)
{
  if (narg < 11) error->all(FLERR,"Incorrect args for angle coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nangletypes,ilo,ihi);

  double k_one = force->numeric(FLERR,arg[1]);
  double b_one = force->numeric(FLERR,arg[2]);
  double omega_one = force->numeric(FLERR,arg[3]);
  double theta0_one = force->numeric(FLERR,arg[4]);
  double skew_one = force->numeric(FLERR,arg[5]);
  if (skew_one>90.0 || skew_one<=-90.0)
    error->all(FLERR,"Incorrect args for angle coefficients: skew angle too large");
  double pw_one = force->numeric(FLERR,arg[6]);
  int start_one = force->inumeric(FLERR,arg[7]);
  if (arg[8][0] != 'c')
    error->all(FLERR,"Incorrect args for angle coefficients: compute ID required");
  char *suffix = new char[strlen(arg[8])];
  strcpy(suffix ,&arg[8][2]);
  int ic = modify->find_compute(suffix);
  if (ic < 0)
    error->all(FLERR,"Compute ID for angle coeff does not exist");
  double pr_one = force->numeric(FLERR,arg[9]);
  double fraction_one = force->numeric(FLERR,arg[10]);
  double tau_one = 0.0;
  if (narg>=12)
    tau_one = force->numeric(FLERR,arg[11]);


  // convert theta0 and b from degrees to radians

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    theta0[i] = theta0_one/180.0 * MY_PI;
    b[i] = b_one/180.0 * MY_PI;
    omega[i] = omega_one;
    skew[i] = skew_one/180.0 * MY_PI;
    pw[i] = pw_one;
    start[i] = start_one;
    icompute[i] = ic;
    pr[i] = pr_one;
    fraction[i] = fraction_one;
    tau[i] = tau_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleHarmonicPowconst::equilibrium_angle(int i)
{
  return theta0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void AngleHarmonicPowconst::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&b[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&omega[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&theta0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&skew[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&pw[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&start[1],sizeof(int),atom->nangletypes,fp);
  fwrite(&icompute[1],sizeof(int),atom->nangletypes,fp);
  fwrite(&pr[1],sizeof(double),atom->nangletypes,fp);  
  fwrite(&fraction[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&tau[1],sizeof(double),atom->nangletypes,fp);

}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleHarmonicPowconst::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->nangletypes,fp);
    fread(&b[1],sizeof(double),atom->nangletypes,fp);
    fread(&omega[1],sizeof(double),atom->nangletypes,fp);
    fread(&theta0[1],sizeof(double),atom->nangletypes,fp);
    fread(&skew[1],sizeof(double),atom->nangletypes,fp);
    fread(&pw[1],sizeof(double),atom->nangletypes,fp);
    fread(&start[1],sizeof(int),atom->nangletypes,fp);
    fread(&icompute[1],sizeof(int),atom->nangletypes,fp);
    fread(&pr[1],sizeof(double),atom->nangletypes,fp);    
    fread(&fraction[1],sizeof(double),atom->nangletypes,fp);    
    fread(&tau[1],sizeof(double),atom->nangletypes,fp);    

  }
  MPI_Bcast(&k[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&b[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&omega[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&theta0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&skew[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&pw[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&start[1],atom->nangletypes,MPI_INT,0,world);  
  MPI_Bcast(&icompute[1],atom->nangletypes,MPI_INT,0,world);
  MPI_Bcast(&pr[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&fraction[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&tau[1],atom->nangletypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleHarmonicPowconst::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g %d %d %g %g %g\n",i,k[i],b[i]/MY_PI*180.0,omega[i],theta0[i]/MY_PI*180.0,skew[i]/MY_PI*180.0,pw[i],start[i],icompute[i],pr[i],fraction[i],tau[i]);
}

/* ---------------------------------------------------------------------- */

double AngleHarmonicPowconst::single(int type, int i1, int i2, int i3)
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
void AngleHarmonicPowconst::update_omega(){
  int n = atom->nangletypes;
  double fv,target_omega;
  float sign;
  for (int i=1; i<=n; i++){
    if (update->ntimestep>=start[i]){
      fv = -modify->compute[icompute[i]]->compute_scalar();
      //if ((fv>0 && pw[i]>0) || (fv<0 && pw[i]<0))
        //target_omega = pow(pw[i]/fv,1.0/pr[i])*omega[i];
        if (omega[i]>0) sign=1.0; else sign=-1.0;
        if (fv>pw[i]){
          if(pr[i]>=1.0)
            omega[i] -= omega[i]*fraction[i];
          else
            omega[i] -= sign*fraction[i];
        }
        else{
          if(pr[i]>=1.0)
            omega[i] += omega[i]*fraction[i];
          else
            omega[i] += sign*fraction[i];
        }
    }
  }
}
double AngleHarmonicPowconst::shiftmap(double x, double a0, double b0, double a, double b)
{
  if (!(x>=a0 && x<=b0 && fabs(b0-a0)>=SMALL))
    error->all(FLERR,"Passing inappropriate arguments to shiftmap()");
  return (x-a0)/(b0-a0)*(b-a)+a;
}
