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

#include <cstring>
#include <cstdlib>
#include "fix_addlubforce.h"
#include "atom.h"
#include "atom_masks.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "comm.h"
#include "universe.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"


using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixAddLubForce::FixAddLubForce(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  idregion(NULL)
{
  if (narg < 9) error->all(FLERR,"Illegal fix addforce command");

  dynamic_group_allow = 0;
  scalar_flag = 0;
  vector_flag = 0;
  size_vector = 3;
  global_freq = 1;
  extscalar = 0;
  extvector = 0;
  respa_level_support = 1;
  ilevel_respa = 0;
  virial_flag = 0;
  eta = force->numeric(FLERR,arg[3]);
  aij = force->numeric(FLERR,arg[4]);
  hc = force->numeric(FLERR,arg[5]);
  hregu = force->numeric(FLERR,arg[6]);
  f0_rep = force->numeric(FLERR,arg[7]);
  tstart = force->numeric(FLERR,arg[8]);
  // optional args

  nevery = 1;
  iregion = -1;

  int iarg = 9;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addlubforce command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix addlubforce command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addforce command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix addforce does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix addforce command");
  }
}

/* ---------------------------------------------------------------------- */

FixAddLubForce::~FixAddLubForce()
{
  delete [] idregion;
}

/* ---------------------------------------------------------------------- */

int FixAddLubForce::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAddLubForce::init()
{
  // check variables

  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix addforce does not exist");
  }

  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

}

void FixAddLubForce::init_list(int /*id*/, NeighList *ptr){
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixAddLubForce::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixAddLubForce::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddLubForce::post_force(int vflag)
{
  int *ilist,*jlist,*numneigh,**firstneigh;
  int i,j,k,l,ii,jj,inum,jnum,imol,jmol;
  double rsq,delx,dely,delz,delvx,delvy,delvz,h,r,flx,fly,flz,fl,fr,hc_inv,rx_hat,ry_hat,rz_hat;
  hc_inv = 1.0/hc;
  double a1,b0,b1,b2,c0,tau;
  tau = 1/(0.01*aij);
  a1 = 3.0/2*M_PI*eta*aij*aij;  
  b0 = 3.0/4*M_PI*sqrt(2);
  b1 = 231.0/80*M_PI*sqrt(2);
  b2 = b1/2.0/aij;
  c0 = pow(2.0*aij/hc,1.5)*(b0+hc/2.0/aij*b1);
  double **x = atom->x;
  double **f = atom->f;
  tagint *molecule = atom->molecule;
  double **v = atom->v;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  if (update->ntimestep % nevery) return;
  //if (update->ntimestep*update->dt < tstart) return;

  // energy and virial setup

  if (vflag) v_setup(vflag);
  else evflag = 0;

  if (lmp->kokkos)
    atom->sync_modify(Host, (unsigned int) (F_MASK | MASK_MASK),
                      (unsigned int) F_MASK);

  // update region if necessary

  Region *region = NULL;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  for (ii=0; ii<inum; ii++){
    i = ilist[ii];
    imol = molecule[i];
    if (mask[i] & groupbit){
      if (iregion >= 0 &&
              !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
            continue;
      jlist = firstneigh[i];
      jnum = numneigh[i];
      for (jj=0; jj<jnum; jj++){
        j = jlist[jj];
        j &= NEIGHMASK;
        if (mask[j] & groupbit){
          if (iregion >= 0 &&
              !domain->regions[iregion]->match(x[j][0],x[j][1],x[j][2]))
            continue;
          jmol = molecule[j];
          if (imol!=jmol){
            delx = x[i][0] - x[j][0];
            dely = x[i][1] - x[j][1];
            delz = x[i][2] - x[j][2];
            delvx = v[i][0] - v[j][0];
            delvy = v[i][1] - v[j][1];
            delvz = v[i][2] - v[j][2];
            rsq = delx*delx + dely*dely + delz*delz;
            r = sqrt(rsq);
            rx_hat = delx/r;
            ry_hat = dely/r;
            rz_hat = delz/r;
            h = r-2*aij;
            if (h>=hc) continue;
            if (domain->dimension == 3){
              if(update->ntimestep*update->dt>tstart){
                if (h>hregu)
                  fl = -a1*(rx_hat*delvx + ry_hat*delvy + rz_hat*delvz)*(1.0/h-hc_inv);
                else
                  fl = -a1*(rx_hat*delvx + ry_hat*delvy + rz_hat*delvz)*(1.0/hregu-hc_inv);
              }
              else
                fl = 0;

              fr = f0_rep*tau*exp(-h*tau)/(1-exp(-h*tau));
              //if (h<0) printf("h=%f,fl=%f,fr=%f\n",h,fl,fr);
              //if (fl>fr) printf("step=%d,h=%f,fl=%f,fr=%f,dv=%f,a1=%f\n",update->ntimestep,h,fl,fr,sqrt(delvx*delvx+delvy*delvy+delvz*delvz),a1);
              //printf("imol=%d,jmol=%d,fr=%f\n",imol,jmol,fr);
              f[i][0] += rx_hat * (fl+fr);
              f[i][1] += ry_hat * (fl+fr);
              f[i][2] += rz_hat * (fl+fr);
              f[j][0] -= rx_hat * (fl+fr);
              f[j][1] -= ry_hat * (fl+fr);
              f[j][2] -= rz_hat * (fl+fr);
            } else if (domain->dimension == 2){
              if(update->ntimestep*update->dt>tstart){
                if (h>hregu)              
                  fl = -0.5*eta*(rx_hat*delvx + ry_hat*delvy + rz_hat*delvz)*((pow(2*aij/h,1.5)*(b0+h*b2)-c0));
                else
                  fl = -0.5*eta*(rx_hat*delvx + ry_hat*delvy + rz_hat*delvz)*((pow(2*aij/hregu,1.5)*(b0+hregu*b2)-c0));
              }
              else
                fl = 0;
              fr = f0_rep*tau*exp(-h*tau)/(1-exp(-h*tau));
              //printf("fl = %f\n",fl);
              f[i][0] += rx_hat * (fl+fr);
              f[i][1] += ry_hat * (fl+fr);
              f[j][0] -= rx_hat * (fl+fr);
              f[j][1] -= ry_hat * (fl+fr);
            }

          }
        }
      }
    }

  }
}

/* ---------------------------------------------------------------------- */

void FixAddLubForce::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddLubForce::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

// double FixAddLubForce::compute_scalar()
// {
//   // only sum across procs one time

//   if (force_flag == 0) {
//     MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
//     force_flag = 1;
//   }
//   return foriginal_all[0];
// }

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

// double FixAddLubForce::compute_vector(int n)
// {
//   // only sum across procs one time

//   if (force_flag == 0) {
//     MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
//     force_flag = 1;
//   }
//   return foriginal_all[n+1];
// }

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

// double FixAddLubForce::memory_usage()
// {
//   double bytes = 0.0;
//   if (varflag == ATOM) bytes = maxatom*4 * sizeof(double);
//   return bytes;
// }

/* ----------------------------------------------------------------------
   recalculate adaptive force
------------------------------------------------------------------------- */

/*void FixAddLubForce::adapt_force()
{ 
  int i,j;
  double dm[5],ff[3],vv,qq;
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  FILE *f_read;
  
  niter++;
  if (comm->me == 0){
    ff[1] = 0.0;
    ff[2] = 0.0;
    j = universe->me/96 + 1;
    sprintf(fname,"vel_t%d.%d.plt",j,niter*adapt_every);
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open vel_t file");

    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %lf %lf",&dm[0],&dm[1],&dm[2],&vv);
    qq = vv/20;
    ff[0] = xvalue*q_targ/qq;
    fclose(f_read);
    if (logfile) fprintf(logfile,"x: j = %d, niter = %d, f = %f, %f; q = %f, %f \n",j,niter,xvalue,ff[0],qq,q_targ);
  }
  MPI_Bcast(&ff[0],3,MPI_DOUBLE,0,world);
  xvalue = ff[0];
  yvalue = ff[1];
  zvalue = ff[2];
}*/

/* ---------------------------------------------------------------------- */
/*
void FixAddLubForce::adapt_force()
{ 
  int i,j;
  double dm[5],ff[3],da,vv,qq;
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  FILE *f_read;
  
  niter++;
  if (comm->me == 0){
    ff[2] = 0.0;
    sprintf(fname,"vel_end.%d.plt",niter*adapt_every);
    da = 32.0*11.0/50.0/16.0;
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open vel_end file");

    qq = 0.0;
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    for (i=0; i<50*16; i++){
      fgets(buf,BUFSIZ,f_read);
      sscanf(buf,"%lf %lf %lf %lf",&dm[0],&dm[1],&dm[2],&vv);
      qq += vv*da;
    }
    ff[0] = xvalue*q_targ/qq;
    fclose(f_read);
    if (logfile) fprintf(logfile,"x: niter = %d, f = %f, %f; q = %f, %f \n",niter,xvalue,ff[0],qq,q_targ);

    sprintf(fname,"vel_t.%d.plt",niter*adapt_every);
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open vel_t file");
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %lf %lf %lf",&dm[0],&dm[1],&dm[2],&dm[3],&vv);
    fclose(f_read);
    ff[1] = yvalue - vv; 
    if (logfile) fprintf(logfile,"y: niter = %d, f = %f, %f; vv = %f \n",niter,yvalue,ff[1],vv);
  
  }
  MPI_Bcast(&ff[0],3,MPI_DOUBLE,0,world);
  xvalue = ff[0];
  yvalue = ff[1];
  zvalue = ff[2];
}
*/

/* ---------------------------------------------------------------------- */
