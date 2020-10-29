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

#include <cstring>
#include <cstdlib>
#include <cmath>
#include "fix_active_fluct.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "force.h"
#include "pair.h"
#include "random_mars.h"
#include "comm.h"
#include "universe.h"
#include "group.h"

#define CNT -10.0

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CONSTANT,GAUSS,GAMMA};

/* ---------------------------------------------------------------------- */

FixActiveFluct::FixActiveFluct(LAMMPS *lmp, int narg, char **arg) : 
  Fix(lmp, narg, arg)
{
  int igroup;

  if (narg < 8) error->all(FLERR,"Illegal fix active/fluct command");

  time_style = CONSTANT;
  momentum_ind = 0;
  read_restart_ind = 0;
  restart_global = 1;
  ind_norm = force->inumeric(FLERR,arg[3]);
  f_time = force->numeric(FLERR,arg[4]); 
  f0 = force->numeric(FLERR,arg[5]);
  prob = force->numeric(FLERR,arg[6]);
  norm_every = force->inumeric(FLERR,arg[7]);
  
  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"gauss") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix active/fluct command");
      f_sigma = force->numeric(FLERR,arg[iarg+1]);
      time_style = GAUSS;
      iarg += 2;
    } else if (strcmp(arg[iarg],"gamma") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix active/fluct command");
      beta = force->numeric(FLERR,arg[iarg+1]);
      alpha = f_time;
      d = alpha - 1.0/3.0;
      c = 1.0/sqrt(9.0*d);
      time_style = GAMMA;
      iarg += 2;
    } else if (strcmp(arg[iarg],"momentum") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix active/fluct command");
      igroup = group->find(arg[iarg+1]);
      if (igroup == -1) error->all(FLERR,"Could not find fix group ID for momentum conservation");
      groupbit_sol = group->bitmask[igroup];
      r_mom = force->numeric(FLERR,arg[iarg+2]);
      momentum_ind = 1;
      iarg += 3;
    } else error->all(FLERR,"Illegal fix active/fluct command");
  }
  
  step_start = 200000;
  num_steps = 0;
  write_each = 200000;
  random = NULL;
  norm = NULL;
  dirr = NULL;
  active_sp_dist = NULL;
  dt = update->dt;

  if (time_style != CONSTANT)
    random = new RanMars(lmp,10 + comm->me);
}

/* ---------------------------------------------------------------------- */

FixActiveFluct::~FixActiveFluct()
{
  memory->sfree(dirr);
  memory->sfree(active_sp_dist);
  if (ind_norm)
    memory->destroy(norm);
  if (random) 
    delete random; 
}

/* ---------------------------------------------------------------------- */

int FixActiveFluct::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */


void FixActiveFluct::setup(int vflag)
{
  int i;  
  tagint ind_max1,ind_min1;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  if (read_restart_ind == 0){
    ind_min1 = 1000000000;
    ind_max1 = -1;  
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit){
        if (tag[i] < ind_min1)
          ind_min1 = tag[i];
        if (tag[i] > ind_max1)
          ind_max1 = tag[i]; 
      }   
    MPI_Allreduce(&ind_max1,&ind_max,1,MPI_LMP_TAGINT,MPI_MAX,world); 
    MPI_Allreduce(&ind_min1,&ind_min,1,MPI_LMP_TAGINT,MPI_MIN,world); 
    nn = ind_max - ind_min + 1;

    dirr = (double *) memory->smalloc(4*nn*sizeof(double),"fix_active_fluct:dirr");
    for (i = 0; i < 4*nn; i++)
      dirr[i] = CNT;
  } 
  
  if (ind_norm){
    memory->create(norm,nn,3,"fix_active_fluct:norm"); 
    calc_norm(); 
  }

  active_sp_dist = (double *) memory->smalloc((nn+1)*sizeof(double),"fix_active_fluct:active_sp_dist");
  for (i = 0; i < nn+1; i++)
    active_sp_dist[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void FixActiveFluct::pre_force(int vflag)
{
  int i,j,k,n,ii,jj,inum,jnum;
  double dirh[4*nn], nr[3];
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  double rq,dr,th1,th2;

  int f_ind[nn],f_indh[nn],mom_neigh[nn];
  int *ilist,*jlist,*numneigh,**firstneigh;

  if (momentum_ind){
    inum = force->pair->list->inum;
    ilist = force->pair->list->ilist;
    numneigh = force->pair->list->numneigh;
    firstneigh = force->pair->list->firstneigh;

    for (i = 0; i < nn; i++){
      f_ind[i] = 0;
      f_indh[i] = 0;
      mom_neigh[i] = 0;
    }

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];       
      jlist = firstneigh[i];
      jnum = numneigh[i];
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        if ((mask[i] & groupbit) && (mask[j] & groupbit_sol)){
          dr = sqrt((x[i][0]-x[j][0])*(x[i][0]-x[j][0]) + (x[i][1]-x[j][1])*(x[i][1]-x[j][1]) + (x[i][2]-x[j][2])*(x[i][2]-x[j][2]));
          if (dr < r_mom){
            n = tag[i] - ind_min;
            f_indh[n]++;
	  }
	}
        if ((mask[i] & groupbit_sol) && (mask[j] & groupbit)){
          dr = sqrt((x[i][0]-x[j][0])*(x[i][0]-x[j][0]) + (x[i][1]-x[j][1])*(x[i][1]-x[j][1]) + (x[i][2]-x[j][2])*(x[i][2]-x[j][2]));
          if (dr < r_mom){
            n = tag[j] - ind_min;
            f_indh[n]++;
	  }
	}
      }
    }
    MPI_Allreduce(&f_indh[0],&mom_neigh[0],nn,MPI_INT,MPI_SUM,world);

    for (i = 0; i < nn; i++)
      f_indh[i] = 0; 
  }

  for (i = 0; i < 4*nn; i++)
    dirh[i] = CNT;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      n = tag[i] - ind_min;
      k = 4*n;
      for (j = 0; j < 4; j++) 
        dirh[k+j] = dirr[k+j];
      if (dirh[k] < 1.0){
        dirh[k] = CNT;
        if (random->uniform() < prob){ 
          if (time_style == GAUSS){
            dr = -1.0;       
            while (dr < 0.0)
              dr = (f_time + f_sigma*random->gaussian())/dt;  
            dirh[k] = dr;
          } else if (time_style == GAMMA)
            dirh[k] = generate_gamma()/dt;
          else 
            dirh[k] = f_time/dt;
          if (ind_norm){ 
            dr = 1.0;
            if (ind_norm == 2){
              if (random->uniform() < 0.5)
                dr = -1.0; 
            } else if (ind_norm == 3)
              dr = 2.0*(random->uniform()-0.5);
            for (j = 0; j < 3; j++)
              dirh[k+1+j] = dr*norm[n][j];  
          } else{ 
            th1 = 2.0*M_PI*random->uniform(); 
            if (domain->dimension == 2){
              nr[0] = cos(th1);
              nr[1] = sin(th1);
              nr[2] = 0.0;
	    } else{
              th2 = 2.0*random->uniform() - 1.0;
              rq = sqrt(1.0-th2*th2);
              nr[0] = rq*cos(th1);
              nr[1] = rq*sin(th1);
              nr[2] = th2;
	    }
            rq = sqrt(nr[0]*nr[0] + nr[1]*nr[1] + nr[2]*nr[2]);
            for (j = 0; j < 3; j++)
              dirh[k+1+j] = nr[j]/rq;     
          }

          for (j = 0; j < 3; j++)
            f[i][j] += f0*dirh[k+1+j]; 
          dirh[k] -= 1.0;
          if (momentum_ind) f_indh[n] = 1;
        }
      } else {
        for (j = 0; j < 3; j++)
          f[i][j] += f0*dirh[k+1+j]; 
        dirh[k] -= 1.0; 
        if (momentum_ind) f_indh[n] = 1;
      }
    }

  MPI_Allreduce(&dirh[0],dirr,4*nn,MPI_DOUBLE,MPI_MAX,world);

  if (momentum_ind){
    MPI_Allreduce(&f_indh[0],&f_ind[0],nn,MPI_INT,MPI_SUM,world);

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];       
      jlist = firstneigh[i];
      jnum = numneigh[i];
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        if ((mask[i] & groupbit) && (mask[j] & groupbit_sol)){
          dr = sqrt((x[i][0]-x[j][0])*(x[i][0]-x[j][0]) + (x[i][1]-x[j][1])*(x[i][1]-x[j][1]) + (x[i][2]-x[j][2])*(x[i][2]-x[j][2]));
          n = tag[i] - ind_min;
          if (dr < r_mom && f_ind[n]){
            for (k = 0; k < 3; k++)
              f[j][k] -= f0*dirr[4*n+1+k]/mom_neigh[n];  
	  }
	}
        if ((mask[i] & groupbit_sol) && (mask[j] & groupbit)){
          dr = sqrt((x[i][0]-x[j][0])*(x[i][0]-x[j][0]) + (x[i][1]-x[j][1])*(x[i][1]-x[j][1]) + (x[i][2]-x[j][2])*(x[i][2]-x[j][2]));
          n = tag[j] - ind_min;
          if (dr < r_mom && f_ind[n]){
            for (k = 0; k < 3; k++)
              f[i][k] -= f0*dirr[4*n+1+k]/mom_neigh[n];  
	  }
	}
      }
    }
  }  
 
  if (ind_norm) 
    if (update->ntimestep > 0 && update->ntimestep%norm_every == 0)
      calc_norm();

  if (comm->me == 0){
    if (update->ntimestep > step_start){
      num_steps++;
      j = 0;  
      for (k = 0; k < nn; k++)
        if (dirr[4*k] != CNT) j++;
      active_sp_dist[j] += 1.0;
        

      if ((update->ntimestep-step_start)%write_each == 0){
        FILE* out_stat;
        char f_name[FILENAME_MAX];

        if (num_steps){
          sprintf(f_name,"active_sp_dist%d.dat",universe->iworld);
          out_stat=fopen(f_name,"w");
          for (k = 0; k < nn+1; k++)
            fprintf(out_stat,"%d %15.10lf \n",k,active_sp_dist[k]/num_steps);
          fclose(out_stat);
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixActiveFluct::calc_norm()
{
  int i,j,n,i1,i2,i3;
  double d21x,d21y,d21z,d31x,d31y,d31z;
  double nx,ny,nz,nh;
  double nr[nn*3],nr_tmp[nn*3];
  double **x = atom->x;
  tagint *tag = atom->tag;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
 
  for (i = 0; i < nn*3; i++){ 
    nr[i] = 0.0;
    nr_tmp[i] = 0.0;
  }

  if (domain->dimension == 2){
    for (n = 0; n < nbondlist; n++) {
      i1 = bondlist[n][0];
      i2 = bondlist[n][1];
      ny = x[i1][0] - x[i2][0];
      nx = x[i2][1] - x[i1][1];  
      nh = sqrt(nx*nx + ny*ny);  
      if (tag[i1] >= ind_min && tag[i1] <= ind_max){
        j = tag[i1] - ind_min;
        nr_tmp[j*3] += nx/nh;
        nr_tmp[j*3 + 1] += ny/nh;
      } 
      if (tag[i2] >= ind_min && tag[i2] <= ind_max){
        j = tag[i2] - ind_min;
        nr_tmp[j*3] += nx/nh;
        nr_tmp[j*3 + 1] += ny/nh;
      }  
    }   

  } else{
    for (n = 0; n < nanglelist; n++) {

      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];

      // 2-1 distance
      d21x = x[i2][0] - x[i1][0];
      d21y = x[i2][1] - x[i1][1];
      d21z = x[i2][2] - x[i1][2];

      // 3-1 distance
      d31x = x[i3][0] - x[i1][0];
      d31y = x[i3][1] - x[i1][1];
      d31z = x[i3][2] - x[i1][2];

      // calculate normal
      nx = d21y*d31z - d31y*d21z;
      ny = d31x*d21z - d21x*d31z;
      nz = d21x*d31y - d31x*d21y;
      nh = sqrt(nx*nx + ny*ny + nz*nz);

      if (tag[i1] >= ind_min && tag[i1] <= ind_max){
        j = tag[i1] - ind_min;
        nr_tmp[j*3] += nx/nh;
        nr_tmp[j*3 + 1] += ny/nh;
        nr_tmp[j*3 + 2] += nz/nh;
      } 
      if (tag[i2] >= ind_min && tag[i2] <= ind_max){
        j = tag[i2] - ind_min;
        nr_tmp[j*3] += nx/nh;
        nr_tmp[j*3 + 1] += ny/nh;
        nr_tmp[j*3 + 2] += nz/nh;
      }
      if (tag[i3] >= ind_min && tag[i3] <= ind_max){
        j = tag[i3] - ind_min;
        nr_tmp[j*3] += nx/nh;
        nr_tmp[j*3 + 1] += ny/nh;
        nr_tmp[j*3 + 2] += nz/nh;
      }
    }
  }

  MPI_Allreduce(&nr_tmp[0],&nr[0],nn*3,MPI_DOUBLE,MPI_SUM,world);
  for (i = 0; i < nn; i++){
    for (j = 0; j < 3; j++)
        norm[i][j] = 0.0;
    nh = sqrt(nr[i*3]*nr[i*3] + nr[i*3 + 1]*nr[i*3 + 1] + nr[i*3 + 2]*nr[i*3 + 2]);
    if (nh > 0.0)
      for (j = 0; j < 3; j++)
        norm[i][j] = nr[i*3 + j]/nh;
  }

  /*char fname[FILENAME_MAX];
  FILE *f_write;
  if (comm->me == 0) {
    sprintf(fname,"normals.dat");
    f_write = fopen(fname,"w");
    for (i=0; i<nn; i++)
      fprintf(f_write,"%lf %lf %lf \n",norm[i][0], norm[i][1], norm[i][2]);
    fclose(f_write);
  }*/  
}

/* ---------------------------------------------------------------------- */

double FixActiveFluct::generate_gamma()
{
  double z,v,vv,u;
  int flag = 1;
  
  while (flag){ 
    z = random->gaussian();
    v = 1.0 + c*z; 
    if (v > 0.0){
      vv = v*v*v;
      u = random->uniform();
      if (log(u) < 0.5*z*z + d - d*vv + d*log(vv)) flag = 0;
    }
  } 

  return d*vv/beta;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixActiveFluct::write_restart(FILE *fp)
{
  int i;
  int n = 0;
  int nsize = 4*nn + 3;
  double *list;
  
  memory->create(list,nsize,"fix_active_fluct:list");
  list[n++] = static_cast<double> (ind_min);
  list[n++] = static_cast<double> (ind_max);
  list[n++] = static_cast<double> (nn);
  for (i = 0; i < 4*nn; i++)
    list[n++] = dirr[i];

  if (comm->me == 0) {
    int size = nsize * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),nsize,fp);
  }

  memory->destroy(list);
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixActiveFluct::restart(char *buf)
{
  int i;
  int n = 0;
  double *list = (double *) buf;

  ind_min = static_cast<tagint> (list[n++]);
  ind_max = static_cast<tagint> (list[n++]);
  nn = static_cast<int> (list[n++]);

  dirr = (double *) memory->smalloc(4*nn*sizeof(double),"fix_active_fluct:dirr");
  for (i = 0; i < 4*nn; i++)
    dirr[i] = list[n++];

  read_restart_ind = 1;
}

/* ---------------------------------------------------------------------- */


