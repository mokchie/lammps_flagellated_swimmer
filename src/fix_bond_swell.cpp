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



#include "fix_bond_swell.h"
#include <cmath>
#include <mpi.h>
#include <cstring>
#include <cstdlib>

#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "domain.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTA 16
#define DR 0.01

/* ---------------------------------------------------------------------- */

FixBondSwell::FixBondSwell(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  weight1 = weight2 = NULL;
  set_weight_ind = 1;
  dr_inv = 1.0/DR;
  int grph;
  if(atom->individual<1) error->all(FLERR,"Individual must be >0 for fix bond/swell command");
  if (narg < 9) error->all(FLERR,"Illegal fix bond/swell command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  btype = force->inumeric(FLERR,arg[3]);
  ks = force->numeric(FLERR,arg[4]);
  rhoc = force->numeric(FLERR,arg[5]);

  grph = group->find(arg[6]);
  if (grph == -1) error->all(FLERR,"Could not find group_rho ID in fix bond/swell");
  groupbit_rho = group->bitmask[grph];
  rmax_factor = force->numeric(FLERR,arg[7]);
  cutoff = force->numeric(FLERR,arg[8]);

  if (narg>=10)
    t_start = force->numeric(FLERR,arg[9]);
  else t_start = 0;

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/swell command");
  if (ks <= 0) error->all(FLERR,"Illegal fix bond/swell command");  
}

/* ---------------------------------------------------------------------- */

FixBondSwell::~FixBondSwell()
{
  memory->destroy(weight1);
  memory->destroy(weight2);
}

/* ---------------------------------------------------------------------- */

int FixBondSwell::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondSwell::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void FixBondSwell::post_integrate()
{
  if (set_weight_ind){
    set_weight();
    set_weight_ind = 0;
  }
  //if(update->ntimestep==0) return;
  double **x = atom->x;
  double *fluid_rho = atom->fluid_rho;
  int *type = atom->type;
  int *mask = atom->mask;    
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int i,j,k,ii,jj;
  int inum, jnum, itype, jtype;
  int *ilist, *jlist, *numneigh, **firstneigh;  
  double xtmp,ytmp,ztmp,rsq,wd,r,cutsq,delx,dely,delz;
  cutsq = cutoff*cutoff;
 // printf("csum = %f\n",csumall);
  for (i = 0; i < nall; i++) {
    fluid_rho[i] = 0.0;
  }
  neighbor->build_one(list,0);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms to calculate fluid_rho

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if(mask[i] & groupbit_rho || mask[i] & groupbit){
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];
      for(jj = 0; jj < jnum; jj++){
        j = jlist[jj];
        j &= NEIGHMASK;
        if ((mask[i] & groupbit_rho && mask[j] & groupbit) || (mask[i] & groupbit && mask[j] & groupbit_rho)){
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;
          jtype = type[j];

          if (rsq < cutsq) {
            r = sqrt(rsq);
            k = static_cast<int> (r*dr_inv);
            wd = (weight1[itype][jtype][k+1]-weight1[itype][jtype][k])*(r*dr_inv - k) + weight1[itype][jtype][k];
            fluid_rho[i] += wd;
            if (force->newton_pair || j < nlocal) {
              if (mask[j] & groupbit)
                fluid_rho[j] += wd;   
            }
          }
        }
      }
    }
  }

  comm->reverse_comm_fix(this,1);
  comm->forward_comm_fix(this,1);

  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;  
  double *bondlist_length = neighbor->bondlist_length;
  double **bond_length = atom->bond_length;
  double **bond_length0 = atom->bond_length0;
  if(bond_length0==0) error->all(FLERR,"bond_length0 = NULL in this pair_style");
  int typ,m;
  double rhom;

  for (i=0; i<nlocal; i++){
    if (!(mask[i] & groupbit)) continue;
    for (m=0; m<num_bond[i]; m++){
      j = atom->map(bond_atom[i][m]);
      if(!(mask[j] & groupbit)) continue;
      rhom = (fluid_rho[i]+fluid_rho[j])/2;
      //if(fluid_rho[i]>3.0 || fluid_rho[j]>3.0)
        //printf("i=%d,j=%d,rhoi=%.2f,rhoj=%.2f\n",i,j,fluid_rho[i],fluid_rho[j]);
      if (bond_length[i][m] < bond_length0[i][m]*rmax_factor && update->ntimestep*update->dt>t_start && rhom>rhoc){
        bond_length[i][m] += ks*pow(rhom-rhoc,1.0/domain->dimension)*bond_length0[i][m]*update->dt;
      }
    }
  }
   

}

/* ---------------------------------------------------------------------- */
void FixBondSwell::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

void FixBondSwell::set_weight()
{
  //using the quintic spline kernel
  int i,j,k,l;
  int n = atom->ntypes;
  double rr,omega0,h;
  int nw_max;
 
  if (weight1) { 
    memory->destroy(weight1);
    weight1 = NULL;
  }
  if (weight2) { 
    memory->destroy(weight2);
    weight2 = NULL;
  }

  if (domain->dimension == 3) {
      omega0 = 3.0 / (359.0 * M_PI);
    } else{
      omega0 = 7.0 / (478.0 * M_PI);
  }


  rr = -1.0;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++) 
      rr = MAX(rr,cutoff);


  nw_max = static_cast<int> (rr*dr_inv) + 2;
  if (nw_max < 1 || nw_max > 600) error->all(FLERR,"Non-positive or too large value for nw_max to initialize weight arrays: PairSDPDVENoEdyn2D::set_weight.");
  memory->create(weight1,n+1,n+1,nw_max,"pair:weight1");
  memory->create(weight2,n+1,n+1,nw_max,"pair:weight2");

  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++){
      k = static_cast<int> (cutoff*dr_inv) + 1;
      if (k > nw_max) error->all(FLERR,"Error in PairSDPDVENoEdyn2D::set_weight - k > nw_max");

      if (cutoff > 0.0){
        h = cutoff/3.0;
        for (l = 0; l < nw_max; l++){
          rr = l*DR/h;
          if (rr > 3.0){
            weight1[i][j][l] = 0.0;
            weight2[i][j][l] = 0.0;
          } else{
            if (domain->dimension == 3) {
              if(rr<=1.0){
                weight1[i][j][l] = omega0/h/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 6.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr) + 15.0*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 30.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr) + 75.0*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr))/rr;
              } else if(rr<=2.0){
                weight1[i][j][l] = omega0/h/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 6.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 30.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr))/rr;
              } else {
                weight1[i][j][l] = omega0/h/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr))/rr;
              }
            } else{
              if(rr<=1.0){
                weight1[i][j][l] = omega0/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 6.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr) + 15.0*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 30.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr) + 75.0*(1.0-rr)*(1.0-rr)*(1.0-rr)*(1.0-rr))/rr;
              } else if(rr<=2.0){
                weight1[i][j][l] = omega0/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 6.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr) - 30.0*(2.0-rr)*(2.0-rr)*(2.0-rr)*(2.0-rr))/rr;
              } else {
                weight1[i][j][l] = omega0/h/h*((3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr));
                if(l>0)
                  weight2[i][j][l] = omega0/h/h/h/h*(5.0*(3.0-rr)*(3.0-rr)*(3.0-rr)*(3.0-rr))/rr;
              }

            }
          }
          weight1[j][i][l] = weight1[i][j][l];
          weight2[j][i][l] = weight2[i][j][l];
        } 
        weight2[i][j][0] = weight2[i][j][1];
        weight2[j][i][0] = weight2[j][i][1];
      } else{
        for (l = 0; l < nw_max; l++){ 
          weight1[i][j][l] = 0.0;
          weight2[i][j][l] = 0.0;
          weight1[j][i][l] = 0.0;
          weight2[j][i][l] = 0.0;
        }
      }
    }

}

/* ---------------------------------------------------------------------- */

int FixBondSwell::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;
  double *fluid_rho = atom->fluid_rho;

  m = 0;
  for (i = 0; i < n; i ++) {
    j = list[i];
    buf[m++] = fluid_rho[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondSwell::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  double *fluid_rho = atom->fluid_rho;  

  m = 0;
  last = first + n ;
  for (i = first; i < last; i++){
    fluid_rho[i] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int FixBondSwell::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  double *fluid_rho = atom->fluid_rho;  

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = fluid_rho[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondSwell::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  double *fluid_rho = atom->fluid_rho;  

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    fluid_rho[j] += buf[m++];
  }
}
/* ---------------------------------------------------------------------- */
void FixBondSwell::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}
