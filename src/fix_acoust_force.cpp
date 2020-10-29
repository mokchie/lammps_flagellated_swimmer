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
#include <cmath>
#include "fix_acoust_force.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "math_const.h"
#include "comm.h"
#include "universe.h"
#include "neighbor.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

#define  MOLINC 20

/* ---------------------------------------------------------------------- */

FixAcoustForce::FixAcoustForce(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  coeff(NULL), lambda(NULL), dir(NULL), dath(NULL), datt(NULL)
{  
  if (narg < 6 || (narg-3)%3 != 0) error->all(FLERR,"Illegal fix acoust/force command");

  nw = round((narg-3)/3.0);
  memory->create(coeff,nw,"fix_acoust_force:coeff");
  memory->create(lambda,nw,"fix_acoust_force:lambda");
  memory->create(dir,nw,2,"fix_acoust_force:dir");

  double theta,lm;
  int iarg = 3;
  for (int i = 0; i < nw; i++){
    coeff[i] = force->numeric(FLERR,arg[iarg]);
    lm = force->numeric(FLERR,arg[iarg+1]); 
    theta = force->numeric(FLERR,arg[iarg+2]);  
    coeff[i] *= MY_PI/lm;
    lambda[i] = 4.0*MY_PI/lm;
    theta *= MY_PI/180.0;
    dir[i][0] = cos(theta);
    dir[i][1] = sin(theta);
    iarg += 3;
  }

  n_mol_limit = MOLINC;
  memory->create(dath,3*n_mol_limit,"fix_acoust_force:dath");
  memory->create(datt,3*n_mol_limit,"fix_acoust_force:datt");  
}

/* ---------------------------------------------------------------------- */

FixAcoustForce::~FixAcoustForce()
{
  memory->destroy(coeff);
  memory->destroy(lambda);
  memory->destroy(dir);
  memory->destroy(dath);
  memory->destroy(datt);
}

/* ---------------------------------------------------------------------- */

int FixAcoustForce::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAcoustForce::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAcoustForce::post_force(int vflag)
{
  int n,i,k,i1,i2,i3;
  tagint m;
  double xx,vv,ff;
  double xx1[3],xx2[3],xx3[3];
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  tagint *n_atoms = atom->mol_size;
  int nlocal = atom->nlocal;
  
  if (update->ntimestep % nevery) return;

  n_mol_max = atom->n_mol_max;
  if (n_mol_max > n_mol_limit){
    n_mol_limit = n_mol_max + MOLINC;
    memory->grow(dath,3*n_mol_limit,"fix_acoust_force:dath");
    memory->grow(datt,3*n_mol_limit,"fix_acoust_force:datt");
  }

  for (n = 0; n < 3*n_mol_max; n++){
    dath[n] = 0.0;
    datt[n] = 0.0;
  }  

  // calculate volume (in 3D) or area (in 2D) 
  
  if (domain->dimension == 3){
  
    int **anglelist = neighbor->anglelist;
    int nanglelist = neighbor->nanglelist;
    double d21x,d21y,d21z,d31x,d31y,d31z;
    double nx,ny,nz,mx,my,mz;

    for (n = 0; n < nanglelist; n++) {

      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];
      m = molecule[i1]-1;

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

      // calculate center
      domain->unmap(x[i1],atom->image[i1],xx1);
      domain->unmap(x[i2],atom->image[i2],xx2);
      domain->unmap(x[i3],atom->image[i3],xx3);

      mx =  xx1[0] + xx2[0] + xx3[0];
      my =  xx1[1] + xx2[1] + xx3[1];
      mz =  xx1[2] + xx2[2] + xx3[2];

      // calculate volume
      vv = (nx*mx + ny*my + nz*mz)/18.0;
      dath[3*m] += vv;
    }
  } else{

    int **bondlist = neighbor->bondlist;
    int nbondlist = neighbor->nbondlist;

    for (n = 0; n < nbondlist; n++) {

      i1 = bondlist[n][0];
      i2 = bondlist[n][1];
      m = molecule[i1]-1;

      domain->unmap(x[i1],atom->image[i1],xx1);
      domain->unmap(x[i2],atom->image[i2],xx2);

      vv = 0.5*(xx1[0]*xx2[1] - xx2[0]*xx1[1]);
      dath[3*m] += vv;
    }
  }

  for (i = 0; i < nlocal; i++){
    m = molecule[i];
    if (m){
      m--;
      domain->unmap(x[i],atom->image[i],xx1);
      dath[3*m+1] += xx1[0];
      dath[3*m+2] += xx1[1];
    }
  }

  for (i = 0; i < n_mol_max; i++){
    k = atom->mol_type[i];
    if (k > -1 && k < atom->n_mol_types){
      dath[3*i+1] /= n_atoms[k];
      dath[3*i+2] /= n_atoms[k];
    }
  }    
  
  MPI_Allreduce(dath,datt,3*n_mol_max,MPI_DOUBLE,MPI_SUM,world);
 
  // apply force

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      m = molecule[i];
      if (m){
        m--;
	k = atom->mol_type[m];
	if (k > -1 && k < atom->n_mol_types){
	  for (n = 0; n < nw; n++) { 
	    xx = datt[3*m+1]*dir[n][0] + datt[3*m+2]*dir[n][1];
	    ff = coeff[n]*datt[3*m]*sin(lambda[n]*xx)/n_atoms[k];
            //if (atom->tag[i] == 8150)
            //  printf("step: %d; me: %d; xx,ff: %g %g; center: %g %g %g; x: %g %g \n",update->ntimestep,comm->me,xx,ff,datt[3*m+1],datt[3*m+2],datt[3*m],x[i][0],x[i][1]); 
	    f[i][0] += ff*dir[n][0];
            f[i][1] += ff*dir[n][1];
	  }
        }
      }
    }
}

/* ---------------------------------------------------------------------- */

