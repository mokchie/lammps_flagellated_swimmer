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



#include "fix_bond_swell_moisture.h"
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
#include "memory.h"
#include "error.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTA 16
#define DR 0.01
#define EPS 1e-4

/* ---------------------------------------------------------------------- */

FixBondSwellMoisture::FixBondSwellMoisture(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if(atom->individual<1) error->all(FLERR,"Individual must be >0 for fix bond/swell/moisture command");
  if (narg < 3) error->all(FLERR,"Illegal fix bond/swell/moisture command");

  btype = force->inumeric(FLERR,arg[3]);
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/swell/moisture command");
}

/* ---------------------------------------------------------------------- */

FixBondSwellMoisture::~FixBondSwellMoisture()
{

}

/* ---------------------------------------------------------------------- */

int FixBondSwellMoisture::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondSwellMoisture::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixBondSwellMoisture::post_integrate()
{
  //if(update->ntimestep==0) return;
  double **x = atom->x;
  int *type = atom->type;
  double **conc = atom->conc;
  int *mask = atom->mask;    
  int nlocal = atom->nlocal;
  int i,j,k,m,dn;
  dn = domain->dimension;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;  
  int **btyp = atom->bond_type;
  double *bondlist_length = neighbor->bondlist_length;
  double **bond_length = atom->bond_length;
  double **bond_length0 = atom->bond_length0;
  if(bond_length0==0) error->all(FLERR,"bond_length0 = NULL in this pair_style");
  double c1,c2,sr;

  for (i=0; i<nlocal; i++){
    if (!(mask[i] & groupbit)) continue;
    c1 = conc[i][0];
    for (m=0; m<num_bond[i]; m++){
      if (btyp[i][m]!=btype) continue;
      j = atom->map(bond_atom[i][m]);
      if(!(mask[j] & groupbit)) continue;
      c2 = conc[j][0];
      if (fabs(c1-c2)<EPS) sr = pow(1-(c1+c2)/2,-1.0/dn);
      else {
        if (dn==1){
          sr = log((1-c2)/(1-c1))/(c1-c2);
        }
        else{
          sr = (pow(1-c2,1.0-1.0/dn)-pow(1-c1,1.0-1.0/dn))/(c1-c2)/(1.0-1.0/dn);
        }
      }
      //if(sr>1.0) printf("sr=%f\n",sr);
      bond_length[i][m] = sr*bond_length0[i][m];
      //b = -moistm;
      //d = -1.0;
      // if (dn==2)
      //   bond_length[i][m] = (-b+sqrt(b*b-4*d))/2*bond_length0[i][m];
      // else if (dn==3){
      //   y1 = b*b*b + (27*d+3*sqrt(81*d*d+12*b*b*b*d))/2;
      //   y2 = b*b*b + (27*d-3*sqrt(81*d*d+12*b*b*b*d))/2;
      //   if (y1<0) cry1 = -pow(-y1,1.0/3.0);
      //   else cry1 = pow(y1,1.0/3.0);
      //   if (y2<0) cry2 = -pow(-y2,1.0/3.0);
      //   else cry2 = pow(y2,1.0/3.0);
      //   bond_length[i][m] = (-b-(cry1+cry2))/3*bond_length0[i][m];
        //printf("%f\n",bond_length[i][m]-bond_length0[i][m]*pow(1-moistm*bond_length0[i][m]/bond_length[i][m],-1.0/dn));

      //}
      //bond_length[i][m] = pow(1-moistm,-1.0/dn)*bond_length0[i][m];

      ////printf("moist=%f\n",moistm);
      ////if(fluid_rho[i]>3.0 || fluid_rho[j]>3.0)
        ////printf("i=%d,j=%d,rhoi=%.2f,rhoj=%.2f\n",i,j,fluid_rho[i],fluid_rho[j]);
      //bond_length[i][m] = bond_length0[i][m]*pow(1.0/(1-moistm),1.0/domain->dimension);
      ////printf("bondlength[%d][%d] = %f, %f\n",i,m,bond_length[i][m],bond_length0[i][m]);
    }
  }
}

/* ---------------------------------------------------------------------- */
void FixBondSwellMoisture::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

