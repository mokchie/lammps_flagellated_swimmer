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
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "math_const.h"
#include "fix_spring_mol.h"
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "respa.h"
#include "domain.h"
#include "force.h"
#include "group.h"
#include "error.h"
using namespace LAMMPS_NS;
using namespace FixConst;
#ifndef MY_PI
#define MY_PI 3.141592653589793
#endif
#define SMALL 1.0e-10


/* ---------------------------------------------------------------------- */

FixSpringMol::FixSpringMol(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix spring/mol command");

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 4;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  dynamic_group_allow = 1;
  respa_level_support = 1;
  ilevel_respa = 0;
  ind_calc_ori = 0; 

  k_spring = force->numeric(FLERR,arg[3]);
  m_start = force->inumeric(FLERR,arg[4]);
  m_end = force->inumertic(FLERR,arg[5]);
  ind_calc_ori = force->inumeric(FLERR,arg[6]);

  ftotal[0] = ftotal[1] = ftotal[2] = ftotal[3] = 0.0;
}

/* ---------------------------------------------------------------------- */

FixSpringMol::~FixSpringMol()
{
  
}

/* ---------------------------------------------------------------------- */

int FixSpringMol::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSpringMol::init()
{
  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
  if(ind_calc_ori){
    double slope,intercept;
    group->lr(igroup,plane,&slope,&intercept);
    phi = atan(slope);
    if (comm->me == 0)
      if (logfile) {
        fprintf(logfile,"fix spring/orientation angle: %f \n",phi);
        fflush(logfile);
      }
  }

}

/* ---------------------------------------------------------------------- */

void FixSpringMol::setup(int vflag)
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

void FixSpringMol::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixSpringMol::post_force(int /*vflag*/)
{
  int n1,n2,n3;
  if (plane==XY) {n1 = 0; n2 = 1; n3 = 2;}
  else if (plane==XZ) {n1 = 0; n2 = 2; n3 = 1;}
  else if (plane==YZ) {n1 = 1; n2 = 2; n3 = 0;}
  else error->all(FLERR,"has to be one of the 'xy','xz','yz' plane");
  double slope,intercept,phit,dphi,M,ra,cc[3],fone,gc;
  double ftotal_one[3];
  ftotal_one[0] = ftotal_one[1] = ftotal_one[2] = 0.0;
  group->lr(igroup,plane,&slope,&intercept);
  group->r2cm(igroup,plane,&ra,cc);
  gc = group->count(igroup);
  phit = atan(slope);
  dphi = phit-phi;
  M = -dphi*k_spring;
  espring = 0.5*k_spring*dphi*dphi;
  //if (plane==XZ) M=-M;
  fone = abs(M/ra/gc);


  // fx,fy,fz = components of k * (r-r0) / masstotal

  // apply restoring force to atoms in group
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double dx,dy;
  double psi;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      dx = x[i][n1]-cc[n1];
      dy = x[i][n2]-cc[n2];
      psi = atan2(dy,dx);
      if (M<0) psi -= MY_PI/2;
      else psi += MY_PI/2;
      f[i][n1] += fone*cos(psi);
      f[i][n2] += fone*sin(psi);
      ftotal_one[n1] += fone*cos(psi);
      ftotal_one[n2] += fone*sin(psi);
      }
  MPI_Allreduce(ftotal_one,ftotal,3,MPI_DOUBLE,MPI_SUM,world);
  ftotal[3] = sqrt(ftotal[0]*ftotal[0] + ftotal[1]*ftotal[1] + ftotal[2]*ftotal[2]);
  
}

/* ---------------------------------------------------------------------- */

void FixSpringMol::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixSpringMol::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of stretched spring
------------------------------------------------------------------------- */

double FixSpringMol::compute_scalar()
{
  return espring;
}

/* ----------------------------------------------------------------------
   return components of total spring force on fix group
------------------------------------------------------------------------- */

double FixSpringMol::compute_vector(int n)
{
  return ftotal[n];
}
