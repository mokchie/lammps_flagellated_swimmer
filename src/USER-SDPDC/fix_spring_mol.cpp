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
#include "memory.h"
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
  if (narg < 6) error->all(FLERR,"Illegal fix spring/mol command");

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 4;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  dynamic_group_allow = 0;
  respa_level_support = 1;
  ilevel_respa = 0;

  k_spring = force->numeric(FLERR,arg[3]);
  m_start = force->inumeric(FLERR,arg[4]);
  m_end = force->inumeric(FLERR,arg[5]);

  memory->create(masstot,m_end-m_start+1,"spring/mol:masstot");
  memory->create(xc0,m_end-m_start+1,3,"spring/mol:xc0");
  memory->create(xcm,m_end-m_start+1,3,"spring/mol:xcm");
  memory->create(ftotal,m_end-m_start+1,4,"spring/mol:ftotal");

}

/* ---------------------------------------------------------------------- */

FixSpringMol::~FixSpringMol()
{
  memory->destroy(masstot);
  memory->destroy(xc0);
  memory->destroy(xcm);
  memory->destroy(ftotal);
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
  int j;
  for(int mol=m_start; mol<=m_end; mol++){
    j = mol-m_start;
    masstot[j] = group->mass_mol(igroup,mol);
    group->xcm_mol(igroup,mol,masstot[j],xc0[j]);
  }
  printf("Centers of the molecules are determined!\n");

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

  // fx,fy,fz = components of k * (r-r0) / masstot

  // apply restoring force to atoms in group
  double **x = atom->x;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  tagint *molecule = atom->molecule;
  double dx,dy,dz,r,fx,fy,fz;
  int j;


  for (int mol=m_start; mol<=m_end; mol++){
    j = mol-m_start;
    group->xcm_mol(igroup,mol,masstot[j],xcm[j]);
    dx = xcm[j][0]-xc0[j][0];
    dy = xcm[j][1]-xc0[j][1];
    dz = xcm[j][2]-xc0[j][2];
    r = sqrt(dx*dx+dy*dy+dz*dz);
    r = MAX(r,SMALL);
    fx = k_spring*dx;
    fy = k_spring*dy;
    fz = k_spring*dz;
    ftotal[j][0] = -fx;
    ftotal[j][1] = -fy;
    ftotal[j][2] = -fz;
    ftotal[j][3] = sqrt(fx*fx+fy*fy+fz*fz);

  }
  double massone,mol;
  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if ((mask[i] & groupbit) && molecule[i]<=m_end && molecule[i]>=m_start) {
        massone = rmass[i];
        mol = molecule[i];
        j = mol-m_start;
        f[i][0] += ftotal[j][0]*massone/masstot[j];
        f[i][1] += ftotal[j][1]*massone/masstot[j];
        f[i][2] += ftotal[j][2]*massone/masstot[j];
      }
  } else {
    for (int i = 0; i < nlocal; i++)
      if ((mask[i] & groupbit) && molecule[i]<=m_end && molecule[i]>=m_start) {
        massone = mass[type[i]];
        mol = molecule[i];        
        j = mol-m_start;
        f[i][0] += ftotal[j][0]*massone/masstot[j];
        f[i][1] += ftotal[j][1]*massone/masstot[j];
        f[i][2] += ftotal[j][2]*massone/masstot[j];
      }
  }
  
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
  return ftotal[0][n];
}
