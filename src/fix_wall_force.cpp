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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "fix_wall_force.h"
#include "atom.h"
#include "input.h"
#include "variable.h"
#include "domain.h"
#include "lattice.h"
#include "update.h"
#include "modify.h"
#include "respa.h"
#include "error.h"
#include "force.h"
#include <iostream>

#include "comm.h"


using namespace LAMMPS_NS;
using namespace FixConst;

enum{XLO=0,XHI=1,YLO=2,YHI=3,ZLO=4,ZHI=5};
enum{NONE=0,EDGE,CONSTANT,VARIABLE,FORCE};

/* ---------------------------------------------------------------------- */

FixWallForce::FixWallForce(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  nwall(0)
{
  scalar_flag = 1;
  vector_flag = 1;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  respa_level_support = 1;
  ilevel_respa = 0;
  total_force = 0;

  // parse args

  int scaleflag = 1;
  fldflag = 0;
  int pbcflag = 0;

  for (int i = 0; i < 6; i++) xstr[i] = estr[i] = sstr[i] = NULL;

  int iarg = 3;
  while (iarg < narg) {
    if ((strcmp(arg[iarg], "xlo") == 0) || (strcmp(arg[iarg], "xhi") == 0) ||
        (strcmp(arg[iarg], "ylo") == 0) || (strcmp(arg[iarg], "yhi") == 0) ||
        (strcmp(arg[iarg], "zlo") == 0) || (strcmp(arg[iarg], "zhi") == 0)) {
      if (iarg + 5 > narg) error->all(FLERR, "Illegal fix wall command");

      int newwall;
      if (strcmp(arg[iarg], "xlo") == 0) newwall = XLO;
      else if (strcmp(arg[iarg], "xhi") == 0) newwall = XHI;
      else if (strcmp(arg[iarg], "ylo") == 0) newwall = YLO;
      else if (strcmp(arg[iarg], "yhi") == 0) newwall = YHI;
      else if (strcmp(arg[iarg], "zlo") == 0) newwall = ZLO;
      else if (strcmp(arg[iarg], "zhi") == 0) newwall = ZHI;

      for (int m = 0; (m < nwall) && (m < 6); m++)
        if (newwall == wallwhich[m])
          error->all(FLERR, "Wall defined twice in fix wall command");

      wallwhich[nwall] = newwall;

      xstyle[nwall] = FORCE;
      int dim = wallwhich[nwall] /2;
      int side = wallwhich[nwall] % 2;
      if (side == 0) coord0[nwall] = domain->boxlo[dim];
      else coord0[nwall] = domain->boxhi[dim];
      force_push[nwall] = force->numeric(FLERR, arg[iarg+1]);
      friction[nwall] = force->numeric(FLERR, arg[iarg+2]);
      epsilon[nwall] = force->numeric(FLERR, arg[iarg + 3]);
      sigma[nwall] = force->numeric(FLERR, arg[iarg + 4]);
      cutoff[nwall] = force->numeric(FLERR, arg[iarg + 5]);
      nwall++;

      iarg += 6;

    }
  }

  size_vector = 2*nwall;

  // error checks

  if (nwall == 0) error->all(FLERR,"Illegal fix wall command");
  for (int m = 0; m < nwall; m++)
    if (cutoff[m] <= 0.0)
      error->all(FLERR,"Fix wall cutoff <= 0.0");

  for (int m = 0; m < nwall; m++)
    if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->dimension == 2)
      error->all(FLERR,"Cannot use fix wall zlo/zhi for a 2d simulation");

  if (!pbcflag) {
    for (int m = 0; m < nwall; m++) {
      if ((wallwhich[m] == XLO || wallwhich[m] == XHI) && domain->xperiodic)
        error->all(FLERR,"Cannot use fix wall in periodic dimension");
      if ((wallwhich[m] == YLO || wallwhich[m] == YHI) && domain->yperiodic)
        error->all(FLERR,"Cannot use fix wall in periodic dimension");
      if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->zperiodic)
        error->all(FLERR,"Cannot use fix wall in periodic dimension");
    }
  }

  // scale factors for wall position for CONSTANT and VARIABLE walls

  int flag = 0;
  for (int m = 0; m < nwall; m++)
    if (xstyle[m] != EDGE) flag = 1;

  if (flag) {
    if (scaleflag) {
      xscale = domain->lattice->xlattice;
      yscale = domain->lattice->ylattice;
      zscale = domain->lattice->zlattice;
    }
    else xscale = yscale = zscale = 1.0;

    for (int m = 0; m < nwall; m++) {
      if (xstyle[m] != CONSTANT) continue;
      if (wallwhich[m] < YLO) coord0[m] *= xscale;
      else if (wallwhich[m] < ZLO) coord0[m] *= yscale;
      else coord0[m] *= zscale;
    }
  }

  // set xflag if any wall positions are variable
  // set varflag if any wall positions or parameters are variable
  // set wstyle to VARIABLE if either epsilon or sigma is a variable

  varflag = xflag = 0;
  for (int m = 0; m < nwall; m++) {
    if (xstyle[m] == VARIABLE) xflag = 1;
    if (xflag || estyle[m] == VARIABLE || sstyle[m] == VARIABLE) varflag = 1;
    if (estyle[m] == VARIABLE || sstyle[m] == VARIABLE) wstyle[m] = VARIABLE;
    else wstyle[m] = CONSTANT;
  }

  eflag = 0;
  for (int m = 0; m <= nwall; m++) ewall[m] = 0.0;
}

/* ---------------------------------------------------------------------- */

FixWallForce::~FixWallForce()
{
  for (int m = 0; m < nwall; m++) {
    delete [] xstr[m];
    delete [] estr[m];
    delete [] sstr[m];
  }
}

/* ---------------------------------------------------------------------- */

int FixWallForce::setmask()
{
  int mask = 0;

  // FLD implicit needs to invoke wall forces before pair style

  if (fldflag) mask |= PRE_FORCE;
  else mask |= POST_FORCE;

  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallForce::init()
{
  // setup coefficients

  for (int m = 0; m < nwall; m++) {
    coordinate[m] = coord0[m];
    coordinate_old[m] = coord0[m];
    precompute(m);
  }

  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallForce::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet")) {
    if (!fldflag) post_force(vflag);
  } else {
    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallForce::min_setup(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   only called if fldflag set, in place of post_force
------------------------------------------------------------------------- */

void FixWallForce::pre_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallForce::post_force(int vflag)
{
  eflag = 0;
  for (int m = 0; m <= nwall; m++) ewall[m] = 0.0;

  // coord = current position of wall
  // evaluate variables if necessary, wrap with clear/add
  // for epsilon/sigma variables need to re-invoke precompute()

  if (varflag) modify->clearstep_compute();

  for (int m = 0; m < nwall; m++){
    total_force = wall_particle(m,wallwhich[m],coordinate[m]);
    wall_integration(m, force_push[m]+total_force);
    precompute(m);

  }

  if (varflag) modify->addstep_compute(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */
void FixWallForce::wall_integration(int m, double force) {
  if (friction[m] > 0) {
    //double old_coord = coordinate[m];
    //double friction = 10000;
    //double velocity = (coordinate[m] - coordinate_old[m])/update->dt;
    //force -= friction*velocity;
    //coordinate[m] += coordinate[m] - coordinate_old[m] + force / mass[m] * update->dt * update->dt;
    //coordinate_old[m] = old_coord;

    coordinate[m] += force/friction[m]*update->dt;

  }
}


/* ---------------------------------------------------------------------- */

void FixWallForce::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallForce::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of wall interaction
------------------------------------------------------------------------- */

double FixWallForce::compute_scalar()
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall,ewall_all,nwall+1,MPI_DOUBLE,MPI_SUM,world);
    eflag = 1;
  }
  return ewall_all[0];
}

/* ----------------------------------------------------------------------
   components of force on wall
------------------------------------------------------------------------- */

double FixWallForce::compute_vector(int n)
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall,ewall_all,nwall+1,MPI_DOUBLE,MPI_SUM,world);
    eflag = 1;
  }

  if (n < nwall){
    return ewall_all[n+1];
  } else {
    return coordinate[n - nwall];
  }
}
