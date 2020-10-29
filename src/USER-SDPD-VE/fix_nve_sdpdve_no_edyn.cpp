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

#include <cstdio>
#include <cstring>
#include "fix_nve_sdpdve_no_edyn.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVESDPDVENoEdyn::FixNVESDPDVENoEdyn(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal fix nve/sdpdve/no/edyn command");

  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixNVESDPDVENoEdyn::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDVENoEdyn::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVESDPDVENoEdyn::initial_integrate(int vflag)
{
  double dtfm, dtfmom;

  // update v, x and omega of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double ***ctensor = atom->ctensor;
  double ***f_ctensor = atom->f_ctensor;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double *moment = atom->moment;
  int j,k;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        dtfmom = dtf / moment[type[i]]; 
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        omega[i][0] += dtfmom * torque[i][0];
        omega[i][1] += dtfmom * torque[i][1];
        omega[i][2] += dtfmom * torque[i][2];
        for (j=0; j<3; j++){
          for (k=0; k<3; k++){
            ctensor[i][j][k] += dtf * f_ctensor[i][j][k];
          }
        }
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        dtfmom = dtf / moment[type[i]]; 
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        omega[i][0] += dtfmom * torque[i][0];
        omega[i][1] += dtfmom * torque[i][1];
        omega[i][2] += dtfmom * torque[i][2]; 
        for (j=0; j<3; j++){
          for (k=0; k<3; k++){
            ctensor[i][j][k] += dtf * f_ctensor[i][j][k];
          }
        }
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDVENoEdyn::final_integrate()
{
  double dtfm, dtfmom;

  // update v and omega of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double ***ctensor = atom->ctensor;
  double ***f_ctensor = atom->f_ctensor;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double *moment = atom->moment;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int j,k;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        dtfmom = dtf / moment[type[i]]; 
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        omega[i][0] += dtfmom * torque[i][0];
        omega[i][1] += dtfmom * torque[i][1];
        omega[i][2] += dtfmom * torque[i][2];
        for (j=0; j<3; j++){
          for (k=0; k<3; k++){
            ctensor[i][j][k] += dtf * f_ctensor[i][j][k];
          }
        }
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        dtfmom = dtf / moment[type[i]]; 
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        omega[i][0] += dtfmom * torque[i][0];
        omega[i][1] += dtfmom * torque[i][1];
        omega[i][2] += dtfmom * torque[i][2]; 
        for (j=0; j<3; j++){
          for (k=0; k<3; k++){
            ctensor[i][j][k] += dtf * f_ctensor[i][j][k];
          }
        }
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDVENoEdyn::initial_integrate_respa(int vflag, int ilevel, int iloop)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v, x and omega
  // all other levels - NVE update of v and omega

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDVENoEdyn::final_integrate_respa(int ilevel, int iloop)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDVENoEdyn::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */
