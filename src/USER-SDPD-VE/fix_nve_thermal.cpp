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
#include "fix_nve_thermal.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVEThermal::FixNVEThermal(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal fix nve/thermal command");

  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixNVEThermal::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVEThermal::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  c_v = atom->c_v;

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVEThermal::initial_integrate(int vflag)
{
  double dtfm,vh;

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double *e = atom->e;
  double *de = atom->de;
  double *pte0 = atom->pte0;
  double *pte1 = atom->pte1;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        vh = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        e[i] += 0.5 * dtv * de[i];
        e[i] -= 0.5*rmass[i]*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2] - vh)/c_v;
        pte0[i] = pte1[i];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        vh = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        e[i] += 0.5 * dtv * de[i];
        e[i] -= 0.5*mass[type[i]]*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2] - vh)/c_v;
        pte0[i] = pte1[i];
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVEThermal::final_integrate()
{
  double dtfm,vh;

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double *e = atom->e;
  double *de = atom->de;
  double *pte0 = atom->pte0;
  double *pte1 = atom->pte1;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  //double energ, energt;
  //energ = 0.0;
  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        vh = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        e[i] += 0.5 * dtv * de[i];
        e[i] -= (0.5*rmass[i]*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2] - vh) + pte1[i] - pte0[i])/c_v;
        //energ += pte1[i] + 0.5*rmass[i]*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]) + e[i]*c_v;
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        vh = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        e[i] += 0.5 * dtv * de[i];
        e[i] -= (0.5*mass[type[i]]*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2] - vh) + pte1[i] - pte0[i])/c_v;
        //energ += pte1[i] + 0.5*mass[type[i]]*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]) + e[i]*c_v;
      }
  }
  
  /*if (update->ntimestep%1000 == 0){
    energt = 0.0;
    if (comm->nprocs > 1) 
      MPI_Reduce(&energ, &energt, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    else
      energt = energ; 
    if (comm->me == 0)
      printf("energ_tot = %lf \n",energt); 
  }*/
}

/* ---------------------------------------------------------------------- */

void FixNVEThermal::initial_integrate_respa(int vflag, int ilevel, int iloop)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVEThermal::final_integrate_respa(int ilevel, int iloop)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVEThermal::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
