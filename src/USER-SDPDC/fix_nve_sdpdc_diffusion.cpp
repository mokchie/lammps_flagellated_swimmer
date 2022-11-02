/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing author: Chaojie Mo
------------------------------------------------------------------------- */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fix_nve_sdpdc_diffusion.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "domain.h"
#include "group.h"
#include "memory.h"


using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVESDPDCDiffusion::FixNVESDPDCDiffusion(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  int grph;
  if (!atom->conc_flag) error->all(FLERR,"compound concentration is not defined, unable to use fix/nve/sdpdc/diffusion");
  if (narg != 5+atom->individual) error->all(FLERR,"Illegal fix nve/sdpdc/diffusion command");
  if (!(strcmp(arg[3],"x")==0 || 
        strcmp(arg[3],"y")==0 || 
        strcmp(arg[3],"z")==0 || 
        strcmp(arg[3],"xy")==0 || 
        strcmp(arg[3],"yx")==0 ||
        strcmp(arg[3],"yz")==0 || 
        strcmp(arg[3],"zy")==0 || 
        strcmp(arg[3],"xz")==0 ||
        strcmp(arg[3],"zx")==0 ||
        strcmp(arg[3],"xzy")==0 ||
        strcmp(arg[3],"yxz")==0 ||
        strcmp(arg[3],"yzx")==0 ||
        strcmp(arg[3],"zxy")==0 ||
        strcmp(arg[3],"zyx")==0 ||
        strcmp(arg[3],"xyz")==0 ||
        strcmp(arg[3],"NULL")==0))
    error->all(FLERR,"Incorrect reset boundary specified in fix/nve/sdpdc");
  else{
    int n = strlen(arg[3])+1;
    resetbd = new char[n];
    strcpy(resetbd,arg[3]);
  }
  grph = group->find(arg[4]);
  if (grph == -1) error->all(FLERR,"Could not find group ID in fix/nve/sdpdc");
  groupbit_reset = group->bitmask[grph];
  memory->create(C0,atom->individual,"nve_sdpdc:C0");
  for(int k=0; k<atom->individual; k++){
    C0[k] = force->numeric(FLERR,arg[5+k]);
  }

  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */
FixNVESDPDCDiffusion::~FixNVESDPDCDiffusion(){
  memory->destroy(C0);
  delete [] resetbd;
}

/* ----------------------------------------------------------------------- */
int FixNVESDPDCDiffusion::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDCDiffusion::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVESDPDCDiffusion::initial_integrate(int vflag)
{
  double dtfm, dtfmom;

  // update v, x and omega of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double **conc = atom->conc;
  double **f_conc = atom->f_conc;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double *moment = atom->moment;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int outside;
  double *lo, *hi;
  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;
  double *boxlo_lamda = domain->boxlo_lamda;
  double *boxhi_lamda = domain->boxhi_lamda;
  double lamda[3];
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  // if (rmass) {
  //   for (int i = 0; i < nlocal; i++)
  //     if (mask[i] & groupbit) {
  //       dtfm = dtf / rmass[i];
  //       dtfmom = dtf / moment[type[i]]; 
  //       v[i][0] += dtfm * f[i][0];
  //       v[i][1] += dtfm * f[i][1];
  //       v[i][2] += dtfm * f[i][2];
  //       x[i][0] += dtv * v[i][0];
  //       x[i][1] += dtv * v[i][1];
  //       x[i][2] += dtv * v[i][2];
  //       omega[i][0] += dtfmom * torque[i][0];
  //       omega[i][1] += dtfmom * torque[i][1];
  //       omega[i][2] += dtfmom * torque[i][2];
  //     }

  // } else {
  //   for (int i = 0; i < nlocal; i++)
  //     if (mask[i] & groupbit) {
  //       dtfm = dtf / mass[type[i]];
  //       dtfmom = dtf / moment[type[i]]; 
  //       v[i][0] += dtfm * f[i][0];
  //       v[i][1] += dtfm * f[i][1];
  //       v[i][2] += dtfm * f[i][2];
  //       x[i][0] += dtv * v[i][0];
  //       x[i][1] += dtv * v[i][1];
  //       x[i][2] += dtv * v[i][2];
  //       omega[i][0] += dtfmom * torque[i][0];
  //       omega[i][1] += dtfmom * torque[i][1];
  //       omega[i][2] += dtfmom * torque[i][2];
  //     }
  // }

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit){
      outside = 0;
      if (domain->triclinic == 0) {
        lo = boxlo;
        hi = boxhi;

        if ((strchr(resetbd,'x') != NULL && (x[i][0]<lo[0] || x[i][0]>hi[0])) ||
            (strchr(resetbd,'y') != NULL && (x[i][1]<lo[1] || x[i][1]>hi[1])) ||
            (strchr(resetbd,'z') != NULL && (x[i][2]<lo[2] || x[i][2]>hi[2]))) outside = 1;
        else outside = 0;

      } else {
        lo = boxlo_lamda;
        hi = boxhi_lamda;

        domain->x2lamda(x[i],lamda);

        if ((strchr(resetbd,'x') != NULL && (lamda[0]<lo[0] || lamda[0]>hi[0])) ||
            (strchr(resetbd,'y') != NULL && (lamda[1]<lo[1] || lamda[1]>hi[1])) ||
            (strchr(resetbd,'z') != NULL && (lamda[2]<lo[2] || lamda[2]>hi[2]))) outside = 1;
        else outside = 0;

      }
      if(outside && mask[i] & groupbit_reset)
        for (int k = 0; k < atom->individual; k++)
           conc[i][k] = C0[k];
      else
        for (int k = 0; k < atom->individual; k++)
           conc[i][k] += dtf * f_conc[i][k];
    }
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDCDiffusion::final_integrate()
{
  double dtfm, dtfmom;

  // update v and omega of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double **x = atom->x;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double **conc = atom->conc;
  double **f_conc = atom->f_conc;  
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double *moment = atom->moment;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *lo, *hi;
  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;
  double *boxlo_lamda = domain->boxlo_lamda;
  double *boxhi_lamda = domain->boxhi_lamda;
  double lamda[3];  
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // if (rmass) {
    // for (int i = 0; i < nlocal; i++)
  //     if (mask[i] & groupbit) {
  //       dtfm = dtf / rmass[i];
  //       dtfmom = dtf / moment[type[i]]; 
  //       v[i][0] += dtfm * f[i][0];
  //       v[i][1] += dtfm * f[i][1];
  //       v[i][2] += dtfm * f[i][2];
  //       omega[i][0] += dtfmom * torque[i][0];
  //       omega[i][1] += dtfmom * torque[i][1];
  //       omega[i][2] += dtfmom * torque[i][2]; 
  //     }

  // } else {
  //   for (int i = 0; i < nlocal; i++)
  //     if (mask[i] & groupbit) {
  //       dtfm = dtf / mass[type[i]];
  //       dtfmom = dtf / moment[type[i]]; 
  //       v[i][0] += dtfm * f[i][0];
  //       v[i][1] += dtfm * f[i][1];
  //       v[i][2] += dtfm * f[i][2];
  //       omega[i][0] += dtfmom * torque[i][0];
  //       omega[i][1] += dtfmom * torque[i][1];
  //       omega[i][2] += dtfmom * torque[i][2]; 
  //     }
  // }
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit){
      for (int k = 0; k < atom->individual; k++)
        conc[i][k] += dtf * f_conc[i][k];
    }
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDCDiffusion::initial_integrate_respa(int vflag, int ilevel, int iloop)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v, x and omega
  // all other levels - NVE update of v and omega

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDCDiffusion::final_integrate_respa(int ilevel, int iloop)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVESDPDCDiffusion::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */
