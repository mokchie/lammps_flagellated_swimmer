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
   Contributing author: Masoud Hoore (FZJ)
------------------------------------------------------------------------- */

/*------------------------------------------------------------------------
		Created on  : 17 June 2015
		Modified on : 20 Dec 2017
		Created by  : Masoud Hoore
------------------------------------------------------------------------- */
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "fix_polymer_activate.h"
#include "atom.h"
#include "update.h"
#include "respa.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"


using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- 
Syntax: fix ID group-ID polymer/activate style style_keywords Nevery itype jtype btype
             0        1                2     3            3+
style:  angle density unwrap
	angle keywords:		theta_cut
	density keywords:	cutoff  
	unwrap keywords:	cutoff theta_cut
	cutoff:    cutoff for density calculation to measure polymer masking
	theta_cut: cutoff angle to measure stretching of polymer
Nevery: the fix is implemented on each Nevery steps
itype:  polymer non-interacting particle type
jtype:  polymer interacting particle type
btype:  polymer adhesive bonds bond type
-------------------------------------------------------------------------*/

FixPolymerActivate::FixPolymerActivate(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  arg_shift = 0;
  if (narg < 4) error->all(FLERR,"Illegal fix polymer/activate command");
  else if (strcmp(arg[3],"angle") == 0) {
    angoff = force->numeric(FLERR,arg[4]);
    if (angoff < 0.0 || angoff > 180.0) error->all(FLERR,"Illegal angle in fix polymer/activate command -- theta_cut must be in [0,180] range ");
    cos_angoff = cos(angoff/180.0*3.14159);
    arg_shift = 1;
    styleflag = 2;
    }
  else if (strcmp(arg[3],"density") == 0) {
    cutoff = force->numeric(FLERR,arg[4]);
    arg_shift = 1;
    styleflag = 1;
    }
  else if (strcmp(arg[3],"unwrap") == 0) {
    cutoff = force->numeric(FLERR,arg[4]);
    angoff = force->numeric(FLERR,arg[5]);
    if (angoff < 0.0 || angoff > 180.0) error->all(FLERR,"Illegal angle in fix polymer/activate command -- theta_cut must be in [0,180] range ");
    cos_angoff = cos(angoff/180.0*3.14159);
    arg_shift = 2;
    styleflag = 0;
    }
  else error->all(FLERR,"Illegal fix polymer/activate command");
  if (narg < 8 + arg_shift) error->all(FLERR,"Illegal fix polymer/activate command");

  nevery = force->inumeric(FLERR,arg[4 + arg_shift]);
  if (nevery < neighbor->every) error->all(FLERR,"polymer/activate Nevery is less than neigh_modify every");
  itype = force->inumeric(FLERR,arg[5 + arg_shift]);
  jtype = force->inumeric(FLERR,arg[6 + arg_shift]);
  btype = force->inumeric(FLERR,arg[7 + arg_shift]);

  cutoffsq = cutoff*cutoff;
}

/* ---------------------------------------------------------------------- */

int FixPolymerActivate::setmask()
{
  int mask = 0;
  mask |= FINAL_INTEGRATE;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPolymerActivate::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixPolymerActivate::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixPolymerActivate::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ------------------------------------------------------------------------------- 
	change the types of the atoms of the specified polymer group 
		from itype to jtype when the polymer unwraps
----------------------------------------------------------------------------------*/

void FixPolymerActivate::final_integrate()
{
  if (update->ntimestep % nevery) return;

  tagint *tag = atom->tag;
  double **x = atom->x;
  double *rho = atom->rho;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  tagint *molecule = atom->molecule;

  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;

  int inum, jnum, imolecule, jmolecule;
  int *ilist,*jlist,*numneigh,**firstneigh;

  //neighbor->build_one(list, 1);
  //inum = list->inum;				// # of I atoms neighbors are stored for
  //ilist = list->ilist;					// local indices of I atoms
  //numneigh = list->numneigh;		// # of J neighbors for each I atom
  //firstneigh = list->firstneigh;			// ptr to 1st J int value of each I atom

  int jj, kk;
  int i, j;
  if (styleflag == 0) {
    // loop over neighbors of my atoms to calculate both number density and angle
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit  &&  atom->num_bond[i] < 3) {
        num_dens = 0;
        ang_flag = 0;
        jj = 0;
        kk = 0;
        for (j = 0; j < nall; j++) {
          if (mask[j] & groupbit)
	    if (molecule[i] == molecule[j]  &&  tag[i] != tag[j]) {
              delx = x[i][0] - x[j][0];
              dely = x[i][1] - x[j][1];
              delz = x[i][2] - x[j][2];
              distsq = delx*delx + dely*dely + delz*delz;
	      if (distsq < cutoffsq) {num_dens += 1;}
              // finding angles
              if (tag[i] == tag[j] + 1) {jj = j;}
             else if (tag[i] == tag[j] - 1) {kk = j;}
	      }
	  }
        // if an angle exists
        if ( kk != 0  &&  jj != 0) {
          delx = x[jj][0] - x[i][0];
          dely = x[jj][1] - x[i][1];
          delz = x[jj][2] - x[i][2];
          distsq = delx*delx + dely*dely + delz*delz;
          delx2 = x[kk][0] - x[i][0];
          dely2 = x[kk][1] - x[i][1];
          delz2 = x[kk][2] - x[i][2];
          distsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
          cos_theta = delx*delx2 + dely*dely2 + delz*delz2;
          cos_theta = cos_theta / sqrt(distsq*distsq2);
          if ( cos_theta < cos_angoff) {ang_flag = 1;}
          }
        else {ang_flag = 1;}
        if ( (num_dens < 3)  &&  ang_flag == 1) {type[i] = jtype;}
        else {type[i] = itype;}
        }
      }
    }
if (styleflag == 1) {
    // loop over neighbors of my atoms to calculate number density
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit  &&  atom->num_bond[i] < 3) {
        num_dens = 0;
        ang_flag = 0;
        for (j = 0; j < nall; j++) {
          if (mask[j] & groupbit)
	    if (molecule[i] == molecule[j]  &&  tag[i] != tag[j]) {
              delx = x[i][0] - x[j][0];
              dely = x[i][1] - x[j][1];
              delz = x[i][2] - x[j][2];
              distsq = delx*delx + dely*dely + delz*delz;
	      if (distsq < cutoffsq) {num_dens += 1;}
	      }
	  }
        if (num_dens < 3) {type[i] = jtype;}
        else {type[i] = itype;}
        }
      }
    }
if (styleflag == 2) {
    // loop over neighbors of my atoms to calculate angle
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit  &&  atom->num_bond[i] < 3) {
        num_dens = 0;
        ang_flag = 0;
        jj = 0;
        kk = 0;
        for (int j = 0; j < nall; j++)
          if (mask[j] & groupbit)
	    if (molecule[i] == molecule[j]  &&  tag[i] != tag[j])
              if (tag[i] == tag[j] + 1) {jj = j;}
              else if (tag[i] == tag[j] - 1) {kk = j;}
        // if an angle exists
        if ( kk != 0  &&  jj != 0) {
	  delx = x[jj][0] - x[i][0];
          dely = x[jj][1] - x[i][1];
          delz = x[jj][2] - x[i][2];
          distsq = delx*delx + dely*dely + delz*delz;
          delx2 = x[kk][0] - x[i][0];
          dely2 = x[kk][1] - x[i][1];
          delz2 = x[kk][2] - x[i][2];
          distsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
          cos_theta = delx*delx2 + dely*dely2 + delz*delz2;
          cos_theta = cos_theta / sqrt(distsq*distsq2);
          if ( cos_theta < cos_angoff) {ang_flag = 1;}
          }
        if ( ang_flag == 1) {type[i] = jtype;}
        else {type[i] = itype;}
        }
      }
    }

  // reactivate bound particles (for satisfying detailed balance)
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int i1,i2;

  for (i = 0; i < nbondlist; i++) {
    i1 = bondlist[i][0];
    i2 = bondlist[i][1];
    if (bondlist[i][2] == btype) {
      if (type[i1] == itype) type[i1] = jtype;
      if (type[i2] == itype) type[i2] = jtype;
    }
  }

}

/* ---------------------------------------------------------------------- */

void FixPolymerActivate::final_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) final_integrate();
}

