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
#include <cstdlib>
#include "fix_lees_edwards.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "domain.h"
#include "update.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixLeesEdwards::FixLeesEdwards(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{ 
  if (narg != 6) error->all(FLERR,"Illegal fix lees/edwards command");

  uu = 0.0;
  le_ind = force->inumeric(FLERR,arg[3]);
  if (le_ind)
    uu = force->numeric(FLERR,arg[4]);
  else
    const_shift = force->numeric(FLERR,arg[4]);
  shift = force->numeric(FLERR,arg[5]);
    
  comm->le = 1;
  comm->u_le = uu;
  restart_global = 1;
}

/* ---------------------------------------------------------------------- */

int FixLeesEdwards::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixLeesEdwards::init()
{
  dtv = update->dt;
  if (le_ind)
    comm->shift = shift;
  else
    comm->shift = const_shift;
}

/* ---------------------------------------------------------------------- */

void FixLeesEdwards::pre_exchange()
{
  int i;
  imageint idim,otherdims;
  double *lo,*hi,*period;
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  imageint *image = atom->image;

  if (domain->xperiodic == 0) error->all(FLERR,"Cannot use fix lees/edwards with non-periodicity in x direction!");

  lo = domain->boxlo;
  hi = domain->boxhi;
  period = domain->prd;

  for (i = 0; i < nlocal; i++) {
    if (x[i][0] < lo[0]) {
      x[i][0] += period[0];
      idim = image[i] & IMGMASK;
      otherdims = image[i] ^ idim;
      idim--;
      idim &= IMGMASK;
      image[i] = otherdims | idim;

      x[i][1] += comm->shift;
      v[i][1] += comm->u_le;
      while (x[i][1] >= hi[1]){
        x[i][1] -= period[1];
        x[i][1] = MAX(x[i][1],lo[1]);
        idim = (image[i] >> IMGBITS) & IMGMASK;
        otherdims = image[i] ^ (idim << IMGBITS);
        idim++;
        idim &= IMGMASK;
        image[i] = otherdims | (idim << IMGBITS);
      }
      while (x[i][1] < lo[1]){
        x[i][1] += period[1];
        idim = (image[i] >> IMGBITS) & IMGMASK;
        otherdims = image[i] ^ (idim << IMGBITS);
        idim--;
        idim &= IMGMASK;
        image[i] = otherdims | (idim << IMGBITS);
      }
    }
    if (x[i][0] >= hi[0]) {
      x[i][0] -= period[0];
      x[i][0] = MAX(x[i][0],lo[0]);
      idim = image[i] & IMGMASK;
      otherdims = image[i] ^ idim;
      idim++;
      idim &= IMGMASK;
      image[i] = otherdims | idim;

      x[i][1] -= comm->shift;
      v[i][1] -= comm->u_le;
      while (x[i][1] >= hi[1]){
        x[i][1] -= period[1];
        x[i][1] = MAX(x[i][1],lo[1]);
        idim = (image[i] >> IMGBITS) & IMGMASK;
        otherdims = image[i] ^ (idim << IMGBITS);
        idim++;
        idim &= IMGMASK;
        image[i] = otherdims | (idim << IMGBITS);
      }
      while (x[i][1] < lo[1]){
        x[i][1] += period[1];
        idim = (image[i] >> IMGBITS) & IMGMASK;
        otherdims = image[i] ^ (idim << IMGBITS);
        idim--;
        idim &= IMGMASK;
        image[i] = otherdims | (idim << IMGBITS);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixLeesEdwards::post_integrate()
{ 
  if (le_ind == 0) uu = 0.0;
  comm->u_le = uu;

  if (le_ind)
    comm->shift += dtv*uu;
  else
    comm->shift = const_shift;

  shift = comm->shift;
}

/* ----------------------------------------------------------------------
 *    pack entire state of Fix into one write
 *    ------------------------------------------------------------------------- */

void FixLeesEdwards::write_restart(FILE *fp)
{
  int l;

  if (comm->me == 0){
    l = sizeof(double);
    fwrite(&l,sizeof(int),1,fp);
    fwrite(&shift,sizeof(double),1,fp);
  }
}

/* ----------------------------------------------------------------------
 *    use state info from restart file to restart the Fix
 *    ------------------------------------------------------------------------- */

void FixLeesEdwards::restart(char *buf)
{
  double *list = (double *) buf;

  shift = list[0];
}

/* ---------------------------------------------------------------------- */

