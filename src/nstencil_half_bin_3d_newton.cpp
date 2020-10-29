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

#include "nstencil_half_bin_3d_newton.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NStencilHalfBin3dNewton::NStencilHalfBin3dNewton(LAMMPS *lmp) : NStencil(lmp) {}

/* ----------------------------------------------------------------------
   create stencil based on bin geometry and cutoff
------------------------------------------------------------------------- */

void NStencilHalfBin3dNewton::create()
{
  int i,j,k;

  nstencil = 0;

  if (comm->le){
    for (k = -sz; k <= sz; k++)
      for (j = -sy; j <= sy; j++)
        for (i = 0; i <= sx; i++)
          if (i > 0 || j > 0 || (j == 0 && k > 0))
            if (bin_distance(i,j,k) < cutneighmaxsq)
              stencil[nstencil++] = k*mbiny*mbinx + j*mbinx + i;
  } else{
    for (k = 0; k <= sz; k++)
      for (j = -sy; j <= sy; j++)
        for (i = -sx; i <= sx; i++)
          if (k > 0 || j > 0 || (j == 0 && i > 0))
            if (bin_distance(i,j,k) < cutneighmaxsq)
              stencil[nstencil++] = k*mbiny*mbinx + j*mbinx + i;
  }
}
