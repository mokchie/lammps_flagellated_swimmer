/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(setvel/var,FixSetVelVar)

#else

#ifndef LMP_FIX_SETVELVAR_H
#define LMP_FIX_SETVELVAR_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSetVelVar : public Fix {
 public:
  FixSetVelVar(class LAMMPS *, int, char **);
  ~FixSetVelVar();
  int setmask();
  void init();
  void post_integrate();

 private:
  char *xvarstr, *yvarstr, *zvarstr;
  double xvalue,yvalue,zvalue;
  int xstyle,ystyle,zstyle,xvar,yvar,zvar;
  int xvarstyle,yvarstyle,zvarstyle;
  int ind_const,iregion,step_each;
  char *idregion;
  class RanPark *random;
};

}

#endif
#endif

