/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS

BondStyle(area/wlc/pow/visc,BondArea_WLC_POW_VISC)

#else

#ifndef LMP_BOND_AREA_WLC_POW_VISC_H
#define LMP_BOND_AREA_WLC_POW_VISC_H

#include <cstdio>
#include "bond.h"

namespace LAMMPS_NS {

class BondArea_WLC_POW_VISC : public Bond {
 public:
  BondArea_WLC_POW_VISC(class LAMMPS *);
  ~BondArea_WLC_POW_VISC();
  void compute(int, int);
  void coeff(int, char **);
  void init_style();
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  double single(int, double, int, int, double &);

 protected:
  double *Y0, *pp, *r0, *ka, *kl, *circum, *area, *temp;
  double *gamc, *gamt, *sigc, *sigt;
  double wrr[4], delx, dely, delz;
  double eng_vdwl;
  class RanMars *random;
  class MTRand *mtrand;  
  tagint n_mol_max, n_mol_limit;
  double *dath, *datt;

  void allocate();
  void generate_wrr();
};

}

#endif
#endif

/* ERROR/WARNING messages:

W: "WLC single bond too long: ...

Self-explanatory.

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

E: Individual has wrong value or is not set! Using bond wlc/pow/all/visc only possible with individual =1

Self-explanatory.  Check the input script second parameter at atom_style

E: Gamma_t > 3*Gamma_c

Gamma_t is to high for set Gamma_c. Change one of the values.

*/
      
