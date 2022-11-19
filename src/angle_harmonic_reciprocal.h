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

#ifdef ANGLE_CLASS

AngleStyle(harmonic/reciprocal,AngleHarmonicReciprocal)

#else

#ifndef LMP_ANGLE_HARMONIC_RECIPROCAL_H
#define LMP_ANGLE_HARMONIC_RECIPROCAL_H

#include <cstdio>
#include "angle.h"

namespace LAMMPS_NS {

class AngleHarmonicReciprocal : public Angle {
 public:
  AngleHarmonicReciprocal(class LAMMPS *);
  virtual ~AngleHarmonicReciprocal();
  virtual void compute(int, int);
  void coeff(int, char **);
  double equilibrium_angle(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  virtual double single(int, int, int, int);

 protected:
  double *k,*b,*omega,*theta0,*skew,*tau;
  double shiftmap(double, double, double, double, double);
  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for angle coefficients

Self-explanatory.  Check the input script or data file.

*/
