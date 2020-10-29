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

AngleStyle(harmonic/stochastic/omega,AngleStochasticOmega)

#else

#ifndef LMP_ANGLE_STOCHASTIC_OMEGA_H
#define LMP_ANGLE_STOCHASTIC_OMEGA_H

#include <cstdio>
#include "angle.h"

namespace LAMMPS_NS {

class AngleStochasticOmega : public Angle {
 public:
  AngleStochasticOmega(class LAMMPS *);
  virtual ~AngleStochasticOmega();
  virtual void compute(int, int);
  void coeff(int, char **);
  double equilibrium_angle(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  virtual double single(int, int, int, int);

 protected:
  double *k,*b,*omega,*omega_sigma,*theta0,*skew,*tau;
  double *omg_t, *T_t;
  class RanMars *random;
  class MTRand *mtrand;
  double shiftmap(double, double, double, double, double);
  void allocate();
  void update_omega();
  double gaussian();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for angle coefficients

Self-explanatory.  Check the input script or data file.

*/
