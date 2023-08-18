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

AngleStyle(harmonic/theta/QL,AngleHarmonicThetaQL)

#else

#ifndef LMP_ANGLE_THETA_QL_H
#define LMP_ANGLE_THETA_QL_H

#include <cstdio>
#include "angle.h"

namespace LAMMPS_NS {

class AngleHarmonicThetaQL : public Angle {
 public:
  AngleHarmonicThetaQL(class LAMMPS *);
  virtual ~AngleHarmonicThetaQL();
  virtual void compute(int, int);
  void coeff(int, char **);
  double equilibrium_angle(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  void write_Qmatrix(FILE *);
  virtual double single(int, int, int, int);

 protected:
  double *k,*b,*omega,*theta0min,*theta0max,*skew,*tau,*alpha,*gamma,*epsilon, *xtarget, *ytarget, *ztarget;
  double *theta0;

  // alpha is the learning rate; gamma is the far-sighted factor; epsilon is the greedy factor;
  // theta0max is the amplitude of the largest offset
  int *Nl; // Nl is the discrete number between [-theta0max,theta0max]
  double ***Qmatrix;
  double *Q0;
  // Qmatrix is the Q matrix. The second index is the State value, the third index is the Action value.
  // For state value i, it means the current theta0 is -theta0max + i*(2*theta0max)/(Nl-1)
  // For action value j, if j==0, decrease theta0, elif j==1, no action, elif==2, increase theta0;
  double *dist0,*dist;
  double *T_t;
  double *reward; // rewards
  int **sa; //state and action
  //int *imin;

  class RanMars *random;
  class MTRand *mtrand;
  double shiftmap(double, double, double, double, double);
  void allocate();
  //void update_omega();
  // double gaussian();
  bool Qallocated,*first_learn;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for angle coefficients

Self-explanatory.  Check the input script or data file.

*/
