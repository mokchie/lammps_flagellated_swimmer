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

#ifdef BOND_CLASS

BondStyle(indivharmonic/QL,BondIndivHarmonicQL)

#else

#ifndef LMP_BOND_INDIVHARMONIC_QL_H
#define LMP_BOND_INDIVHARMONIC_QL_H

#include <cstdio>
#include "bond.h"

namespace LAMMPS_NS {

class BondIndivHarmonicQL : public Bond {
 public:
  BondIndivHarmonicQL(class LAMMPS *);
  virtual ~BondIndivHarmonicQL();
  virtual void compute(int, int);
  void coeff(int, char **);
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  double single(int, double, int, int, double &);
  void write_Qmatrix(FILE *);

 protected:
  double *k,*r0,*r0min,*r0max,*ra,*omega,*tau;
  double *alpha,*gamma,*epsilon, *xtarget, *ytarget, *ztarget;
  int *follow_btype;
  int *Nl; // Nl is the discrete number between [-r0min,r0max]
  double ***Qmatrix;
  double *Q0;
  // Qmatrix is the Q matrix. The second index is the State value, the third index is the Action value.
  // For state value i, it means the current theta0 is -theta0max + i*(2*theta0max)/(Nl-1)
  // For action value j, if j==0, decrease theta0, elif j==1, no action, elif==2, increase theta0;
  double *dist0,*dist;
  double *T_t;
  double *reward; // rewards
  int **sa; //state and action
  bool Qallocated,*first_learn;
  int *QL_on;

  class RanMars *random;
  class MTRand *mtrand;  

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

*/
