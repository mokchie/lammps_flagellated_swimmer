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

#ifdef PAIR_CLASS

PairStyle(sdpd/full,PairSDPDFull)

#else

#ifndef LMP_PAIR_SDPD_FULL_H
#define LMP_PAIR_SDPD_FULL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSDPDFull : public Pair {
 public:
  PairSDPDFull(class LAMMPS *);
  virtual ~PairSDPDFull();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  double single(int, int, int, int, double, double, double, double &);

 protected:
  int seed,nw_max,set_weight_ind,g_exp,groupbit,groupbit_rho,num_fix_bc,init_fix;
  int *list_fix_bc;
  double cut_global,temperature,p0,bb,dr_inv,rho_reset;
  double **cut, **eta, **rho0, **zeta;
  double wrr[4], delx, dely, delz;
  class RanMars *random;
  class MTRand *mtrand;
  double ***weight1,***weight2;

  void allocate();
  void set_weight();
  void generate_wrr();
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  int pack_forward_comm(int , int *, double *, int, int *);
  void unpack_forward_comm(int , int , double *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair sdpd/no/moment requires ghost atoms store velocity

Use the communicate vel yes command to enable this.

W: Pair sdpd/no/moment needs newton pair on for momentum conservation

Self-explanatory.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
