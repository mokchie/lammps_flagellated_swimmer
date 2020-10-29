/* -*- c++ -*- ----------------------------------------------------------
   Dmitry Fedosov 08/12/05 - accumulation of statistics

   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef STATISTIC_CLASS

StatisticStyle(com,StatisticCOM)

#else

#ifndef LMP_STATISTIC_COM_H
#define LMP_STATISTIC_COM_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticCOM : public Statistic {
 public:
  StatisticCOM(class LAMMPS *, int, char **);
  ~StatisticCOM();

 private:
  double *c_m, *c_mt, **rdh;
  double ***c_dist;
  double ***gx, ***gy, ***gz, ***vcx, ***vcy, ***vcz;
  double ***fx, ***fy, ***fz;
  tagint *mol_list;
  double nm_hpm1;
  int init_on;
  tagint nm, nm_h;
 
  void write_stat(bigint);
  void calc_stat();

};

}

#endif
#endif
