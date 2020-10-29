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

StatisticStyle(dens,StatisticDens)

#else


#ifndef LMP_STATISTIC_DENS_H
#define LMP_STATISTIC_DENS_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticDens : public Statistic {
 public:
  StatisticDens(class LAMMPS *, int, char **);
  ~StatisticDens();

 private:
  double ***num;
  double ***nmass;

  void write_stat(bigint);
  void calc_stat();
};

}

#endif
#endif
