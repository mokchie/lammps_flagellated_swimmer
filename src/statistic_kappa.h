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

StatisticStyle(kappa,StatisticKappa)

#else

#ifndef LMP_STATISTIC_KAPPA_H
#define LMP_STATISTIC_KAPPA_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticKappa : public Statistic {
 public:

  StatisticKappa(class LAMMPS *, int, char **);
   ~StatisticKappa();
 
  private:
   double ***num;
   double ***kappa11, ***kappa12, ***kappa13, ***kappa21, ***kappa22, ***kappa23, ***kappa31, ***kappa32, ***kappa33;
   void write_stat(bigint);
   void calc_stat();

  
};

}

#endif
#endif
