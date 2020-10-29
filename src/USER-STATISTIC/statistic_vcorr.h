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

StatisticStyle(vcorr,StatisticVCorr)

#else


#ifndef LMP_STATISTIC_VCORR_H
#define LMP_STATISTIC_VCORR_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticVCorr : public Statistic {
 public:
  StatisticVCorr(class LAMMPS *, int, char **);
  ~StatisticVCorr();

 private:
  int st_accum, nwind, igroup;
  int *corr_map, *rot_list;
  double time_corr, num_part, masstotal;
  double **vel, *corr; 

  void write_stat(bigint);
  void calc_stat();
};

}

#endif
#endif
