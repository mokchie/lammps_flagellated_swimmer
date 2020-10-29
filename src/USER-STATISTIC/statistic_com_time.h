/* ----------------------------------------------------------------------
   Kathrin Mueller 31/03/15 - accumulation of statistics

   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef STATISTIC_CLASS

StatisticStyle(com/time,StatisticCOMTime)

#else

#ifndef LMP_STATISTIC_COMTIME_H
#define LMP_STATISTIC_COMTIME_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticCOMTime : public Statistic {
 public:
  StatisticCOMTime(class LAMMPS *, int, char **);
  ~StatisticCOMTime();

 private:
  double *com, *com_all, *com_vel, *com_vel_all, *cyl_com;
  double **com_av, **com_vel_av;
  tagint *mol_list;
  int init_on, writetest;
  tagint n_stat;
  void write_stat(bigint);
  void calc_stat();
  void shift_coordinates(double&, double&, double&);
};

}

#endif
#endif
