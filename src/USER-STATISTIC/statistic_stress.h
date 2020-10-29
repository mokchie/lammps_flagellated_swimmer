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

StatisticStyle(stress,StatisticStress)

#else


#ifndef LMP_STATISTIC_STRESS_H
#define LMP_STATISTIC_STRESS_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticStress : public Statistic {
 public:
  StatisticStress(class LAMMPS *, int, char **);
  ~StatisticStress();

  void virial1(int);
  void virial2(int, double[], int);
  void virial3(int, int, double[]);
  void virial4(int, int, int, double[]);
  void virial5(int, int, int, int, double[]);
  void virial6(int, double[], double[]);

 private: 
  double ****ss, ****ss_p, ****vv, ****vv_p;
  double ****ss1, ****ss2, ****ss_p1, ****ss_p2;
  int poly_ind, is1, js1, ks1, map1;

  void write_stat(bigint);
  void calc_stat();
};

}

#endif
#endif
