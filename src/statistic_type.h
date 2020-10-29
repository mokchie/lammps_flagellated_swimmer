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

StatisticStyle(type,StatisticType)

#else


#ifndef LMP_STATISTIC_TYPE_H
#define LMP_STATISTIC_TYPE_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticType : public Statistic {
 public:
  StatisticType(class LAMMPS *, int, char **);
  ~StatisticType();

 private:
  int itype, jtype;

  char f_name[FILENAME_MAX];

  void write_stat(bigint);
  void calc_stat();
};

}

#endif
#endif
