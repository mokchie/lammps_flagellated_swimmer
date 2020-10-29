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

StatisticStyle(trj,StatisticTRJ)

#else

#ifndef LMP_STATISTIC_TRJ_H
#define LMP_STATISTIC_TRJ_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticTRJ : public Statistic {
 public:
  StatisticTRJ(class LAMMPS *, int, char **);
  ~StatisticTRJ();

 private:
  int *mol_flag;
  char f_name[FILENAME_MAX];
  
  double *xcm, *ycm, *zcm, *xcm_tmp, *ycm_tmp, *zcm_tmp;
  double *Rg, *Rsx_tmp, *Rsy_tmp, *Rsz_tmp, *Rg_tmp;
  double *dxmin, *dymin, *dzmin, *dxmin_tmp, *dymin_tmp, *dzmin_tmp;
  double *dxmax, *dymax, *dzmax, *dxmax_tmp, *dymax_tmp, *dzmax_tmp;
  
  void write_stat(bigint);
  void calc_stat();

};

}

#endif
#endif
