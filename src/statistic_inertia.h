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

StatisticStyle(inertia,StatisticINERTIA)

#else

#ifndef LMP_STATISTIC_INERTIA_H
#define LMP_STATISTIC_INERTIA_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticINERTIA : public Statistic {
 public:
  StatisticINERTIA(class LAMMPS *, int, char **);
  ~StatisticINERTIA();

 private:
  int *mol_flag;
  char f_name[FILENAME_MAX];

  double *xcm, *ycm, *zcm, *xcm_tmp, *ycm_tmp, *zcm_tmp;
  double *vxcm, *vycm, *vzcm, *vxcm_tmp, *vycm_tmp, *vzcm_tmp;
  double *Ixx, *Iyy, *Izz, *Ixy, *Iyz, *Ixz, *Ixx_tmp, *Iyy_tmp, *Izz_tmp, *Ixy_tmp, *Iyz_tmp, *Ixz_tmp;
  double *Lx, *Ly, *Lz, *Lx_tmp, *Ly_tmp, *Lz_tmp;
  double *mtot, *mtot_tmp;
  
  void write_stat(bigint);
  void calc_stat();

};

}

#endif
#endif
