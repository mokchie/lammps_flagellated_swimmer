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

#ifndef LMP_STATISTIC_H
#define LMP_STATISTIC_H

#include <cstdio>
#include "pointers.h"

namespace LAMMPS_NS {

class Statistic : protected Pointers {
 public:
  char *style;
  int nx, ny, nz, groupbit, cyl_ind;
  int st_start, step_each, dump_each, num_step;
  double xx, yy, zz;
  double xs, ys, zs;
  double xlo, ylo, zlo, xhi, yhi, zhi;
  double dxlo, dylo, dzlo, dxhi, dyhi, dzhi;
  double dxs, dys, dzs;
  double dxpm1, dypm1, dzpm1;
  double dx, dy, dz;
  int xper, yper, zper;
  int is, js, ks, jv;
  char fname[FILENAME_MAX];
  
  Statistic(class LAMMPS *, int, char **);
  virtual ~Statistic();
  void init();
  int map_index(double, double, double);
  virtual void write_stat(bigint)=0;
  virtual void calc_stat()=0;
  virtual void virial1(int){};
  virtual void virial2(int, double[], int){};
  virtual void virial3(int, int, double[]){};
  virtual void virial4(int, int, int, double[]){};
  virtual void virial5(int, int, int, int, double[]){};
  virtual void virial6(int, double[], double[]){};
};
}

#endif

/* ERROR/WARNING messages:

E: Illegal statistic command

Self-explanatory. Check input script

E: Illegal division for statistics

Check in input script that the number of bins for every direction is >=1  

E: Illegal coordinates for statistics

Check the coodinates in which region the statistic should be calculated

*/
