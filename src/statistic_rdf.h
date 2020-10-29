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

StatisticStyle(rdf,StatisticRdf)

#else


#ifndef LMP_STATISTIC_RDF_H
#define LMP_STATISTIC_RDF_H

#include "statistic.h"

namespace LAMMPS_NS {

class StatisticRdf : public Statistic {
 public:
  StatisticRdf (class LAMMPS *, int, char **);
  ~StatisticRdf ();

 private:
  double *rdf;

  void write_stat(bigint);
  void calc_stat();

  double dr, inv_dr, drsq, cutoff, cutoffsq;
  int rdf_time;
  double Lx, Ly, Lz; // Box lengths
  double Lxhalf, Lyhalf, Lzhalf; //Half box length
  int nall;
  int nall_tmp;
  int nbin;
 

};

}

#endif
#endif
