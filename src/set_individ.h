/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMMAND_CLASS

CommandStyle(set/individ,Set_Individ)

#else

#ifndef LMP_SET_INDIVID_H
#define LMP_SET_INDIVID_H

#include "pointers.h"

namespace LAMMPS_NS {

class Set_Individ : protected Pointers {
 public:
  Set_Individ(class LAMMPS *lmp) : Pointers(lmp) {};
  void command(int, char **);

 private:
  int style,type,count;

  double value;

  void set(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Set/individ command before simulation box is defined

The set/individ command cannot be used before a read_data, read_restart,
or create_box command.

E: Set/individ command with no atoms existing

No atoms are yet defined so the set/individ command cannot be used.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Usage of set/individ is not allowed for not atom individual styles

This command should be only used with the parameter individual=1.

E: Invalid value in set command

The value specified for the setting is invalid, likely because it is
too small or too large.

*/
