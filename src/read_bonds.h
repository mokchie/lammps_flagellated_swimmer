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

CommandStyle(read_bonds,ReadBonds)

#else

#ifndef LMP_READ_BONDS_H
#define LMP_READ_BONDS_H

#include "pointers.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace LAMMPS_NS {

class ReadBonds : protected Pointers {
 public:
  ReadBonds(class LAMMPS *);
  void command(int, char **);

 private:
  std::string  line;
  std::ifstream f_bond;
  inline int sbmask(int j) {
    return j >> SBBITS & 3;
  }
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: read_bonds command before simulation box is defined

Self-explanatory.

E: Cannot use read_bonds unless atoms have IDs

This command requires a mapping from global atom IDs to local atoms,
but the atoms that have been defined have no IDs.

E: Cannot use read_bonds with non-molecular system

Self-explanatory.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot find read_bonds group ID

Self-explanatory.

E: Invalid bond type in read_bonds command

Self-explanatory.

E: read_bonds requires a pair style be defined

Self-explanatory.

E: read_bonds max distance > neighbor cutoff

Can only create bonds for atom pairs that will be in neighbor list.

W: read_bonds max distance > minimum neighbor cutoff

This means atom pairs for some atom types may not be in the neighbor
list and thus no bond can be created between them.

E: read_bonds command requires special_bonds 1-2 weights be 0.0

This is so that atom pairs that are already bonded to not appear in
the neighbor list.

E: read_bonds command requires no kspace_style be defined

This is so that atom pairs that are already bonded to not appear
in the neighbor list.

E: New bond exceeded bonds per atom in read_bonds

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

*/
