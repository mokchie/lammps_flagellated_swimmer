/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cmath>
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include "fix_remove_string.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTA 16

/* ---------------------------------------------------------------------- */

FixRemoveString::FixRemoveString(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  partner(NULL), finalpartner(NULL), broken(NULL), copy(NULL), totalnum_bond(NULL)
{
  if(atom->individual<1) error->all(FLERR,"Individual must be >0 for fix remove/string command");
  if (narg < 7) error->all(FLERR,"Illegal fix remove/string command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix remove/string command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  btype = force->inumeric(FLERR,arg[4]);
  bnc1 = force->numeric(FLERR,arg[5]);
  bnc2 = force->numeric(FLERR,arg[6]);  
  if(!force->newton) {
    bnc1 = 2*bnc1;
    bnc2 = 2*bnc2;
  }
  
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix remove/string command");
  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix remove/string with non-molecular systems");

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

  comm_forward = MAX(2,2+atom->maxspecial);
  comm_reverse = 1;

  // allocate arrays local to this fix

  nmax = 0;

  maxbreak = 0;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special_one() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  // zero out stats

  breakcount = 0;
  breakcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixRemoveString::~FixRemoveString()
{

  // delete locally stored arrays

  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(broken);
  memory->destroy(totalnum_bond);
  delete [] copy;
}

/* ---------------------------------------------------------------------- */

int FixRemoveString::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRemoveString::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // enable angle/dihedral/improper breaking if any defined

  if (atom->nangles) angleflag = 1;
  else angleflag = 0;
  if (atom->ndihedrals) dihedralflag = 1;
  else dihedralflag = 0;
  if (atom->nimpropers) improperflag = 1;
  else improperflag = 0;

  if (force->improper) {
    if (force->improper_match("class2") || force->improper_match("ring"))
      error->all(FLERR,"Cannot yet use fix remove/string with this "
                 "improper style");
  }

  lastcheck = -1;

  // DEBUG
  //print_bb();
}

/* ---------------------------------------------------------------------- */

void FixRemoveString::post_integrate()
{
  int i,j,k,m,n,i1,i2,n1,n3,type;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check

  if (lastcheck < neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(totalnum_bond);
    nmax = atom->nmax;
    memory->create(partner,nmax,"remove/string:partner");
    memory->create(finalpartner,nmax,"remove/string:finalpartner");
    memory->create(totalnum_bond,nmax,"remove/string:totalnum_bond");
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    totalnum_bond[i] = 0;
  }

  // loop over bond list
  // setup possible partner list of bonds to break

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double *bondlist_length = neighbor->bondlist_length;
  double **bond_length = atom->bond_length;


  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    if (!(mask[i1] & groupbit)) continue;
    if (!(mask[i2] & groupbit)) continue;
    if (type != btype) continue;
    totalnum_bond[i1]++;
    totalnum_bond[i2]++;
  }
  commflag = 0;
  //reverse and forward bond number info
  comm->reverse_comm_fix(this,1);
  comm->forward_comm_fix(this,1);

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    if (!(mask[i1] & groupbit)) continue;
    if (!(mask[i2] & groupbit)) continue;
    if (type != btype) continue;
    if(totalnum_bond[i1]<=bnc2 && totalnum_bond[i2]<=bnc2){
      partner[i1] = tag[i2];
      partner[i2] = tag[i1];
    }      
    if(totalnum_bond[i1]<=bnc1 || totalnum_bond[i2]<=bnc1){
      partner[i1] = tag[i2];
      partner[i2] = tag[i1];
    }  
  }
  // reverse comm of partner info
  commflag = 1;
  //if (force->newton_bond) comm->reverse_comm_fix(this);
  //no need to reverse comm the partner info

  // each atom now knows its winning partner
  // forward comm of partner, so ghosts have it
  comm->forward_comm_fix(this,1);

  //for (i=0; i<nlocal; i++)
  //  if(distsq[i]>0)
  //    printf("distsq[%d]=%f\n",i,distsq[i]);
  // break bonds
  // if both atoms list each other as winning bond partner
  // and probability constraint is satisfied

  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  nbreak = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;
    // delete bond from atom I if I stores it
    // atom J will also do this
    for (m = 0; m < num_bond[i]; m++) {
      if (bond_atom[i][m] == partner[i]) {
        for (k = m; k < num_bond[i]-1; k++) {
          bond_atom[i][k] = bond_atom[i][k+1];
          bond_type[i][k] = bond_type[i][k+1];
          bond_length[i][k] = bond_length[i][k+1];
        }
        num_bond[i]--;
        break;
      }
    }
    // remove J from special bond list for atom I
    // atom J will also do this, whatever proc it is on
    slist = special[i];
    n1 = nspecial[i][0];
    for (m = 0; m < n1; m++)
      if (slist[m] == partner[i]) break;
    n3 = nspecial[i][2];
    for (; m < n3-1; m++) slist[m] = slist[m+1];
    nspecial[i][0]--;
    nspecial[i][1]--;
    nspecial[i][2]--;
    // store final broken bond partners and count the broken bond once

    finalpartner[i] = tag[j];
    finalpartner[j] = tag[i];
    if (tag[i] < tag[j]) nbreak++;
  }
  // tally stats

  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;

  // trigger reneighboring if any bonds were broken
  // this insures neigh lists will immediately reflect the topology changes
  // done if no bonds broken

  if (breakcount) next_reneighbor = update->ntimestep;
  if (!breakcount) return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds

  commflag = 2;
  comm->forward_comm_fix(this);

  // create list of broken bonds that influence my owned atoms
  //   even if between owned-ghost or ghost-ghost atoms
  // finalpartner is now set for owned and ghost atoms so loop over nall
  // OK if duplicates in broken list due to ghosts duplicating owned atoms
  // check J < 0 to insure a broken bond to unknown atom is included
  //   i.e. bond partner outside of cutoff length

  nbreak = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (nbreak == maxbreak) {
        maxbreak += DELTA;
        memory->grow(broken,maxbreak,2,"remove/string:broken");
      }
      broken[nbreak][0] = tag[i];
      broken[nbreak][1] = finalpartner[i];
      nbreak++;
    }
  }

  // update special neigh lists of all atoms affected by any broken bond
  // also remove angles/dihedrals/impropers broken by broken bonds

  update_topology();

  // DEBUG
  // print_bb();
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly deleted
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is deleted
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixRemoveString::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall)
    error->all(FLERR,"Fix remove/string needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
   double loop over my atoms and broken bonds
   influenced = 1 if atom's topology is affected by any broken bond
     yes if is one of 2 atoms in bond
     yes if both atom IDs appear in atom's special list
     else no
   if influenced:
     check for angles/dihedrals/impropers to break due to specific broken bonds
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
------------------------------------------------------------------------- */

void FixRemoveString::update_topology()
{
  int i,j,k,n,influence,influenced,found;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  nangles = 0;
  ndihedrals = 0;
  nimpropers = 0;

  //printf("NBREAK %d: ",nbreak);
  //for (i = 0; i < nbreak; i++)
  //  printf(" %d %d,",broken[i][0],broken[i][1]);
  //printf("\n");

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < nbreak; j++) {
      id1 = broken[j][0];
      id2 = broken[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2) influence = 1;
      else {
        n = nspecial[i][2];
        found = 0;
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2) found++;
        if (found == 2) influence = 1;
      }
      if (!influence) continue;
      influenced = 1;

      if (angleflag) break_angles(i,id1,id2);
      if (dihedralflag) break_dihedrals(i,id1,id2);
      if (improperflag) break_impropers(i,id1,id2);
    }

    if (influenced) rebuild_special_one(i);
  }

  int newton_bond = force->newton_bond;

  int all;
  if (angleflag) {
    MPI_Allreduce(&nangles,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 3;
    atom->nangles -= all;
  }
  if (dihedralflag) {
    MPI_Allreduce(&ndihedrals,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->ndihedrals -= all;
  }
  if (improperflag) {
    MPI_Allreduce(&nimpropers,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->nimpropers -= all;
  }
}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs
------------------------------------------------------------------------- */

void FixRemoveString::rebuild_special_one(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ----------------------------------------------------------------------
   break any angles owned by atom M that include atom IDs 1 and 2
   angle is broken if ID1-ID2 is one of 2 bonds in angle (I-J,J-K)
------------------------------------------------------------------------- */

void FixRemoveString::break_angles(int m, tagint id1, tagint id2)
{
  int j,found;

  int num_angle = atom->num_angle[m];
  int *angle_type = atom->angle_type[m];
  tagint *angle_atom1 = atom->angle_atom1[m];
  tagint *angle_atom2 = atom->angle_atom2[m];
  tagint *angle_atom3 = atom->angle_atom3[m];

  int i = 0;
  while (i < num_angle) {
    found = 0;
    if (angle_atom1[i] == id1 && angle_atom2[i] == id2) found = 1;
    else if (angle_atom2[i] == id1 && angle_atom3[i] == id2) found = 1;
    else if (angle_atom1[i] == id2 && angle_atom2[i] == id1) found = 1;
    else if (angle_atom2[i] == id2 && angle_atom3[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_angle-1; j++) {
        angle_type[j] = angle_type[j+1];
        angle_atom1[j] = angle_atom1[j+1];
        angle_atom2[j] = angle_atom2[j+1];
        angle_atom3[j] = angle_atom3[j+1];
      }
      num_angle--;
      nangles++;
    }
  }

  atom->num_angle[m] = num_angle;
}

/* ----------------------------------------------------------------------
   break any dihedrals owned by atom M that include atom IDs 1 and 2
   dihedral is broken if ID1-ID2 is one of 3 bonds in dihedral (I-J,J-K.K-L)
------------------------------------------------------------------------- */

void FixRemoveString::break_dihedrals(int m, tagint id1, tagint id2)
{
  int j,found;

  int num_dihedral = atom->num_dihedral[m];
  int *dihedral_type = atom->dihedral_type[m];
  tagint *dihedral_atom1 = atom->dihedral_atom1[m];
  tagint *dihedral_atom2 = atom->dihedral_atom2[m];
  tagint *dihedral_atom3 = atom->dihedral_atom3[m];
  tagint *dihedral_atom4 = atom->dihedral_atom4[m];

  int i = 0;
  while (i < num_dihedral) {
    found = 0;
    if (dihedral_atom1[i] == id1 && dihedral_atom2[i] == id2) found = 1;
    else if (dihedral_atom2[i] == id1 && dihedral_atom3[i] == id2) found = 1;
    else if (dihedral_atom3[i] == id1 && dihedral_atom4[i] == id2) found = 1;
    else if (dihedral_atom1[i] == id2 && dihedral_atom2[i] == id1) found = 1;
    else if (dihedral_atom2[i] == id2 && dihedral_atom3[i] == id1) found = 1;
    else if (dihedral_atom3[i] == id2 && dihedral_atom4[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_dihedral-1; j++) {
        dihedral_type[j] = dihedral_type[j+1];
        dihedral_atom1[j] = dihedral_atom1[j+1];
        dihedral_atom2[j] = dihedral_atom2[j+1];
        dihedral_atom3[j] = dihedral_atom3[j+1];
        dihedral_atom4[j] = dihedral_atom4[j+1];
      }
      num_dihedral--;
      ndihedrals++;
    }
  }

  atom->num_dihedral[m] = num_dihedral;
}

/* ----------------------------------------------------------------------
   break any impropers owned by atom M that include atom IDs 1 and 2
   improper is broken if ID1-ID2 is one of 3 bonds in improper (I-J,I-K,I-L)
------------------------------------------------------------------------- */

void FixRemoveString::break_impropers(int m, tagint id1, tagint id2)
{
  int j,found;

  int num_improper = atom->num_improper[m];
  int *improper_type = atom->improper_type[m];
  tagint *improper_atom1 = atom->improper_atom1[m];
  tagint *improper_atom2 = atom->improper_atom2[m];
  tagint *improper_atom3 = atom->improper_atom3[m];
  tagint *improper_atom4 = atom->improper_atom4[m];

  int i = 0;
  while (i < num_improper) {
    found = 0;
    if (improper_atom1[i] == id1 && improper_atom2[i] == id2) found = 1;
    else if (improper_atom1[i] == id1 && improper_atom3[i] == id2) found = 1;
    else if (improper_atom1[i] == id1 && improper_atom4[i] == id2) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom2[i] == id1) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom3[i] == id1) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom4[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_improper-1; j++) {
        improper_type[j] = improper_type[j+1];
        improper_atom1[j] = improper_atom1[j+1];
        improper_atom2[j] = improper_atom2[j+1];
        improper_atom3[j] = improper_atom3[j+1];
        improper_atom4[j] = improper_atom4[j+1];
      }
      num_improper--;
      nimpropers++;
    }
  }

  atom->num_improper[m] = num_improper;
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates
------------------------------------------------------------------------- */

int FixRemoveString::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}

/* ---------------------------------------------------------------------- */

void FixRemoveString::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixRemoveString::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,ns;
  if (commflag == 0) {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(totalnum_bond[j]).d;
    }
    return m;
  }
  if (commflag == 1) {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(partner[j]).d;
    }
    return m;
  }

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(finalpartner[j]).d;
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRemoveString::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;
  
  if (commflag == 0) {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      totalnum_bond[i] = ubuf(buf[m++]).i;
    }

  }   
  else if (commflag == 1) {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      partner[i] = (tagint) ubuf(buf[m++]).i;
    }

  } else {

    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      finalpartner[i] = (tagint) ubuf(buf[m++]).i;
      ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (j = 0; j < ns; j++)
        special[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixRemoveString::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  if(commflag==0){
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      buf[m++] = ubuf(totalnum_bond[i]).d;
    }
    return m;    
  }
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRemoveString::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  tagint tt;
  if(commflag==0){
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      totalnum_bond[j] += ubuf(buf[m++]).i;
    }
  } 
  else {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      tt = (tagint) ubuf(buf[m++]).i;
      if (tt < partner[j]) {
        partner[j] = tt;
      }
    }
  }
}


/* ---------------------------------------------------------------------- */

void FixRemoveString::print_bb()
{
  for (int i = 0; i < atom->nlocal; i++) {
    printf("TAG " TAGINT_FORMAT ": %d nbonds: ",atom->tag[i],atom->num_bond[i]);
    for (int j = 0; j < atom->num_bond[i]; j++) {
      printf(" %d",atom->bond_atom[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d nangles: ",atom->tag[i],atom->num_angle[i]);
    for (int j = 0; j < atom->num_angle[i]; j++) {
      printf(" %d %d %d,",atom->angle_atom1[i][j],
             atom->angle_atom2[i][j],atom->angle_atom3[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d ndihedrals: ",atom->tag[i],atom->num_dihedral[i]);
    for (int j = 0; j < atom->num_dihedral[i]; j++) {
      printf(" %d %d %d %d,",atom->dihedral_atom1[i][j],
             atom->dihedral_atom2[i][j],atom->dihedral_atom3[i][j],
             atom->dihedral_atom4[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d %d %d nspecial: ",atom->tag[i],
           atom->nspecial[i][0],atom->nspecial[i][1],atom->nspecial[i][2]);
    for (int j = 0; j < atom->nspecial[i][2]; j++) {
      printf(" %d",atom->special[i][j]);
    }
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixRemoveString::print_copy(const char *str, tagint m,
                              int n1, int n2, int n3, int *v)
{
  printf("%s %i: %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}

/* ---------------------------------------------------------------------- */

double FixRemoveString::compute_vector(int n)
{
  if (n == 0) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixRemoveString::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(tagint);
  bytes += nmax * sizeof(double);
  return bytes;
}
