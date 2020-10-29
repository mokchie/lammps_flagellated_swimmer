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
#include "fix_bond_create_break.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20
#define DELTA 16

/* ---------------------------------------------------------------------- */

FixBondCreateBreak::FixBondCreateBreak(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 9) error->all(FLERR,"Illegal fix bond/create/break command");

  MPI_Comm_rank(world,&me);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/create/break command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  iatomtype = force->inumeric(FLERR,arg[4]);
  jatomtype = force->inumeric(FLERR,arg[5]);
  double cutoff = force->numeric(FLERR,arg[6]);
  btype = force->inumeric(FLERR,arg[7]);

  if (iatomtype < 1 || iatomtype > atom->ntypes ||
      jatomtype < 1 || jatomtype > atom->ntypes)
    error->all(FLERR,"Invalid atom type in fix bond/create/break command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/create/break command");
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/create/break command");

  cutsq = cutoff*cutoff;

  //set type of bonds (simple, dembo or bell)

  int step = 0;

  k0_on = k0_off = spring = sigma = sigma_on = sigma_off = delta = 0.0;
  temp = 1.0;

  if (strcmp(arg[8],"simple") == 0) type_flag = 0;
  else if (strcmp(arg[8],"dembo") == 0) {
    type_flag = 1;
    step = 6;
    if (narg < 9+step) error->all(FLERR,"Illegal fix bond/create/break command");
    spring = force->numeric(FLERR,arg[9]);
    k0_on = force->numeric(FLERR,arg[10]);
    k0_off = force->numeric(FLERR,arg[11]);
    sigma_on = force->numeric(FLERR,arg[12]);
    sigma_off = force->numeric(FLERR,arg[13]);
    temp = force->numeric(FLERR,arg[14]);
  } else if (strcmp(arg[8],"bell") == 0) {
    type_flag = 2;
    step = 6;
    if (narg < 9+step) error->all(FLERR,"Illegal fix bond/create/break command");
    spring = force->numeric(FLERR,arg[9]);
    k0_on = force->numeric(FLERR,arg[10]);
    k0_off = force->numeric(FLERR,arg[11]);
    sigma = force->numeric(FLERR,arg[12]);
    delta = force->numeric(FLERR,arg[13]);
    temp = force->numeric(FLERR,arg[14]);
  } else error->all(FLERR,"Invalid atom type in fix bond/create/break command");


  // optional keywords

  imaxbond = 0;
  inewtype = iatomtype;
  jmaxbond = 0;
  jnewtype = jatomtype;
  fraction_b = 1.0;
  fraction_c = 1.0;
  ieach = jeach = 1;
  int seed = 12345;
  catch_flag = 0;

  int iarg = 9+step;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"iparam") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create/break command");
      imaxbond = force->inumeric(FLERR,arg[iarg+1]);
      inewtype = force->inumeric(FLERR,arg[iarg+2]);
      if (imaxbond < 0) error->all(FLERR,"Illegal fix bond/create/break command");
      if (inewtype < 1 || inewtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/create/break command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"jparam") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create/break command");
      jmaxbond = force->inumeric(FLERR,arg[iarg+1]);
      jnewtype = force->inumeric(FLERR,arg[iarg+2]);
      if (jmaxbond < 0) error->all(FLERR,"Illegal fix bond/create/break command");
      if (jnewtype < 1 || jnewtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/create/break command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"prob/break") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create/break command");
      if (type_flag>0) if(me ==0) error->warning(FLERR,"Command prob/break will be ignored for this style");
      fraction_b = force->numeric(FLERR,arg[iarg+1]);
      if (fraction_b < 0.0 || fraction_b > 1.0)
        error->all(FLERR,"Illegal fix bond/create/break command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"prob/create") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create/break command");
      if (type_flag>0) if(me ==0) error->warning(FLERR,"Command prob/create will be ignored");
      fraction_c = force->numeric(FLERR,arg[iarg+1]);
      if (fraction_c < 0.0 || fraction_c > 1.0)
        error->all(FLERR,"Illegal fix bond/create/break command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"seed") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create/break command");
      seed = force->inumeric(FLERR,arg[iarg+1]);
      if (seed <= 0) error->all(FLERR,"Illegal fix bond/create/break command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"ieach") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create/break command");
      ieach = force->inumeric(FLERR,arg[iarg+1]);
      if (ieach < 1) error->all(FLERR,"Illegal fix bond/create/break command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"jeach") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create/break command");
      jeach = force->inumeric(FLERR,arg[iarg+1]);
      if (jeach < 1) error->all(FLERR,"Illegal fix bond/create/break command");
      iarg += 2;
//     } else if (strcmp(arg[iarg],"catch") == 0) {
//       if (type_flag==0) if(me ==0) error->warning(FLERR,"Command catch will be ignored");
//       catch_flag = 1;
//       iarg += 1;
    } else error->all(FLERR,"Illegal fix bond/create/break command");
  }

  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix bond/create/break with non-molecular systems");
  if (iatomtype == jatomtype &&
      ((imaxbond != jmaxbond) || (inewtype != jnewtype)))
    error->all(FLERR,
               "Inconsistent iparam/jparam values in fix bond/create/break command");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + me);

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bondcount values will be initialized in setup()

  bondcount = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

//   comm_forward = MAX(2,2+atom->maxspecial); old
  comm_forward = 6;
  comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;
  //partner = finalpartner = NULL;
  partner_b =  NULL;
  partner_c =  NULL;
  distsq_b = NULL;
  distsq_c = NULL;
  vec_fraction_b = NULL;
  vec_fraction_c = NULL;
  maxcreate = 0;
  created = NULL;
 
  maxbreak = 0;
  broken = NULL;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  // zero out stats

  createcount = 0;
  createcounttotal = 0;

  breakcount = 0;
  breakcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondCreateBreak::~FixBondCreateBreak()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  delete random;

  // delete locally stored arrays

  memory->destroy(bondcount);
  memory->destroy(partner_b);
  memory->destroy(partner_c);
  memory->destroy(distsq_b);
  memory->destroy(distsq_c);
  memory->destroy(vec_fraction_b);
  memory->destroy(vec_fraction_c);
  memory->destroy(created);
  memory->destroy(broken);
}

/* ---------------------------------------------------------------------- */

int FixBondCreateBreak::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // check cutoff for iatomtype,jatomtype

  if (force->pair == NULL || cutsq > force->pair->cutsq[iatomtype][jatomtype])
    error->all(FLERR,"Fix bond/create/break cutoff is longer than pairwise cutoff");

  // need a half neighbor list, built every Nevery steps
 
  dt = update->dt;
  
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::setup(int vflag)
{
  int i,j,m;

  // compute initial bondcount if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++) bondcount[i] = 0;

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0) 
            error->one(FLERR,"Fix bond/create/break needs ghost atoms "
                       "from further away");
          bondcount[m]++;
        }
      }
    }

  // if newton_bond is set, need to sum bondcount

  commflag = 1;
  if (newton_bond) comm->reverse_comm_fix(this,1);
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible,i1,i2,type_bond, jb, jc;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check
  // needs to be <= test b/c neighbor list could have been re-built in
  //   same timestep as last post_integrate() call, but afterwards
  // NOTE: no longer think is needed, due to error tests on atom->map()
  // NOTE: if delete, can also delete lastcheck and check_ghosts()

  //if (lastcheck <= neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // forward comm of bondcount, so ghosts have it

  commflag = 1;
  comm->forward_comm_fix(this,1);

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {

    memory->destroy(partner_b);
    memory->destroy(partner_c);

    memory->destroy(distsq_b);
    memory->destroy(distsq_c);

    memory->destroy(vec_fraction_b);
    memory->destroy(vec_fraction_c);

    nmax = atom->nmax;
    memory->create(partner_b,nmax,"bond/create/break:partner_b");
    memory->create(partner_c,nmax,"bond/create/break:partner_c");

    memory->create(distsq_b,nmax,"bond/create/break:distsq_b");
    memory->create(distsq_c,nmax,"bond/create/break:distsq_c");

    memory->create(vec_fraction_b,nmax,"bond/create/break:vec_fraction_b");
    memory->create(vec_fraction_c,nmax,"bond/create/break:vec_fraction_c");

    probability_b = distsq_b;
    probability_c = distsq_c;
    
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner_b[i] = 0;
    partner_c[i] = 0;
    vec_fraction_b[i] = 0.0;
    vec_fraction_c[i] = 0.0;
    distsq_b[i] = 0.0;
    distsq_c[i] = BIG;
  }

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  //Break:
  // loop over bond list
  // setup possible partner list of bonds to break

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type_bond = bondlist[n][2];
    if (!(mask[i1] & groupbit)) continue;
    if (!(mask[i2] & groupbit)) continue;
    if (type_bond != btype) continue;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    rsq = delx*delx + dely*dely + delz*delz;

    if (rsq <= cutsq) continue;

    if (rsq > distsq_b[i1]) {
      partner_b[i1] = tag[i2];
      distsq_b[i1] = rsq;
    }
    if (rsq > distsq_b[i2]) {
      partner_b[i2] = tag[i1];
      distsq_b[i2] = rsq;
    }
  }

  commflag = 2;
  if (force->newton_bond)   comm->reverse_comm_fix(this);

  //Create:
  // loop over neighbors of my atoms
  // each atom sets one closest eligible partner atom ID to bond with

  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int *type = atom->type;
  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int **bond_type = atom->bond_type;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      jtype = type[j];

      // test if number of bonds is allowed
      // test if density of atom i or j should be reduced 

      possible = 0;
      if (itype == iatomtype && jtype == jatomtype) {
        if ((imaxbond == 0 || bondcount[i] < imaxbond) &&
            (jmaxbond == 0 || bondcount[j] < jmaxbond)) 
          if (tag[i]%ieach == 0 && tag[j]%jeach == 0) possible = 1;
      } else if (itype == jatomtype && jtype == iatomtype) {
        if ((jmaxbond == 0 || bondcount[i] < jmaxbond) &&
            (imaxbond == 0 || bondcount[j] < imaxbond))
          if (tag[i]%jeach == 0 && tag[j]%ieach == 0) possible = 1;
      }
      if (!possible) continue;

      // do not allow a duplicate bond to be created
      // check bonds

      for(m = 0; m<num_bond[i];m++)
	if(bond_atom[i][m] == tag[j]) possible = 0;
      if (!possible) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq >= cutsq) continue;

      if (rsq < distsq_c[i]) {
        partner_c[i] = tag[j];
        distsq_c[i] = rsq;
      }
      if (rsq < distsq_c[j]) {
        partner_c[j] = tag[i];
        distsq_c[j] = rsq;
      }
    }
  }

  // reverse comm of distsq and partner
  // not needed if newton_pair off since I,J pair was seen by both procs
  commflag = 3;
  if (force->newton_pair) comm->reverse_comm_fix(this);

  // each atom now knows its winning partner
  // simple: for prob check, generate random value for each atom with a bond partner
  // others: for prob check, calculate probability and generate random value for each atom with a bond partner
  // forward comm of partner, random value, and probability, so ghosts have it

  if (type_flag == 0) {
    if (fraction_b < 1.0) {
      for (i = 0; i < nlocal; i++){
        if (partner_b[i]) {
          probability_b[i] = random->uniform();
        }
        if (partner_c[i]) {
          probability_c[i] = random->uniform();
        } 
      }
    }
    commflag = 2;
    comm->forward_comm_fix(this,4);

  } else if(type_flag == 1) {
    for (i = 0; i < nlocal; i++){
      if (partner_b[i]) {
        vec_fraction_b[i] = dembo_break(i, partner_b[i], sqrt(distsq_b[i]));
        probability_b[i] = random->uniform();

      }
      if (partner_c[i]) {
        vec_fraction_c[i] = dembo_create(sqrt(distsq_c[i]));
        probability_c[i] = random->uniform();
      }
    }

    commflag = 3;
    comm->forward_comm_fix(this,6);

  } else if(type_flag == 2) {
    for (i = 0; i < nlocal; i++){
      if (partner_b[i]) {
        vec_fraction_b[i] = bell_break(i, partner_b[i], sqrt(distsq_b[i]));
        probability_b[i] = random->uniform();
      }
      if (partner_c[i]) {
        vec_fraction_c[i] = bell_create(sqrt(distsq_c[i]));
        probability_c[i] = random->uniform();
      }
    }
    commflag = 3;
    comm->forward_comm_fix(this,6);
  }

  // create bonds for atoms I own
  // only if both atoms list each other as winning bond partner
  //   and probability constraint is satisfied
  // if other atom is owned by another proc, it should do same thing

  int newton_bond = force->newton_bond;
  int cont_flag_b,  cont_flag_c;

  ncreate = 0;
  nbreak = 0;
  for (i = 0; i < nlocal; i++) {
    cont_flag_b = 0;
    cont_flag_c = 0;

    //Check if possible partner exists
    if (partner_b[i] == 0) cont_flag_b = 1;
    if (partner_c[i] == 0) cont_flag_c = 1;

    if(cont_flag_b == 1 && cont_flag_c == 1) continue;

    //If possible partner exists, check if both bond atoms chose each other as partner
    if(!cont_flag_b) {
      jb = atom->map(partner_b[i]);
      if (partner_b[jb] != tag[i]) cont_flag_b = 1;
    }

    if(!cont_flag_c) {
      jc = atom->map(partner_c[i]);
      if (partner_c[jc] != tag[i]) cont_flag_c = 1;
    }

    if(cont_flag_b == 1 && cont_flag_c == 1) continue;

   // If possible partner exists and both bond atoms chose each other as partner:
   // compare probability to RN for atom with smallest ID
    if (type_flag == 0) {
      if(!cont_flag_b) {
        if (fraction_b < 1.0) {
          if (tag[i] < tag[jb]) {
            if (probability_b[i] >= fraction_b) cont_flag_b = 1;
          } else {
            if (probability_b[jb] >= fraction_b) cont_flag_b = 1;
          }
        }
      }
      if(!cont_flag_c) {
        if (fraction_c < 1.0) {
          if (tag[i] < tag[jc]) {
            if (probability_c[i] >= fraction_c) cont_flag_c = 1;
          } else {
            if (probability_c[jc] >= fraction_c) cont_flag_c = 1;
          }
        }
      }
    } else {
      if(!cont_flag_b) {
        if (tag[i] < tag[jb]) {
          if (probability_b[i] >= vec_fraction_b[i]) cont_flag_b = 1;
        } else {
          if (probability_b[jb] >= vec_fraction_b[jb]) cont_flag_b = 1;
        }
      }
      if(!cont_flag_c) {
        if (tag[i] < tag[jc]) {
          if (probability_c[i] >= vec_fraction_c[i]) cont_flag_c = 1;
        } else {
          if (probability_c[jc] >= vec_fraction_c[jc]) cont_flag_c = 1;
        }
      }
    }

    if(cont_flag_b == 1 && cont_flag_c == 1) continue;

    //If eveything applies break or create the bond

    //Break:
    // delete bond from atom I if I stores it
    // atom J will also do this
    if(!cont_flag_b){
      for (m = 0; m < num_bond[i]; m++) {
        if (bond_atom[i][m] == partner_b[i]) {
          for (k = m; k < num_bond[i]-1; k++) {
            bond_atom[i][k] = bond_atom[i][k+1];
            bond_type[i][k] = bond_type[i][k+1];
          }
          num_bond[i]--;
          break;
        }
      }

      // decrement bondcount, convert atom to old type if limit is not reached anymore
      // If iatomtype and jatomtype are different but inewtype and jnewtype are the same this does not work
      // atom J will also do this, whatever proc it is on

      bondcount[i]--;
      if (type[i] == inewtype) {
        if (bondcount[i] == imaxbond-1) type[i] = iatomtype;
      } else {
        if (bondcount[i] == jmaxbond-1) type[i] = jatomtype;
      }

      if (tag[i] < tag[jb]) nbreak++;
    }

    //Create
    if(!cont_flag_c) {

      // if newton_bond is set, only store with I or J
      // if not newton_bond, store bond with both I and J
      // atom J will also do this consistently, whatever proc it is on

      if (!newton_bond || tag[i] < tag[jc]) {
        if (num_bond[i] == atom->bond_per_atom)
          error->one(FLERR,"New bond exceeded bonds per atom in fix bond/create/break");
        bond_type[i][num_bond[i]] = btype;
        bond_atom[i][num_bond[i]] = tag[jc];
        num_bond[i]++;
      }

      // increment bondcount, convert atom to new type if limit reached
      // atom J will also do this, whatever proc it is on

      bondcount[i]++;
      if (type[i] == iatomtype) {
        if (bondcount[i] == imaxbond) type[i] = inewtype;
      } else {
        if (bondcount[i] == jmaxbond) type[i] = jnewtype;
      }

      // store final created bond partners and count the created bond once

      if (tag[i] < tag[jc]) ncreate++;
    }
  }

  // tally stats
  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;
  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcount;
  atom->nbonds += createcount;

  // trigger reneighboring if any bonds were formed
  // this insures neigh lists will immediately reflect the topology changes
  // done if any bonds created

  if (createcount || breakcount) next_reneighbor = update->ntimestep;


  // DEBUG
  //print_bb();
}

/* ---------------------------------------------------------------------- */

double FixBondCreateBreak::dembo_create(double dist)
{
  
  double k_on;
  double probability;

  k_on = k0_on*exp(- 0.5*sigma_on* (dist - spring)*(dist - spring) / temp );
  probability = 1.0 - exp(-k_on*dt);
//   printf("%g \n", probability);

  return probability;
}

/* ---------------------------------------------------------------------- */


double FixBondCreateBreak::dembo_break(int i, int j, double dist)
{
  
  double k_off;
  double probability;

  k_off = k0_off*exp(0.5*sigma_off* (dist - spring)*(dist - spring) / temp );

//   if(catch_flag) {
//     double fbond = 0.0;
//     double eng = 0.0;
//     Bond *bond = force->bond;
//     eng = bond->single(btype,dist*dist,i,j,fbond); 
//     double r0 = bond->equilibrium_distance(btype);
// //fbond*dist
//     probability = 1.0 - exp(- ( k_off*dt - eng/temp));
//     
//     if(probability<0.0)
//     	if(me ==0) error->warning(FLERR,"Negative probability. Choose other parameters");
//    printf("%g %g %g %g %g\n", dist, k_off*dt, 1.0 - exp(-k_off*dt), eng/temp, probability);
// 
//   }
//   else
    probability = 1.0 - exp(-k_off*dt);

  return probability;
}

/* ---------------------------------------------------------------------- */


double FixBondCreateBreak::bell_create(double dist)
{
  
  double k_on;
  double probability;

  k_on = k0_on*exp(sigma* fabs( dist - spring )*(delta - 0.5*fabs( dist - spring )) / temp );
  probability = 1.0 - exp(-k_on*dt);

  return probability;
}

/* ---------------------------------------------------------------------- */


double FixBondCreateBreak::bell_break(int i, int j, double dist)
{
  
  double k_off;
  double probability;

  k_off = k0_off*exp(sigma* delta* fabs( dist - spring )/ temp );
  
//   if(catch_flag) {
//     double fbond = 0.0;
//     double eng = 0.0;
//     Bond *bond = force->bond;
//     eng = bond->single(btype,dist*dist,i,j,fbond); 
//     double r0 = bond->equilibrium_distance(btype);
// 
//     probability = 1.0 - exp(- ( k_off*dt + eng/temp));
//   }
//   else
    probability = 1.0 - exp(-k_off*dt);

  return probability;	
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly created
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is added
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixBondCreateBreak::check_ghosts()
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
    error->all(FLERR,"Fix bond/create/break needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondCreateBreak::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m,ns;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(bondcount[j]).d;
    }
    return m;
  }

  if (commflag == 2) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(partner_b[j]).d;
      buf[m++] = ubuf(partner_c[j]).d;
      buf[m++] = probability_b[j];
      buf[m++] = probability_c[j];
    }
    return m;
  }

//  if commflag == 3
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(partner_b[j]).d;
    buf[m++] = ubuf(partner_c[j]).d;
    buf[m++] = vec_fraction_b[j];
    buf[m++] = vec_fraction_c[j];
    buf[m++] = probability_b[j];
    buf[m++] = probability_c[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++)
      bondcount[i] = (int) ubuf(buf[m++]).i;

  } else if (commflag == 2) {
    for (i = first; i < last; i++) {
      partner_b[i] = (tagint) ubuf(buf[m++]).i;
      partner_c[i] = (tagint) ubuf(buf[m++]).i;
      probability_b[i] = buf[m++];
      probability_c[i] = buf[m++];
    }
  } else if (commflag == 3) {
    for (i = first; i < last; i++) {
      partner_b[i] = (tagint) ubuf(buf[m++]).i;
      partner_c[i] = (tagint) ubuf(buf[m++]).i;
      vec_fraction_b[i] = buf[m++];
      vec_fraction_c[i] = buf[m++];
      probability_b[i] = buf[m++];
      probability_c[i] = buf[m++];
    }
  } 

}

/* ---------------------------------------------------------------------- */

int FixBondCreateBreak::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++)
      buf[m++] = ubuf(bondcount[i]).d;
    return m;
  } 
  
  if (commflag == 2) {
    for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner_b[i]).d;
    buf[m++] = distsq_b[i];
    }
    return m;
  }

  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner_c[i]).d;
    buf[m++] = distsq_c[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      bondcount[j] += (int) ubuf(buf[m++]).i;
    }
  } else if(commflag == 2) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (buf[m+1] > distsq_b[j]) {
        partner_b[j] = (tagint) ubuf(buf[m++]).i;
        distsq_b[j] = buf[m++];
      } else m += 2;
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (buf[m+1] < distsq_c[j]) {
        partner_c[j] = (tagint) ubuf(buf[m++]).i;
        distsq_c[j] = buf[m++];
      } else m += 2;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixBondCreateBreak::grow_arrays(int nmax)
{
  memory->grow(bondcount,nmax,"bond/create/break:bondcount");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixBondCreateBreak::copy_arrays(int i, int j, int delflag)
{
  bondcount[j] = bondcount[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixBondCreateBreak::pack_exchange(int i, double *buf)
{
  buf[0] = bondcount[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixBondCreateBreak::unpack_exchange(int nlocal, double *buf)
{
  bondcount[nlocal] = static_cast<int> (buf[0]);
  return 1;
}

/* ---------------------------------------------------------------------- */
//Her ist mir nocn unklar wofuer dieseFunktion gut ist
double FixBondCreateBreak::compute_vector(int n)
{
  printf("In compute_vector\n");
  if (n == 0) return (double) createcount;
  return (double) createcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondCreateBreak::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = nmax * sizeof(int);
  bytes += 2*nmax * sizeof(tagint);
  bytes += 2*nmax * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::print_bb()
{
  for (int i = 0; i < atom->nlocal; i++) {
    printf("TAG " TAGINT_FORMAT ": %d nbonds: ",atom->tag[i],atom->num_bond[i]);
    for (int j = 0; j < atom->num_bond[i]; j++) {
      printf(" " TAGINT_FORMAT,atom->bond_atom[i][j]);
    }
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixBondCreateBreak::print_copy(const char *str, tagint m, 
                              int n1, int n2, int n3, int *v)
{
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}
