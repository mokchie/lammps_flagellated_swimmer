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

/* ----------------------------------------------------------------------
   Contributing author: Brooke Huisman (UMich), Masoud Hoore (FZJ), Kathrin Mueller (FZJ), Dmitry Fedosov (FZJ)
------------------------------------------------------------------------- */

#include <cmath>
#include "mpi.h"
#include <cstring>
#include <cstdlib>
#include "fix_break_pol.h"
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
/* ---------------------------------------------------------------------- 
Syntax: fix	ID group-ID	break/pol	Nevery 	type	b_type	k_rate	keyword key_value
arg[]		0	1	2		3       4	5	6	7

keywords:
seed cleave
keyvalues:
seed: seed
cleave: yes or no (if no: file_name)
-------------------------------------------------------------------------*/
FixBreakPol::FixBreakPol(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix break/pol command");

  //MPI_Comm_rank(world,&me);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix break/pol command: Nevery is smaller than zero");
  if (neighbor->delay != 0)
    if (nevery % neighbor->delay != 0)
      error->all(FLERR,"Nevery in fix break/pol should be devisible by neighbor->delay.");
  else
    if (nevery % neighbor->every != 0)
      error->all(FLERR,"Nevery in fix break/pol should be devisible by neighbor->every.");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 0;
  size_vector = 2;
  global_freq = nevery;
  extvector = 0;

  atype = force->inumeric(FLERR,arg[4]);
  if (atype < 1 || atype > atom->ntypes)
    error->all(FLERR,"Invalid atom type in fix break/pol command: atom types are wrong");

  btype = force->inumeric(FLERR,arg[5]);
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix break/pol command: bond type is wrong");

  krate = force->numeric(FLERR,arg[6]);
  if (krate < 0.0)
    error->all(FLERR,"Illegal fix break/pol command: k_rate is negative");

  // optional keywords

  int seed = 12345;
  cleave_ind = 1;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"seed") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix break/pol command");
      seed = force->inumeric(FLERR,arg[iarg+1]);
      if (seed <= 0) error->all(FLERR,"Illegal fix break/pol command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"cleave") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix fix break/pol command");
      if (strcmp(arg[iarg+1],"yes") == 0) {
        cleave_ind = 1;
        iarg += 2;
      } else if (strcmp(arg[iarg+1],"no") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix fix break/pol command");
        sprintf(fname,arg[iarg+2]);
        cleave_ind = 0;
        iarg += 3;
      }
      else error->all(FLERR,"cleave keyvalues must be yes or no.");
    } else error->all(FLERR,"Illegal fix catch/bond command");
  }

  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix break/pol with non-molecular systems");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + comm->me);

  // perform initial allocation of atom-based arrays
  // register with Atom class

  atom->add_callback(0);

  if (!comm->me && cleave_ind == 0) {
      FILE* output;
      output=fopen(fname,"w");
      fprintf(output," --- Cleaved bonds dump \n");
      fprintf(output,"ITEM: timestep mol atom1 \n");
      fclose(output);
    }

}

/* ---------------------------------------------------------------------- */

FixBreakPol::~FixBreakPol()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id,0);
  delete random;

  // delete locally stored arrays
}

/* ---------------------------------------------------------------------- */

int FixBreakPol::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBreakPol::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  vec_fraction = 1.0 - exp(-krate*nevery*update->dt);

  // need a half neighbor list, built every Nevery steps
  
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixBreakPol::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBreakPol::post_integrate()
{
  if (update->ntimestep % nevery) return;

  int i,j,k,i2;

  int n_b = 0;
  int *rcounts = NULL;
  int n_b_all = 0;
  tagint *data_b = NULL;
  int n_b_max = 1000;
  int *displs = NULL;

  memory->create(data_b,n_b_max,"fix_break_pol:data_b");
  memory->create(rcounts,comm->nprocs,"fix_break_pol:rcounts");
  memory->create(displs,comm->nprocs,"fix_break_pol:displs");

  comm->forward_comm();

  int nlocal = atom->nlocal;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  tagint n_mol_max = atom->n_mol_max;
  int n_mol_types = atom->n_mol_types;
  int *mol_type = atom->mol_type;
  tagint *mol_size = atom->mol_size;
  tagint *molecule = atom->molecule;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  double **bond_length = atom->bond_length;

  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (type[i] != atype) continue;
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] != btype) continue;
      i2 = atom->map(bond_atom[i][j]);
      if (i2 < 0)
        error->one(FLERR,"Fix break/pol needs ghost atoms from further away");
      if (!(mask[i2] & groupbit)) continue;
      if (molecule[i] != molecule[i2]) continue;
      if (type[i2] != atype) continue;
      if (!force->newton_bond && tag[i] > tag[i2]) continue;
      if (random->uniform() >= vec_fraction) continue;
      data_b[n_b] = molecule[i];
      n_b++;
      data_b[n_b] = tag[i];
      n_b++;
      if (n_b > n_b_max) {
        n_b_max += n_b_max;
        memory->grow(data_b,n_b_max, "fix_break_pol:data_b");
      }
      if (cleave_ind == 1) {
        for (k = j; k < num_bond[i]-1; k++) {
          bond_atom[i][k] = bond_atom[i][k+1];
          bond_type[i][k] = bond_type[i][k+1];
          if (atom->individual) bond_length[i][k] = bond_length[i][k+1];
        }
        num_bond[i]--;
        j--;
      }
    }
  }

  // data gathering
  MPI_Allreduce(&n_b, &n_b_all, 1, MPI_INT, MPI_SUM, world);

  if (cleave_ind == 1)
    atom->nbonds -= floor(0.5*n_b_all);

  if (n_b_all) {
    tagint *data_b_all = NULL;
    memory->create(data_b_all, n_b_all, "fix_break_pol:data_b_all");

    MPI_Gather(&n_b, 1, MPI_INT, rcounts, 1, MPI_INT, 0, world);

    if (!comm->me){
      int offset = 0;
      for (i = 0; i < comm->nprocs; i++) {
        displs[i] = offset;
        offset += rcounts[i];
      }
    }

    MPI_Gatherv(data_b, n_b, MPI_LMP_TAGINT, data_b_all, rcounts, displs, MPI_LMP_TAGINT, 0, world);

    if (!comm->me) {
      // sorting based on tags
      tagint dum[2];
      for (i = 0; i < n_b_all; i+=2)
        for (j = i+2; j < n_b_all; j+=2)
          if (data_b_all[i+1] > data_b_all[j+1]) {
            dum[0] = data_b_all[i];
            dum[1] = data_b_all[i+1];
            data_b_all[i] = data_b_all[j];
            data_b_all[i+1] = data_b_all[j+1];
            data_b_all[j] = dum[0];
            data_b_all[j+1] = dum[1];
          }

    }

    MPI_Bcast(data_b_all, n_b_all, MPI_LMP_TAGINT, 0, world);

    if (cleave_ind == 1) {
      // create new molecules
      for (i = 0; i < nlocal; i++) {
        if (!(mask[i] & groupbit)) continue;
        for (j = n_b_all-2; j >= 0 ; j-=2)
          if (molecule[i] == data_b_all[j])
            if (tag[i] > data_b_all[j+1])
              molecule[i] = n_mol_max + floor(0.5*j) + 1;
      }
      //compute molecule sizes
      atom->calc_n_per_molecule();

      // trigger reneighboring if any bonds were formed
      // this insures neigh lists will immediately reflect the topology changes
      // done if any bonds created

      if (n_b_all) next_reneighbor = update->ntimestep;
    } else {
      if (!comm->me) {
        FILE* output;
        output=fopen(fname,"a");
        for (i = 0; i < n_b_all ; i+=2) {
          fprintf(output,"%lu %lu " TAGINT_FORMAT " \n", update->ntimestep, data_b_all[i], data_b_all[i+1]);
        }
        fclose(output);
      }
    }

    memory->destroy(data_b_all);
  }

  memory->destroy(rcounts);
  memory->destroy(data_b);
  memory->destroy(displs);

  //printf("NOTE: nbonds %i n_b %i \n", atom->nbonds, n_b);
  //error->one(FLERR,"Here it comes!");

  // DEBUG
  //print_bb();

}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly created
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is added
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixBreakPol::check_ghosts()
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
    error->all(FLERR,"Fix break/pol needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixBreakPol::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBreakPol::memory_usage()
{
  double bytes = 0.0;
  bytes += 2*comm->nprocs * sizeof(int);
  bytes += 2*atom->nbonds * sizeof(tagint);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixBreakPol::print_bb()
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

void FixBreakPol::print_copy(const char *str, tagint m, 
                              int n1, int n2, int n3, int *v)
{
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}
