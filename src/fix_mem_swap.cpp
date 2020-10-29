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
   Contributing author: Masoud Hoore (FZJ), Dmitry Fedosov (FZJ)
------------------------------------------------------------------------- */

#include <cmath>
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include "fix_mem_swap.h"
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
#include "domain.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{IGNORE,WARN,ERROR};           // same as thermo.cpp

#define DELTA 100000
#define SQRT3 1.732050808
/* ---------------------------------------------------------------------- 
Syntax: fix ID group-ID	mem/swap Nevery  btype atype dtype prob keyword key_value
arg[]       0  1        2        3       4     5     6     7    8       9

keywords:
seed
keyvalues:
seed: seed_value
-------------------------------------------------------------------------*/
FixMemSwap::FixMemSwap(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Illegal fix mem/swap command");

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"fix mem/swap Nevery must be positive integer.");
  if (nevery % neighbor->every != 0) error->all(FLERR,"fix mem/swap Nevery must be divisible by neighbor every.");
  if (neighbor->delay > 0 && (nevery % neighbor->delay) != 0)
    error->all(FLERR,"fix mem/swap Nevery must be divisible by neighbor delay if delay is more than zero.");

  btype = force->inumeric(FLERR,arg[4]);
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix mem/swap command.");

  atype = force->inumeric(FLERR,arg[5]);
  if (atype < 1 || atype > atom->nangletypes)
    error->all(FLERR,"Invalid angle type in fix mem/swap command.");

  dtype = force->inumeric(FLERR,arg[6]);
  if (dtype < 1 || dtype > atom->ndihedraltypes)
    error->all(FLERR,"Invalid dihedral type in fix mem/swap command.");

  prob = force->numeric(FLERR,arg[7]);
  if (prob < 0.0 || prob > 1.0)
    error->all(FLERR,"Invalid probability in fix mem/swap command.");

  // optional keywords

  int seed = 12345;

  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"seed") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix mem/swap command");
      seed = force->inumeric(FLERR,arg[iarg+1]);
      if (seed <= 0) error->all(FLERR,"Illegal fix mem/swap command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix mem/swap command");
  }

  // error check

  if (!force->newton_bond)
    error->all(FLERR,"fix mem/swap works with newton bond only.");

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix mem/swap with non-molecular systems");


  // set comm sizes needed by this fix
  comm_forward = 1 + 3 * atom->angle_per_atom;

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + comm->me);

  // perform initial allocation of atom-based arrays
  // register with Atom class
  //atom->add_callback(0);

/*
  ///////////////////////////////////////////////////
  //DEBUG
  //if (!comm->me) {
    sprintf(fname,"tags.%i.dat",comm->me);
    FILE* output;
    output=fopen(fname,"w");
    fclose(output);
  //}
  ///////////////////////////////////////////////////
*/

}

/* ---------------------------------------------------------------------- */

FixMemSwap::~FixMemSwap()
{
  // unregister callbacks to this fix from Atom class
  //atom->delete_callback(id,0);
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixMemSwap::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMemSwap::init()
{
  data_dih = NULL;
  data_dih_all = NULL;
  data_adj = NULL;
  data_adj_all = NULL;

  //// %%%%%%%%%%%%%% WHAT IS THIS? %%%%%%%%%%%%%%%%

  // need a half neighbor list, built every Nevery steps

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  //// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
}

/* ---------------------------------------------------------------------- */

void FixMemSwap::pre_exchange()
{

  if (update->ntimestep % nevery) return;

  // updating position of ghost atoms
  comm->forward_comm();
  comm->forward_comm_fix(this,0);

  tagint *tag = atom->tag;
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *mol = atom->molecule;

  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;

  int **dihedrallist = neighbor->dihedrallist;
  int ndihedrallist = neighbor->ndihedrallist;

  int i,j,k,m,n,l;
  int i1,i2,i3,i4,i12,i13,i24,i34,ii1,ii2,ii3,ii4,dtype0;
  tagint tag1,tag2,tag3,tag4,tag12,tag13,tag24,tag34;
  int flag12,flag13,flag24,flag34;
  tagint t1,t2,t3,t4;

  int shared;
  int nmax_data;
  double d23[3],d14[3],rsq23,rsq14,rsq23_c,rsq14_c,n1[3],n2[3],ebond;

  nmax_data = DELTA;
  memory->create(data_dih,nmax_data,"fix_mem_swap:data_dih");

  // By convention, the second atom in angle and dihedral
  // must always keep the information; this helps to set
  // smaller neighboring cutoff radius.

  ndih_swap = 0;
  k = 0;
  for (n = 0; n < ndihedrallist; n++) {

    i1 = dihedrallist[n][0];
    i2 = dihedrallist[n][1];
    i3 = dihedrallist[n][2];
    i4 = dihedrallist[n][3];
    dtype0 = dihedrallist[n][4];

    // for dihedral_style bend/dual (condition 0)
    if (tag[i2] > tag[i3]) continue;

    // swap condition 1 (random selection)
    if (random->uniform() > prob) continue;

    if (!(mask[i2] & groupbit)) continue;
    if (dtype != dtype0) continue;

    i1 = domain->closest_image(i2,i1);
    i3 = domain->closest_image(i2,i3);
    i4 = domain->closest_image(i2,i4);

    d23[0] = x[i2][0] - x[i3][0];
    d23[1] = x[i2][1] - x[i3][1];
    d23[2] = x[i2][2] - x[i3][2];

    d14[0] = x[i1][0] - x[i4][0];
    d14[1] = x[i1][1] - x[i4][1];
    d14[2] = x[i1][2] - x[i4][2];

    rsq23 = d23[0]*d23[0] + d23[1]*d23[1] + d23[2]*d23[2];
    rsq14 = d14[0]*d14[0] + d14[1]*d14[1] + d14[2]*d14[2];

    // swap condition 2 (bond energy minimization)
    if (rsq14 > rsq23) continue;
    //if (SQRT3*rsq23 < rsq14 || rsq14 < rsq23) continue;

    data_dih[k++] = static_cast<double> (tag[i1]);
    data_dih[k++] = static_cast<double> (tag[i2]);
    data_dih[k++] = static_cast<double> (tag[i3]);
    data_dih[k++] = static_cast<double> (tag[i4]);
    data_dih[k++] = rsq23;
    data_dih[k++] = rsq14;
    ndih_swap++;

    if (k > nmax_data) {
      nmax_data += DELTA;
      memory->grow(data_dih,nmax_data, "fix_mem_swap:data_dih");
    }
  }

  // communicate tag1,tag2,tag3,tag4,rsq23,rsq14
  communicate(0,6);
  memory->destroy(data_dih);

  memory->create(data_adj,4*ndih_swap_all,"fix_mem_swap:data_adj");

  // find tag12,tag13,tag24,tag34
  k = 0;
  for (n = 0; n < ndih_swap_all; n++) {
    m = 6*n;
    tag1 = static_cast<tagint> (data_dih_all[m]);
    tag2 = static_cast<tagint> (data_dih_all[m+1]);
    tag3 = static_cast<tagint> (data_dih_all[m+2]);
    tag4 = static_cast<tagint> (data_dih_all[m+3]);

    tagint moli;

    i = atom->map(tag1);
    if (i >= 0)
      moli = mol[i];
    else {
      i = atom->map(tag2);
      if (i >= 0)
        moli = mol[i];
      else {
        i = atom->map(tag3);
        if (i >= 0)
          moli = mol[i];
        else {
          i = atom->map(tag4);
          if (i >= 0)
            moli = mol[i];
          else {
            data_adj[k++] = 0;
            data_adj[k++] = 0;
            data_adj[k++] = 0;
            data_adj[k++] = 0;
            continue;
          }
        }
      }
    }

    flag12 = flag13 = flag24 = flag34 = 0;
    tag12 = tag13 = tag24 = tag34 = 0;

    for (j = 0; j < nanglelist; j++) {
      i1 = anglelist[j][0];
      i2 = anglelist[j][1];
      i3 = anglelist[j][2];

      t1 = tag[i1];
      t2 = tag[i2];
      t3 = tag[i3];

      if (!(mask[i2] & groupbit)) continue;
      if (mol[i2] != moli) continue;

      // find tag12
      if (!flag12)
        if (tag1 == t1 || tag1 == t2 || tag1 == t3)
          if (tag2 == t1 || tag2 == t2 || tag2 == t3)
            if (tag3 != t1 && tag3 != t2 && tag3 != t3) {
              if (t1 != tag1 && t1 != tag2)
                tag12 = t1;
              else if (t2 != tag1 && t2 != tag2)
                tag12 = t2;
              else
                tag12 = t3;
              flag12 = 1;
            }

      // find tag13
      if (!flag13)
        if (tag1 == t1 || tag1 == t2 || tag1 == t3)
          if (tag3 == t1 || tag3 == t2 || tag3 == t3)
            if (tag2 != t1 && tag2 != t2 && tag2 != t3) {
              if (t1 != tag1 && t1 != tag3)
                tag13 = t1;
              else if (t2 != tag1 && t2 != tag3)
                tag13 = t2;
              else
                tag13 = t3;
              flag13 = 1;
            }

      // find tag24
      if (!flag24)
        if (tag2 == t1 || tag2 == t2 || tag2 == t3)
          if (tag4 == t1 || tag4 == t2 || tag4 == t3)
            if (tag3 != t1 && tag3 != t2 && tag3 != t3) {
              if (t1 != tag2 && t1 != tag4)
                tag24 = t1;
              else if (t2 != tag2 && t2 != tag4)
                tag24 = t2;
              else
                tag24 = t3;
              flag24 = 1;
            }

      // find tag34
      if (!flag34)
        if (tag3 == t1 || tag3 == t2 || tag3 == t3)
          if (tag4 == t1 || tag4 == t2 || tag4 == t3)
            if (tag2 != t1 && tag2 != t2 && tag2 != t3) {
              if (t1 != tag3 && t1 != tag4)
                tag34 = t1;
              else if (t2 != tag3 && t2 != tag4)
                tag34 = t2;
              else
                tag34 = t3;
              flag34 = 1;
            }

      if (flag12 && flag13 && flag24 && flag34)
        break;
    }

    data_adj[k++] = tag12;
    data_adj[k++] = tag13;
    data_adj[k++] = tag24;
    data_adj[k++] = tag34;
  }

  // communicate tag12,tag13,tag24,tag34
  communicate(1,4);
  memory->destroy(data_adj);

  // decide on colliding swaps
  for (n = 0; n < ndih_swap_all; n++) {
    m = 6*n;
    tag1 = static_cast<tagint> (data_dih_all[m]);
    tag2 = static_cast<tagint> (data_dih_all[m+1]);
    tag3 = static_cast<tagint> (data_dih_all[m+2]);
    tag4 = static_cast<tagint> (data_dih_all[m+3]);
    rsq23 = data_dih_all[m+4];
    rsq14 = data_dih_all[m+5];
    if (rsq23 < 0) continue;

    for (l = n+1; l < ndih_swap_all; l++) {
      m = 6*l;
      t1 = static_cast<tagint> (data_dih_all[m]);
      t2 = static_cast<tagint> (data_dih_all[m+1]);
      t3 = static_cast<tagint> (data_dih_all[m+2]);
      t4 = static_cast<tagint> (data_dih_all[m+3]);
      rsq23_c = data_dih_all[m+4];
      rsq14_c = data_dih_all[m+5];

      shared = 0;
      if (tag1 == t1 || tag1 == t2 || tag1 == t3 || tag1 == t4)
        shared++;
      if (tag2 == t1 || tag2 == t2 || tag2 == t3 || tag2 == t4)
        shared++;
      if (tag3 == t1 || tag3 == t2 || tag3 == t3 || tag3 == t4)
        shared++;
      if (tag4 == t1 || tag4 == t2 || tag4 == t3 || tag4 == t4)
        shared++;

      // based on only bond energy, choose the most probable one among the colliding swaps
      if (shared > 1) {
        if (rsq14 > rsq14_c)
          m = 6*n;
        else
          m = 6*l;
        data_dih_all[m+4] = -1.0;
      }
    }
  }

  // swap bonds
  int *num_del, *num_del_all, *num_cre, *num_cre_all, num_delcre;
  int dum_int;

  memory->create(num_del,3,"fix_mem_swap:num_del");
  memory->create(num_del_all,3,"fix_mem_swap:num_del");
  memory->create(num_cre,3,"fix_mem_swap:num_del");
  memory->create(num_cre_all,3,"fix_mem_swap:num_del");

  for (n = 0; n < 3; n++)
    num_del[n] = num_cre[n] = 0;
  num_delcre = 0;

  for (n = 0; n < ndih_swap_all; n++) {
    m = 6*n;
    rsq23 = data_dih_all[m+4];
    if (rsq23 < 0) continue;
    rsq14 = data_dih_all[m+5];

    tag1 = static_cast<tagint> (data_dih_all[m]);
    tag2 = static_cast<tagint> (data_dih_all[m+1]);
    tag3 = static_cast<tagint> (data_dih_all[m+2]);
    tag4 = static_cast<tagint> (data_dih_all[m+3]);

    m = 4*n;
    tag12 = data_adj_all[m];
    tag13 = data_adj_all[m+1];
    tag24 = data_adj_all[m+2];
    tag34 = data_adj_all[m+3];

    if (tag12 == 0 || tag13 == 0 || tag24 == 0 || tag34 == 0) {
      char str[256];
      sprintf(str,"Error in finding tags 12, 13, 24, 34 at step %li."
                 ,update->ntimestep);
      error->one(FLERR,str);
    }

    i1 = atom->map(tag1);
    i2 = atom->map(tag2);
    i3 = atom->map(tag3);
    i4 = atom->map(tag4);
    i12 = atom->map(tag12);
    i13 = atom->map(tag13);
    i24 = atom->map(tag24);
    i34 = atom->map(tag34);

    // modify bonds
    num_del[0] += delete_bond(i2,i3);
    num_cre[0] += create_bond(i1,i4);

    // modify angles
    area = 0.0;
    num_del[1] += delete_angle(i1,i2,i3);

    for (i = 0; i < 3; i++)
      n1[i] = normal[i];

    num_del[1] += delete_angle(i2,i3,i4);

    for (i = 0; i < 3; i++)
      n2[i] = normal[i];

    num_cre[1] += create_angle(i1,i3,i4,n1,n2);
    num_cre[1] += create_angle(i1,i2,i4,n1,n2);

    // modify dihedrals
    num_del[2] += delete_dihedral(i1,i2,i3,i4);
    num_del[2] += delete_dihedral(i12,i1,i2,i3);
    num_del[2] += delete_dihedral(i13,i1,i3,i2);
    num_del[2] += delete_dihedral(i24,i2,i4,i3);
    num_del[2] += delete_dihedral(i34,i3,i4,i2);

    num_cre[2] += create_dihedral(i3,i4,i1,i2);
    num_cre[2] += create_dihedral(i4,i2,i1,i12);
    num_cre[2] += create_dihedral(i4,i3,i1,i13);
    num_cre[2] += create_dihedral(i1,i4,i2,i24);
    num_cre[2] += create_dihedral(i1,i4,i3,i34);

    // for dihedral_style bend/dual
    num_del[2] += delete_dihedral(i4,i3,i2,i1);
    num_del[2] += delete_dihedral(i3,i2,i1,i12);
    num_del[2] += delete_dihedral(i2,i3,i1,i13);
    num_del[2] += delete_dihedral(i3,i4,i2,i24);
    num_del[2] += delete_dihedral(i2,i4,i3,i34);

    num_cre[2] += create_dihedral(i2,i1,i4,i3);
    num_cre[2] += create_dihedral(i12,i1,i2,i4);
    num_cre[2] += create_dihedral(i13,i1,i3,i4);
    num_cre[2] += create_dihedral(i24,i2,i4,i1);
    num_cre[2] += create_dihedral(i34,i3,i4,i1);

    num_delcre++;

    //if (!comm->me)
    //printf("proc%i: -- tags 1,2,3,4 " TAGINT_FORMAT ", " TAGINT_FORMAT ", " TAGINT_FORMAT ", " TAGINT_FORMAT " \t"
      //     "complt  -- tags 12,13,24,34 " TAGINT_FORMAT ", " TAGINT_FORMAT ", " TAGINT_FORMAT ", " TAGINT_FORMAT " \t"
        //   "at step %li. \n",comm->me,tag1,tag2,tag3,tag4,tag12,tag13,tag24,tag34,update->ntimestep);
  }

  MPI_Reduce(num_del,num_del_all,3,MPI_INT,MPI_SUM,0,world);
  MPI_Reduce(num_cre,num_cre_all,3,MPI_INT,MPI_SUM,0,world);

/*
  ///////////////////////////////////////////////////
  //DEBUG
  //if (!comm->me) {
    k = 0;
    FILE* output;
    output=fopen(fname,"a");
    fprintf(output,"ITEM: timestep\n");
    fprintf(output,"%li \n", update->ntimestep);
    fprintf(output,"ITEM: Number of entries\n");
    fprintf(output,"%li \n", num_delcre);
    fprintf(output,"ITEM: index tag1 tag2 tag3 tag4 tag12 tag13 tag24 tag34 \n");

    for (n = 0; n < ndih_swap_all; n++) {
      m = 6*n;
      rsq23 = data_dih_all[m+4];
      if (rsq23 < 0) continue;
      rsq14 = data_dih_all[m+5];

      tag1 = static_cast<tagint> (data_dih_all[m]);
      tag2 = static_cast<tagint> (data_dih_all[m+1]);
      tag3 = static_cast<tagint> (data_dih_all[m+2]);
      tag4 = static_cast<tagint> (data_dih_all[m+3]);
      rsq23 = data_dih_all[m+4];

      m = 4*n;
      tag12 = data_adj_all[m];
      tag13 = data_adj_all[m+1];
      tag24 = data_adj_all[m+2];
      tag34 = data_adj_all[m+3];

      k++;
      fprintf(output,"%i \t " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT " \t "
                     "" TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT " \t "
                     "rsq23 = %f \n",k,tag1,tag2,tag3,tag4,tag12,tag13,tag24,tag34,rsq23);
    }

    fclose(output);
  //}
  ///////////////////////////////////////////////////
*/

  memory->destroy(data_dih_all);
  memory->destroy(data_adj_all);

  if (!comm->me)
    printf("at step %li del(cre) bond= %i(%i) -- angle= %i(%i) -- dihedral= %i(%i) -- swap_total= %i\n"
          ,update->ntimestep, num_del_all[0], num_cre_all[0], num_del_all[1], num_cre_all[1], num_del_all[2], num_cre_all[2], num_delcre);

  if (!comm->me)
    if (num_del_all[2] - num_cre_all[2] != 0) {
      char str[256];
      sprintf(str,"Error in dihedral swapping at step %li. deletion = %i,  creation = %i"
                 ,update->ntimestep, num_del_all[2], num_cre_all[2]);
      error->one(FLERR,str);
    }

  delete(num_del);
  delete(num_del_all);
  delete(num_cre);
  delete(num_cre_all);

}

/* ---------------------------------------------------------------------- */

int FixMemSwap::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m;

  int *num_angle = atom->num_angle;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(num_angle[j]).d;
    for (k = 0; k < num_angle[j]; k++) {
      buf[m++] = ubuf(angle_atom1[j][k]).d;
      buf[m++] = ubuf(angle_atom2[j][k]).d;
      buf[m++] = ubuf(angle_atom3[j][k]).d;
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixMemSwap::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,k,last;

  int *num_angle = atom->num_angle;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) {
    num_angle[i] = (int) ubuf(buf[m++]).i;
    for (k = 0; k < num_angle[i]; k++) {
      angle_atom1[i][k] = (tagint) ubuf(buf[m++]).i;
      angle_atom2[i][k] = (tagint) ubuf(buf[m++]).i;
      angle_atom3[i][k] = (tagint) ubuf(buf[m++]).i;
    }
  }

}

/* ---------------------------------------------------------------------- */

void FixMemSwap::communicate(int comm_flag, int size)
{
  int i;

  int *rcounts = NULL;
  int *displs = NULL;

  int offset = 0;

  memory->create(rcounts,comm->nprocs,"fix_mem_swap:rcounts");
  memory->create(displs,comm->nprocs,"fix_mem_swap:displs");

  if (comm_flag == 0) {

    MPI_Allreduce(&ndih_swap,&ndih_swap_all,1,MPI_INT,MPI_SUM,world);

    MPI_Allgather(&ndih_swap,1,MPI_INT,rcounts,1,MPI_INT,world);

    for (i = 0; i < comm->nprocs; i++) {
      rcounts[i] *= size;
      displs[i] = offset;
      offset += rcounts[i];
    }

    memory->create(data_dih_all,size*ndih_swap_all,"fix_mem_swap:data_dih_all");
    MPI_Allgatherv(data_dih,size*ndih_swap,MPI_DOUBLE,data_dih_all,rcounts,displs,MPI_DOUBLE,world);

  } else if (comm_flag == 1) {

    memory->create(data_adj_all,size*ndih_swap_all,"fix_mem_swap:data_adj_all");
    MPI_Allreduce(data_adj,data_adj_all,size*ndih_swap_all,MPI_LMP_TAGINT,MPI_SUM,world);

  }

  delete(rcounts);
  delete(displs);
}

/* ---------------------------------------------------------------------- */

int FixMemSwap::delete_bond(int i1, int i2)
{
  if (i1 < 0 || i2 < 0)
    return 0;

  int a,b,iter,i,j;

  tagint *tag = atom->tag;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int nlocal = atom->nlocal;

  iter = 0;
  while ( iter < 2) {

    if (iter == 0) {
      i = i1;
      j = i2;
    } else if (iter == 1) {
      i = i2;
      j = i1;
    }

    if (i < nlocal)
      for (a = 0; a < num_bond[i]; a++)
        if (bond_atom[i][a] == tag[j]) {
          for (b = a; b < num_bond[i]-1; b++) {
            bond_atom[i][b] = bond_atom[i][b+1];
            bond_type[i][b] = bond_type[i][b+1];
            if (atom->individual)
              atom->bond_length[i][b] = atom->bond_length[i][b+1];
          }
          num_bond[i]--;
          return 1;
        }

    iter++;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */
// delete angles for ghost atoms as well in order to find normal vector of 
// the triangle and specify inner volume of the membrane. The calculated normal
// vector is used for creating new triangles with correct normal vectors.
 
int FixMemSwap::delete_angle(int i1, int i2, int i3)
{
  if (i1 < 0 || i2 < 0 || i3 < 0)
    return 0;

  int a,b,i,ii,jj,kk,iter;
  tagint t1,t2,t3,atag1,atag2,atag3;
  double d21[3],d31[3];

  tagint *tag = atom->tag;
  int *num_angle = atom->num_angle;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  int **angle_type = atom->angle_type;
  int nlocal = atom->nlocal;

  double **x = atom->x;

  t1 = tag[i1];
  t2 = tag[i2];
  t3 = tag[i3];

  iter = 0;
  while ( iter < 3) {

    if (iter == 0)
      i = i1;
    else if (iter == 1)
      i = i2;
    else if (iter == 2)
      i = i3;

    for (a = 0; a < num_angle[i]; a++) {
      atag1 = angle_atom1[i][a];
      atag2 = angle_atom2[i][a];
      atag3 = angle_atom3[i][a];

      if (atag1 != t1 && atag1 != t2 && atag1 != t3) continue;
      if (atag2 != t1 && atag2 != t2 && atag2 != t3) continue;
      if (atag3 != t1 && atag3 != t2 && atag3 != t3) continue;

      ii = atom->map(atag1);
      jj = atom->map(atag2);
      kk = atom->map(atag3);

      ii = domain->closest_image(i,ii);
      jj = domain->closest_image(i,jj);
      kk = domain->closest_image(i,kk);

      // distances
      for (b = 0; b < 3; b++) {
        d21[b] = x[jj][b] - x[ii][b];
        d31[b] = x[kk][b] - x[ii][b];
      }

      // calculate normal
      normal[0] = d21[1]*d31[2] - d31[1]*d21[2];
      normal[1] = d31[0]*d21[2] - d21[0]*d31[2];
      normal[2] = d21[0]*d31[1] - d31[0]*d21[1];

      area += 0.5*sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);

      for (b = a; b < num_angle[i]-1; b++) {
        angle_atom1[i][b] = angle_atom1[i][b+1];
        angle_atom2[i][b] = angle_atom2[i][b+1];
        angle_atom3[i][b] = angle_atom3[i][b+1];
        angle_type[i][b] = angle_type[i][b+1];
        if (atom->individual)
          atom->angle_area[i][b] = atom->angle_area[i][b+1];
      }

      num_angle[i]--;
      if (i < nlocal)
        return 1;
      else
        return 0;
    }

    iter++;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixMemSwap::delete_dihedral(int i1, int i2, int i3, int i4)
{
  if (i1 < 0 || i2 < 0 || i3 < 0 || i4 < 0 || i2 >= atom->nlocal)
    return 0;

  int a,b;
  tagint t1,t2,t3,t4,atag1,atag2,atag3,atag4;

  tagint *tag = atom->tag;
  int *num_dihedral = atom->num_dihedral;
  tagint **dihedral_atom1 = atom->dihedral_atom1;
  tagint **dihedral_atom2 = atom->dihedral_atom2;
  tagint **dihedral_atom3 = atom->dihedral_atom3;
  tagint **dihedral_atom4 = atom->dihedral_atom4;
  int **dihedral_type = atom->dihedral_type;
  int nlocal = atom->nlocal;

  double **x = atom->x;

  t1 = tag[i1];
  t2 = tag[i2];
  t3 = tag[i3];
  t4 = tag[i4];

  for (a = 0; a < num_dihedral[i2]; a++) {
    atag1 = dihedral_atom1[i2][a];
    atag2 = dihedral_atom2[i2][a];
    atag3 = dihedral_atom3[i2][a];
    atag4 = dihedral_atom4[i2][a];

    if (atag3 != t3) continue;

    for (b = a; b < num_dihedral[i2]-1; b++) {
      dihedral_atom1[i2][b] = dihedral_atom1[i2][b+1];
      dihedral_atom2[i2][b] = dihedral_atom2[i2][b+1];
      dihedral_atom3[i2][b] = dihedral_atom3[i2][b+1];
      dihedral_atom4[i2][b] = dihedral_atom4[i2][b+1];
      dihedral_type[i2][b] = dihedral_type[i2][b+1];
      if (atom->individual)
        atom->dihedral_angle[i2][b] = atom->dihedral_angle[i2][b+1];
    }
    num_dihedral[i2]--;
    return 1;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixMemSwap::create_bond(int i1, int i2)
{
  if (i1 < 0 || i2 < 0 || i1 >= atom->nlocal)
    return 0;

  tagint *tag = atom->tag;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;

  if (num_bond[i1] >= atom->bond_per_atom)
    error->one(FLERR,"New bond exceeded bond per atom in fix mem/swap");

  bond_type[i1][num_bond[i1]] = btype;
  bond_atom[i1][num_bond[i1]] = tag[i2];
  if (atom->individual)
    atom->bond_length[i1][num_bond[i1]] = 0.0;
  num_bond[i1]++;

  return 1;

}

/* ---------------------------------------------------------------------- */

int FixMemSwap::create_angle(int i1, int i2, int i3, double *n1, double *n2)
{
  if (i1 < 0 || i2 < 0 || i3 < 0 || i2 >= atom->nlocal)
    return 0;

  int b;
  double d21[3],d31[3],dot1,dot2;

  int *num_angle = atom->num_angle;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  int **angle_type = atom->angle_type;
  tagint *tag = atom->tag;
  double **x = atom->x;

  if (num_angle[i2] >= atom->angle_per_atom)
    error->one(FLERR,"New angle exceeded angle per atom in fix mem/swap");

  angle_type[i2][num_angle[i2]] = atype;

  i1 = domain->closest_image(i2,i1);
  i3 = domain->closest_image(i2,i3);

  // distances
  for (b = 0; b < 3; b++) {
    d21[b] = x[i2][b] - x[i1][b];
    d31[b] = x[i3][b] - x[i1][b];
  }

  // calculate normal
  normal[0] = d21[1]*d31[2] - d31[1]*d21[2];
  normal[1] = d31[0]*d21[2] - d21[0]*d31[2];
  normal[2] = d21[0]*d31[1] - d31[0]*d21[1];

  dot1 = normal[0]*n1[0] + normal[1]*n1[1] + normal[2]*n1[2];
  dot2 = normal[0]*n2[0] + normal[1]*n2[1] + normal[2]*n2[2];

  if (dot1 > 0 && dot2 > 0) {
    angle_atom1[i2][num_angle[i2]] = tag[i1];
    angle_atom2[i2][num_angle[i2]] = tag[i2];
    angle_atom3[i2][num_angle[i2]] = tag[i3];
  } else {
    angle_atom1[i2][num_angle[i2]] = tag[i3];
    angle_atom2[i2][num_angle[i2]] = tag[i2];
    angle_atom3[i2][num_angle[i2]] = tag[i1];
  }

  if (atom->individual)
    atom->angle_area[i2][num_angle[i2]] = 0.5*area;
  num_angle[i2]++;

  return 1;
}

/* ---------------------------------------------------------------------- */

int FixMemSwap::create_dihedral(int i1, int i2, int i3, int i4)
{
  if (i1 < 0 || i2 < 0 || i3 < 0 || i4 < 0 || i2 >= atom->nlocal)
    return 0;

  int i;
  double d21[3],d31[3],d34[4],d24[3],d14[3],n1[3],n2[3],nn;

  tagint *tag = atom->tag;
  int *num_dihedral = atom->num_dihedral;
  tagint **dihedral_atom1 = atom->dihedral_atom1;
  tagint **dihedral_atom2 = atom->dihedral_atom2;
  tagint **dihedral_atom3 = atom->dihedral_atom3;
  tagint **dihedral_atom4 = atom->dihedral_atom4;
  int **dihedral_type = atom->dihedral_type;

  double **x = atom->x;

  if (num_dihedral[i2] >= atom->dihedral_per_atom)
    error->one(FLERR,"New dihedral exceeded dihedral per atom in fix mem/swap");

  dihedral_type[i2][num_dihedral[i2]] = dtype;

  i1 = domain->closest_image(i2,i1);
  i3 = domain->closest_image(i2,i3);
  i4 = domain->closest_image(i2,i4);

  // distances
  for (i = 0; i < 3; i++) {
    d21[i] = x[i2][i] - x[i1][i];
    d31[i] = x[i3][i] - x[i1][i];
    d34[i] = x[i3][i] - x[i4][i];
    d24[i] = x[i2][i] - x[i4][i];
    d14[i] = x[i1][i] - x[i4][i];
  }

  // calculate normals
  n1[0] = d21[1]*d31[2] - d31[1]*d21[2];
  n1[1] = d31[0]*d21[2] - d21[0]*d31[2];
  n1[2] = d21[0]*d31[1] - d31[0]*d21[1];

  n2[0] = d34[1]*d24[2] - d24[1]*d34[2];
  n2[1] = d24[0]*d34[2] - d34[0]*d24[2];
  n2[2] = d34[0]*d24[1] - d24[0]*d34[1];

  nn = (n1[0]-n2[0])*d14[0] + (n1[1]-n2[1])*d14[1] + (n1[2]-n2[2])*d14[2];

  if (nn >= 0.0) {
    dihedral_atom1[i2][num_dihedral[i2]] = tag[i1];
    dihedral_atom2[i2][num_dihedral[i2]] = tag[i2];
    dihedral_atom3[i2][num_dihedral[i2]] = tag[i3];
    dihedral_atom4[i2][num_dihedral[i2]] = tag[i4];
  } else {
    dihedral_atom1[i2][num_dihedral[i2]] = tag[i4];
    dihedral_atom2[i2][num_dihedral[i2]] = tag[i2];
    dihedral_atom3[i2][num_dihedral[i2]] = tag[i3];
    dihedral_atom4[i2][num_dihedral[i2]] = tag[i1];
  }

  if (atom->individual)
    atom->dihedral_angle[i2][num_dihedral[i2]] = 0.0;
  num_dihedral[i2]++;

  return 1;
}

/* ---------------------------------------------------------------------- */

void FixMemSwap::bad_topology(tagint t1, tagint t2, tagint t3, tagint t4, tagint t12, tagint t13, tagint t24, tagint t34)
{
  char str[256];
  sprintf(str,"Bad topology in fix mem/swap -- \n"
              "tags 1,2,3,4 " TAGINT_FORMAT ", " TAGINT_FORMAT ", " TAGINT_FORMAT ", " TAGINT_FORMAT " --\n"
              "tags 12,13,24,34 " TAGINT_FORMAT ", " TAGINT_FORMAT ", " TAGINT_FORMAT ", " TAGINT_FORMAT " \t"
              "at step %li.",t1,t2,t3,t4,t12,t13,t24,t34,update->ntimestep);
  error->one(FLERR,str);
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixMemSwap::memory_usage()
{
  double bytes = 0.0;
  bytes += 6*ndih_swap_all * sizeof(int);
  bytes += 2*4*ndih_swap_all * sizeof(tagint);
  return bytes;
}
