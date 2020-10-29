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

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include "comm_mosaic.h"
#include "comm_tiled.h"
#include "universe.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h" 
#include "domain.h"
#include "neighbor.h"
#include "group.h"
#include "modify.h"
#include "fix.h"
#include "compute.h"
#include "output.h"
#include "dump.h"
#include "math_extra.h"
#include "error.h"
#include "memory.h"
#include "update.h"

using namespace std;
using namespace LAMMPS_NS;

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 1000
#define EXT_ARRAY 100
#define EXT_ARRAY_SMALL 10
#define MAX_GBITS 64

enum{LAYOUT_UNIFORM,LAYOUT_NONUNIFORM,LAYOUT_TILED};    // several files

/* ---------------------------------------------------------------------- */

CommMosaic::CommMosaic(LAMMPS *lmp) : Comm(lmp)
{
  style = 2;
  layout = LAYOUT_UNIFORM;
  init_buffers();
}

/* ---------------------------------------------------------------------- */

CommMosaic::~CommMosaic()
{
  free_swap();

  if (sendlist) for (int i = 0; i < maxswap; i++) memory->destroy(sendlist[i]);
  memory->sfree(sendlist);
  memory->destroy(maxsendlist);

  memory->destroy(buf_send);
  memory->destroy(buf_recv);

  memory->destroy(own_bin_orig);
  memory->destroy(nbinh_orig);
  memory->destroy(my_neigh_orig);
  memory->destroy(send_bit);
  memory->destroy(exchange_bit);
}

/* ---------------------------------------------------------------------- */

CommMosaic::CommMosaic(LAMMPS *lmp, Comm *oldcomm) : Comm(*oldcomm)
{
  if (oldcomm->layout == LAYOUT_TILED)
    error->all(FLERR,"Cannot change to comm_style mosaic from tiled layout");

  style = 2;
  layout = oldcomm->layout;
  copy_arrays(oldcomm);
  init_buffers();
}

/* ---------------------------------------------------------------------- */

void CommMosaic::init_buffers()
{
  mode = 0; 
 
  user_part = 1;
  own_bin_orig = own_bin = NULL;
  nbinh_orig = my_neigh_orig = NULL;
  nbinh = my_neigh = twice_comm = neigh_ind = NULL;
  send_bit = exchange_bit = NULL;

  // bufextra = max size of one exchanged atom
  //          = allowed overflow of sendbuf in exchange()
  // atomvec, fix reset these 2 maxexchange values if needed
  // only necessary if their size > BUFEXTRA

  maxexchange = maxexchange_atom + maxexchange_fix;
  bufextra = maxexchange + BUFEXTRA;

  maxsend = BUFMIN;
  memory->create(buf_send,maxsend+bufextra,"comm:buf_send");
  maxrecv = BUFMIN;
  memory->create(buf_recv,maxrecv,"comm:buf_recv");

  maxswap = 6;
  allocate_swap(maxswap);

  sendlist = (int **) memory->smalloc(maxswap*sizeof(int *),"comm:sendlist");
  memory->create(maxsendlist,maxswap,"comm:maxsendlist");
  for (int i = 0; i < maxswap; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i],BUFMIN,"comm:sendlist[i]");
  }
}

/* ---------------------------------------------------------------------- */

void CommMosaic::init()
{
  Comm::init();
}

/* ----------------------------------------------------------------------
   setup spatial-decomposition communication patterns
   function of neighbor cutoff(s) & cutghostuser & current box size
   single mode sets slab boundaries (slablo,slabhi) based on max cutoff
   multi mode sets type-dependent slab boundaries (multilo,multihi)
------------------------------------------------------------------------- */

void CommMosaic::setup()
{
  int i,j,k;

  // cutghost[] = max distance at which ghost atoms need to be acquired
  // for orthogonal:
  //   cutghost is in box coords = neigh->cutghost in all 3 dims

  double cut = MAX(neighbor->cutneighmax,cutghostuser);
  periodicity = domain->periodicity;

  cutghost[0] = cutghost[1] = cutghost[2] = cut;
  if (processed_data_ind == 0){
    process_bin_data(n_neigh,n_own,nbp,bin_ext,nbp_max,period_own,box_min,cutghost);

    for (j = 0; j < 3; j++)
      nbp_loc[j] = nbp_max[j] - nbp_min[j] + 1;

    memory->destroy(m_comm);
    m_comm = NULL;
    memory->create(m_comm,nbp_loc[0],nbp_loc[1],nbp_loc[2],"comm:m_comm");
  
    for (i = 0; i < nbp_loc[0]; i++)
      for (j = 0; j < nbp_loc[1]; j++)
        for (k = 0; k < nbp_loc[2]; k++)
          m_comm[i][j][k] = 0;
  
    // create bit arrays

    nswap = n_neigh + n_own;
    grow_swap(nswap);
    if (2*n_neigh + n_own + 1 > MAX_GBITS)
      error->one(FLERR,"Not enough bits for the communication matrix in comm_mosaic!");

    memory->create(send_bit,n_neigh,"comm:send_bit");
    memory->create(exchange_bit,n_neigh,"comm:exchange_bit");

    build_comm_matrix(n_neigh,m_comm,send_bit,nbp,nbp_max,nbp_loc,bin_ext,period_own,cutghost,sendproc,recvproc,pbc_flag,pbc,1);
  }
  //domain->set_local_box();
}

/* ----------------------------------------------------------------------
   forward communication of atom coords every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

void CommMosaic::forward_comm(int dummy)
{
  int n,iswap,dim;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **x = atom->x;
  double *buf;

  // exchange data with another proc
  // if other proc is self, just copy
  // if comm_x_only set, exchange or copy directly to x, don't unpack

  for (iswap = 0; iswap < n_neigh; iswap++) {

    if (comm_x_only) {
      if (size_forward_recv[iswap]) buf = x[firstrecv[iswap]];
      else buf = NULL;
      if (size_forward_recv[iswap])
        MPI_Irecv(buf,size_forward_recv[iswap],MPI_DOUBLE,
                  recvproc[iswap],recvproc[iswap],world,&request);
      n = avec->pack_comm(sendnum[iswap],sendlist[iswap],
                        buf_send,pbc_flag[iswap],pbc[iswap]);
      if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],me,world);
      if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    } else if (ghost_velocity) {
      if (size_forward_recv[iswap])
        MPI_Irecv(buf_recv,size_forward_recv[iswap],MPI_DOUBLE,
                  recvproc[iswap],recvproc[iswap],world,&request);
      n = avec->pack_comm_vel(sendnum[iswap],sendlist[iswap],
                              buf_send,pbc_flag[iswap],pbc[iswap]);
      if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],me,world);
      if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      avec->unpack_comm_vel(recvnum[iswap],firstrecv[iswap],buf_recv);
    } else {
      if (size_forward_recv[iswap])
        MPI_Irecv(buf_recv,size_forward_recv[iswap],MPI_DOUBLE,
                  recvproc[iswap],recvproc[iswap],world,&request);
      n = avec->pack_comm(sendnum[iswap],sendlist[iswap],
                          buf_send,pbc_flag[iswap],pbc[iswap]);
      if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],me,world);
      if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      avec->unpack_comm(recvnum[iswap],firstrecv[iswap],buf_recv);
    }
  }

  for (dim = 0; dim < n_own; dim++) {
    iswap = n_neigh + dim;
    if (comm_x_only) {
      if (sendnum[iswap])
        n = avec->pack_comm(sendnum[iswap],sendlist[iswap],
                            x[firstrecv[iswap]],pbc_flag[iswap],
                            pbc[iswap]);
    } else if (ghost_velocity) {
      n = avec->pack_comm_vel(sendnum[iswap],sendlist[iswap],
                              buf_send,pbc_flag[iswap],pbc[iswap]);
      avec->unpack_comm_vel(recvnum[iswap],firstrecv[iswap],buf_send);
    } else {
      n = avec->pack_comm(sendnum[iswap],sendlist[iswap],
                          buf_send,pbc_flag[iswap],pbc[iswap]);
      avec->unpack_comm(recvnum[iswap],firstrecv[iswap],buf_send);
    }
  }
}

/* ----------------------------------------------------------------------
   reverse communication of forces on atoms every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

void CommMosaic::reverse_comm()
{
  int n,iswap,dim;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  double *buf;

  // exchange data with another proc
  // if other proc is self, just copy
  // if comm_f_only set, exchange or copy directly from f, don't pack

  for (dim = n_own-1; dim >= 0; dim--) {
    iswap = n_neigh + dim;
    if (comm_f_only) {
      if (sendnum[iswap])
        avec->unpack_reverse(sendnum[iswap],sendlist[iswap],
                            f[firstrecv[iswap]]);
    } else {
      n = avec->pack_reverse(recvnum[iswap],firstrecv[iswap],buf_send);
      avec->unpack_reverse(sendnum[iswap],sendlist[iswap],buf_send);
    }
  }

  for (iswap = n_neigh-1; iswap >= 0; iswap--) {
    if (comm_f_only) {
      if (size_reverse_recv[iswap])
        MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,
                  sendproc[iswap],me,world,&request);
      if (size_reverse_send[iswap]) buf = f[firstrecv[iswap]];
      else buf = NULL;
      if (size_reverse_send[iswap])
        MPI_Send(buf,size_reverse_send[iswap],MPI_DOUBLE,
                 recvproc[iswap],recvproc[iswap],world);
      if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    } else {
      if (size_reverse_recv[iswap])
        MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,
                  sendproc[iswap],me,world,&request);
      n = avec->pack_reverse(recvnum[iswap],firstrecv[iswap],buf_send);
      if (n) MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world);
      if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    }
    avec->unpack_reverse(sendnum[iswap],sendlist[iswap],buf_recv);
  }
}

/* ----------------------------------------------------------------------
   exchange: move atoms to correct processors
   send out atoms that have left my box, receive ones entering my box
   atoms will be lost if not inside a stencil proc's box
     can happen if atom moves outside of non-periodic bounary
     or if atom moves more than one proc away
   this routine called before every reneighboring
------------------------------------------------------------------------- */

void CommMosaic::exchange()
{
  int i,m,nsend,nrecv,nrecv1,nlocal,dim,nn[3],ind;
  double **x;
  double *buf;
  MPI_Request request;
  AtomVec *avec = atom->avec;

  // clear global->local map for owned and ghost atoms
  // b/c atoms migrate to new procs in exchange() and
  //   new ghosts are created in borders()
  // map_set() is done at end of borders()
  // clear ghost count and any ghost bonus data internal to AtomVec

  if (map_style) atom->map_clear();
  atom->nghost = 0;
  atom->avec->clear_bonus();

  // insure send buf is large enough for single atom
  // bufextra = max size of one atom = allowed overflow of sendbuf
  // fixes can change per-atom size requirement on-the-fly

  int bufextra_old = bufextra;
  maxexchange = maxexchange_atom + maxexchange_fix;
  bufextra = maxexchange + BUFEXTRA;
  if (bufextra > bufextra_old)
    memory->grow(buf_send,maxsend+bufextra,"comm:buf_send");

  // loop over neighbors

  for (dim = 0; dim < n_neigh; dim++) {

    // fill buffer with atoms leaving my box, using comm matrix
    // when atom is deleted, fill it in with last atom

    x = atom->x;
    nlocal = atom->nlocal;
    i = nsend = 0;

    //if (me == 0) printf("dim=%d, matrix=%d \n",dim,m_comm[12][18][9]); 
    while (i < nlocal) {
      ind = coords_to_bin_exchange(x[i],nn);
      //if (atom->tag[i] == 213 && (m_comm[nn[0]][nn[1]][nn[2]] & exchange_bit[dim]))
      //  printf("me=%d, step=%d, dim=%d, ind=%d, nn=%d %d %d, bit=%d, matrix=%d \n",me,update->ntimestep,dim,ind,nn[0],nn[1],nn[2],exchange_bit[dim],m_comm[nn[0]][nn[1]][nn[2]]); 
      //printf("hh me=%d, step=%d, dim=%d, nn=%d %d %d, ind=%d, nbp_loc=%d %d %d, x=%f %f %f, tag=%d, bit=%d, matrix=%d, mmat=%d \n",me,update->ntimestep,dim,nn[0],nn[1],nn[2],ind,nbp_loc[0],nbp_loc[1],nbp_loc[2],x[i][0],x[i][1],x[i][2],atom->tag[i],exchange_bit[dim],m_comm[nn[0]][nn[1]][nn[2]],m_comm[12][22][12]); 
      if (ind && (m_comm[nn[0]][nn[1]][nn[2]] & exchange_bit[dim])) {
        //if (nsend < 30)
        //printf("hh me=%d, step=%d, dim=%d, nn=%d %d %d, ind=%d, nbp_loc=%d %d %d, x=%f %f %f, tag=%d, bit=%d, matrix=%d, mmat=%d \n",me,update->ntimestep,dim,nn[0],nn[1],nn[2],ind,nbp_loc[0],nbp_loc[1],nbp_loc[2],x[i][0],x[i][1],x[i][2],atom->tag[i],exchange_bit[dim],m_comm[nn[0]][nn[1]][nn[2]],m_comm[12][22][12]);
        if (nsend > maxsend) grow_send(nsend,1);
        nsend += avec->pack_exchange(i,&buf_send[nsend]);
        avec->copy(nlocal-1,i,1);
        nlocal--;
      } else
        i++;
    }
    atom->nlocal = nlocal;

    // send/recv atoms in both directions
    //if (nsend > 0) printf("me=%d, step=%d, dim=%d, nsend=%d \n",me,update->ntimestep,dim,nsend);

    MPI_Sendrecv(&nsend,1,MPI_INT,sendproc[dim],me,
                 &nrecv1,1,MPI_INT,recvproc[dim],recvproc[dim],world,MPI_STATUS_IGNORE);
    nrecv = nrecv1;
    if (nrecv > maxrecv) grow_recv(nrecv);

    MPI_Irecv(buf_recv,nrecv1,MPI_DOUBLE,recvproc[dim],recvproc[dim],
              world,&request);
    MPI_Send(buf_send,nsend,MPI_DOUBLE,sendproc[dim],me,world);
    MPI_Wait(&request,MPI_STATUS_IGNORE);

    buf = buf_recv;

    // check incoming atoms to see if they are in my box
    // if so, add to my list

    m = 0;
    while (m < nrecv) {
      ind = coords_to_bin(&buf[m+1],nn,nbp_loc,box_min);              // normally should not be checked, should remove after testing
      if ((ind == 0) && (m_comm[nn[0]][nn[1]][nn[2]] & local_bit))
        printf("00 me=%d, step=%ld, dim=%d, nn=%d %d %d \n",me,update->ntimestep,dim,nn[0],nn[1],nn[2]);
      if (ind && (m_comm[nn[0]][nn[1]][nn[2]] & local_bit)) {
        m += avec->unpack_exchange(&buf[m]);
      } else{
        //printf("11 me=%d, step=%d, dim=%d, nn=%d %d %d, ind=%d, nbp_loc=%d %d %d, x=%f %f %f, tag=%f, bit=%d, matrix=%d \n",me,update->ntimestep,dim,nn[0],nn[1],nn[2],ind,nbp_loc[0],nbp_loc[1],nbp_loc[2],buf[m+1],buf[m+2],buf[m+3],buf[m+7],local_bit,m_comm[nn[0]][nn[1]][nn[2]]); 
        m += static_cast<int> (buf[m]);
        error->one(FLERR,"Particles get lost in unpack exchange!");
      }
    }
  }

  x = atom->x;
  for (i = 0; i < atom->nlocal; i++) {
    ind = coords_to_bin(x[i],nn,nbp_loc,box_min);
    if (!(m_comm[nn[0]][nn[1]][nn[2]] & local_bit)){
      fprintf(stderr,"NOT A LOCAL PARTICLE - IT HAS BEEN DELETED: me=%d; step=%d; ind=%d; nn=%d %d %d; x=%f %f %f; tag=%d. \n",me,update->ntimestep,ind,nn[0],nn[1],nn[2],x[i][0],x[i][1],x[i][2],atom->tag[i]);
      avec->copy(atom->nlocal-1,i,1);
      atom->nlocal--;
    }
  }
  //sleep(5);

  if (atom->firstgroupname) atom->first_reorder();
}

/* ----------------------------------------------------------------------
   borders: list nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a communicate, so don't need to explicitly
     call communicate routine on reneighboring timestep
   this routine is called before every reneighboring
------------------------------------------------------------------------- */

void CommMosaic::borders()
{
  int i,j,n,dim;
  int nsend,nrecv,nlast,nn[3],ind;
  double **x;
  double *buf;
  MPI_Request request;
  AtomVec *avec = atom->avec;

  smax = rmax = 0;

  nlast = atom->nlocal;
  for (dim = 0; dim < n_neigh; dim++) {

    // find atoms within send bins 
    // check only local atoms 
    // store sent atom indices in sendlist for use in future timesteps

    x = atom->x;
    nsend = 0;

    for (i = 0; i < nlast; i++){
      ind = coords_to_bin(x[i],nn,nbp_loc,box_min);
      if (ind && (m_comm[nn[0]][nn[1]][nn[2]] & send_bit[dim])) {
        if (nsend == maxsendlist[dim]) grow_list(dim,nsend);
        sendlist[dim][nsend++] = i;
        //if (atom->tag[i] == 2177)
         // printf("SEND: me=%d; to=%d; x:%f %f %f; tag: %d; i=%d; nlocal=%d; pbc: %d - %d %d %d; nn: %d %d %d \n",me,sendproc[dim],x[i][0],x[i][1],x[i][2],atom->tag[i],i,atom->nlocal,pbc_flag[dim],pbc[dim][0],pbc[dim][1],pbc[dim][2],nn[0],nn[1],nn[2]);  
      }
    }

    // pack up list of border atoms   

    if (nsend*size_border > maxsend)
      grow_send(nsend*size_border,0);
    if (ghost_velocity)
      n = avec->pack_border_vel(nsend,sendlist[dim],buf_send,
                                pbc_flag[dim],pbc[dim]);
    else
      n = avec->pack_border(nsend,sendlist[dim],buf_send,
                            pbc_flag[dim],pbc[dim]);

    // swap atoms with other proc
    // no MPI calls except SendRecv if nsend/nrecv = 0
    // put incoming ghosts at end of my atom arrays

    MPI_Sendrecv(&nsend,1,MPI_INT,sendproc[dim],me,
                 &nrecv,1,MPI_INT,recvproc[dim],recvproc[dim],world,MPI_STATUS_IGNORE);
    if (nrecv*size_border > maxrecv) grow_recv(nrecv*size_border);
    if (nrecv) MPI_Irecv(buf_recv,nrecv*size_border,MPI_DOUBLE,
                         recvproc[dim],recvproc[dim],world,&request);
    if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[dim],me,world);
    if (nrecv) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    if (ghost_velocity)
      avec->unpack_border_vel(nrecv,atom->nlocal+atom->nghost,buf);
    else
      avec->unpack_border(nrecv,atom->nlocal+atom->nghost,buf);

    // set all pointers & counters

    smax = MAX(smax,nsend);
    rmax = MAX(rmax,nrecv);
    sendnum[dim] = nsend;
    recvnum[dim] = nrecv;
    size_forward_recv[dim] = nrecv*size_forward;
    size_reverse_send[dim] = nrecv*size_reverse;
    size_reverse_recv[dim] = nsend*size_reverse;
    firstrecv[dim] = atom->nlocal + atom->nghost;
    atom->nghost += nrecv;
  }

  // copy ghost particles within the same core if needed

  for (dim = 0; dim < n_own; dim++) {
    j = n_neigh + dim;
    x = atom->x;
    nlast = atom->nlocal + atom->nghost;
    nsend = 0;

    for (i = 0; i < nlast; i++){
      ind = coords_to_bin(x[i],nn,nbp_loc,box_min);
      if (ind && (m_comm[nn[0]][nn[1]][nn[2]] & own_bit[dim])) {
        if (nsend == maxsendlist[j]) grow_list(j,nsend);
        sendlist[j][nsend++] = i;
      }
    }

    if (nsend){

      // pack up list of border atoms

      if (nsend*size_border > maxsend) grow_send(nsend*size_border,0);
      if (ghost_velocity)
        n = avec->pack_border_vel(nsend,sendlist[j],buf_send,
                              pbc_flag[j],pbc[j]);
      else
        n = avec->pack_border(nsend,sendlist[j],buf_send,
                              pbc_flag[j],pbc[j]);

      nrecv = nsend;
      buf = buf_send;

      // unpack buffer

      if (ghost_velocity)
        avec->unpack_border_vel(nrecv,atom->nlocal+atom->nghost,buf);
      else
        avec->unpack_border(nrecv,atom->nlocal+atom->nghost,buf);

      // set all pointers & counters

      smax = MAX(smax,nsend);
      rmax = MAX(rmax,nrecv);
      sendnum[j] = nsend;
      recvnum[j] = nrecv;
      size_forward_recv[j] = nrecv*size_forward;
      size_reverse_send[j] = nrecv*size_reverse;
      size_reverse_recv[j] = nsend*size_reverse;
      firstrecv[j] = atom->nlocal + atom->nghost;
      atom->nghost += nrecv;
    }
  }

  // insure send/recv buffers are long enough for all forward & reverse comm

  int max = MAX(maxforward*smax,maxreverse*rmax);
  if (max > maxsend) grow_send(max,0);
  max = MAX(maxforward*rmax,maxreverse*smax);
  if (max > maxrecv) grow_recv(max);

  // reset global->local map

  if (map_style) atom->map_set();
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommMosaic::forward_comm_pair(Pair *pair)
{
  int iswap,n,dim;
  double *buf;
  MPI_Request request;

  int nsize = pair->comm_forward;

  for (iswap = 0; iswap < n_neigh; iswap++) {

    // pack buffer

    n = pair->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer


    if (recvnum[iswap])
      MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],recvproc[iswap],
                world,&request);
    if (sendnum[iswap])
      MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],me,world);
    if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    pair->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }

  for (dim = 0; dim < n_own; dim++) {
    iswap = n_neigh + dim;
    n = pair->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                buf_send,pbc_flag[iswap],pbc[iswap]);
    buf = buf_send;
    pair->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommMosaic::reverse_comm_pair(Pair *pair)
{
  int iswap,n,dim;
  double *buf;
  MPI_Request request;

  int nsize = MAX(pair->comm_reverse,pair->comm_reverse_off);

  for (dim = n_own-1; dim >= 0; dim--) {
    iswap = n_neigh + dim;
    n = pair->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);
    buf = buf_send;
    pair->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }

  for (iswap = n_neigh-1; iswap >= 0; iswap--) {

    // pack buffer

    n = pair->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendnum[iswap])
      MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],me,
                  world,&request);
    if (recvnum[iswap])
      MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world);
    if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    pair->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Fix
   size/nsize used only to set recv buffer limit
   size = 0 (default) -> use comm_forward from Fix
   size > 0 -> Fix passes max size per atom
   the latter is only useful if Fix does several comm modes,
     some are smaller than max stored in its comm_forward
------------------------------------------------------------------------- */

void CommMosaic::forward_comm_fix(Fix *fix, int size)
{
  int iswap,n,dim,nsize;
  double *buf;
  MPI_Request request;

  if (size) nsize = size;
  else nsize = fix->comm_forward;

  for (iswap = 0; iswap < n_neigh; iswap++) {

    // pack buffer

    n = fix->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                               buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (recvnum[iswap])
      MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],recvproc[iswap],
                world,&request);
    if (sendnum[iswap])
      MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],me,world);
    if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    fix->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }

  for (dim = 0; dim < n_own; dim++) {
    iswap = n_neigh + dim;
    n = fix->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                     buf_send,pbc_flag[iswap],pbc[iswap]);
    buf = buf_send;
    fix->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Fix
   size/nsize used only to set recv buffer limit
   size = 0 (default) -> use comm_forward from Fix
   size > 0 -> Fix passes max size per atom
   the latter is only useful if Fix does several comm modes,
     some are smaller than max stored in its comm_forward
------------------------------------------------------------------------- */

void CommMosaic::reverse_comm_fix(Fix *fix, int size)
{
  int iswap,n,dim,nsize;
  double *buf;
  MPI_Request request;

  if (size) nsize = size;
  else nsize = fix->comm_reverse;

  for (dim = n_own-1; dim >= 0; dim--) {
    iswap = n_neigh + dim;
    n = fix->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);
    buf = buf_send;
    fix->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }

  for (iswap = n_neigh-1; iswap >= 0; iswap--) {

    // pack buffer

    n = fix->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc

    if (sendnum[iswap])
      MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],me,
                world,&request);
    if (recvnum[iswap])
      MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world);
    if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    fix->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Fix with variable size data
   query fix for pack size to insure buf_send is big enough
   handshake sizes before each Irecv/Send to insure buf_recv is big enough
------------------------------------------------------------------------- */

void CommMosaic::reverse_comm_fix_variable(Fix *fix)
{
  int iswap,dim,nsend,nrecv;
  double *buf;
  MPI_Request request;

  for (dim = n_own-1; dim >= 0; dim--) {
    iswap = n_neigh + dim;
    nsend = fix->pack_reverse_comm_size(recvnum[iswap],firstrecv[iswap]);
    if (nsend > maxsend) grow_send(nsend,0);
    nsend = fix->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    buf = buf_send;
    fix->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }

  for (iswap = n_neigh-1; iswap >= 0; iswap--) {

    // pack buffer

    nsend = fix->pack_reverse_comm_size(recvnum[iswap],firstrecv[iswap]);
    if (nsend > maxsend) grow_send(nsend,0);
    nsend = fix->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc

    MPI_Sendrecv(&nsend,1,MPI_INT,recvproc[iswap],recvproc[iswap],
                 &nrecv,1,MPI_INT,sendproc[iswap],me,world,
                 MPI_STATUS_IGNORE);
    if (sendnum[iswap]) {
      if (nrecv > maxrecv) grow_recv(nrecv);
      MPI_Irecv(buf_recv,maxrecv,MPI_DOUBLE,sendproc[iswap],me,
                world,&request);
    }
    if (recvnum[iswap])
      MPI_Send(buf_send,nsend,MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world);
    if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    fix->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommMosaic::forward_comm_compute(Compute *compute)
{
  int iswap,n,dim;
  double *buf;
  MPI_Request request;

  int nsize = compute->comm_forward;

  for (iswap = 0; iswap < n_neigh; iswap++) {

    // pack buffer

    n = compute->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                   buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer


    if (recvnum[iswap])
      MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],recvproc[iswap],
                world,&request);
    if (sendnum[iswap])
      MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],me,world);
    if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;


    // unpack buffer

    compute->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }

  for (dim = 0; dim < n_own; dim++) {
    iswap = n_neigh + dim;
    n = compute->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                   buf_send,pbc_flag[iswap],pbc[iswap]);
    buf = buf_send;
    compute->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommMosaic::reverse_comm_compute(Compute *compute)
{
  int iswap,n,dim;
  double *buf;
  MPI_Request request;

  int nsize = compute->comm_reverse;

  for (dim = n_own-1; dim >= 0; dim--) {
    iswap = n_neigh + dim;
    n = compute->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);
    buf = buf_send;
    compute->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }

  for (iswap = n_neigh-1; iswap >= 0; iswap--) {

    // pack buffer

    n = compute->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendnum[iswap])
      MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],me,
                world,&request);
    if (recvnum[iswap])
      MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world);
    if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    compute->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommMosaic::forward_comm_dump(Dump *dump)
{
  int iswap,n,dim;
  double *buf;
  MPI_Request request;

  int nsize = dump->comm_forward;

  for (iswap = 0; iswap < n_neigh; iswap++) {

    // pack buffer

    n = dump->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (recvnum[iswap])
      MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],recvproc[iswap],
                world,&request);
    if (sendnum[iswap])
      MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],me,world);
    if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    dump->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }

  for (dim = 0; dim < n_own; dim++) {
    iswap = n_neigh + dim;
    n = dump->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                        buf_send,pbc_flag[iswap],pbc[iswap]);
    buf = buf_send;
    dump->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommMosaic::reverse_comm_dump(Dump *dump)
{
  int iswap,n,dim;
  double *buf;
  MPI_Request request;

  int nsize = dump->comm_reverse;

  for (dim = n_own-1; dim >= 0; dim--) {
    iswap = n_neigh + dim;
    n = dump->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);
    buf = buf_send;
    dump->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }

  for (iswap = n_neigh-1; iswap >= 0; iswap--) {

    // pack buffer

    n = dump->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendnum[iswap])
      MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],me,
                world,&request);
    if (recvnum[iswap])
      MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world);
    if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    dump->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication of N values in per-atom array
------------------------------------------------------------------------- */

void CommMosaic::forward_comm_array(int nsize, double **array)
{
  int i,j,k,m,iswap,last,dim;
  double *buf;
  MPI_Request request;

  // insure send/recv bufs are big enough for nsize
  // based on smax/rmax from most recent borders() invocation

  if (nsize > maxforward) {
    maxforward = nsize;
    if (maxforward*smax > maxsend) grow_send(maxforward*smax,0);
    if (maxforward*rmax > maxrecv) grow_recv(maxforward*rmax);
  }
    
  for (iswap = 0; iswap < n_neigh; iswap++) {

    // pack buffer

    m = 0;
    for (i = 0; i < sendnum[iswap]; i++) {
      j = sendlist[iswap][i];
      for (k = 0; k < nsize; k++)
        buf_send[m++] = array[j][k];
    }

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (recvnum[iswap])
      MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],recvproc[iswap],
                world,&request);
    if (sendnum[iswap])
      MPI_Send(buf_send,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],me,world);
    if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
    buf = buf_recv;

    // unpack buffer

    m = 0;
    last = firstrecv[iswap] + recvnum[iswap];
    for (i = firstrecv[iswap]; i < last; i++)
      for (k = 0; k < nsize; k++)
        array[i][k] = buf[m++];
  }

  for (dim = 0; dim < n_own; dim++) {
    iswap = n_neigh + dim;

    // pack buffer

    m = 0;
    for (i = 0; i < sendnum[iswap]; i++) {
      j = sendlist[iswap][i];
      for (k = 0; k < nsize; k++)
        buf_send[m++] = array[j][k];
    }

    buf = buf_send;

    // unpack buffer

    m = 0;
    last = firstrecv[iswap] + recvnum[iswap];
    for (i = firstrecv[iswap]; i < last; i++)
      for (k = 0; k < nsize; k++)
        array[i][k] = buf[m++];
  } 
}

/* ----------------------------------------------------------------------
   exchange info provided with all 6 stencil neighbors
------------------------------------------------------------------------- */

int CommMosaic::exchange_variable(int n, double *inbuf, double *&outbuf)
{
  int nsend,nrecv,nrecv1,dim;
  MPI_Request request;

  nrecv = n;
  if (nrecv > maxrecv) grow_recv(nrecv);
  memcpy(buf_recv,inbuf,nrecv*sizeof(double));

  // loop over dimensions

  for (dim = 0; dim < n_neigh; dim++) {

    // send/recv info to all neighbors

    nsend = nrecv;
    MPI_Sendrecv(&nsend,1,MPI_INT,sendproc[dim],me,
                 &nrecv1,1,MPI_INT,recvproc[dim],recvproc[dim],world,MPI_STATUS_IGNORE);
    nrecv += nrecv1;

    if (nrecv > maxrecv) grow_recv(nrecv);

    MPI_Irecv(&buf_recv[nsend],nrecv1,MPI_DOUBLE,recvproc[dim],recvproc[dim],
              world,&request);
    MPI_Send(buf_recv,nsend,MPI_DOUBLE,sendproc[dim],me,world);
    MPI_Wait(&request,MPI_STATUS_IGNORE);
  }

  outbuf = buf_recv;
  return nrecv;
}

/* ----------------------------------------------------------------------
   realloc the size of the send buffer as needed with BUFFACTOR and bufextra
   if flag = 1, realloc
   if flag = 0, don't need to realloc with copy, just free/malloc
------------------------------------------------------------------------- */

void CommMosaic::grow_send(int n, int flag)
{
  maxsend = static_cast<int> (BUFFACTOR * n);
  if (flag)
    memory->grow(buf_send,maxsend+bufextra,"comm:buf_send");
  else {
    memory->destroy(buf_send);
    memory->create(buf_send,maxsend+bufextra,"comm:buf_send");
  }
}

/* ----------------------------------------------------------------------
   free/malloc the size of the recv buffer as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommMosaic::grow_recv(int n)
{
  maxrecv = static_cast<int> (BUFFACTOR * n);
  memory->destroy(buf_recv);
  memory->create(buf_recv,maxrecv,"comm:buf_recv");
}

/* ----------------------------------------------------------------------
   realloc the size of the iswap sendlist as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommMosaic::grow_list(int iswap, int n)
{
  maxsendlist[iswap] = static_cast<int> (BUFFACTOR * n);
  memory->grow(sendlist[iswap],maxsendlist[iswap],"comm:sendlist[iswap]");
}

/* ----------------------------------------------------------------------
   realloc the buffers needed for swaps
------------------------------------------------------------------------- */

void CommMosaic::grow_swap(int n)
{
  free_swap();
  allocate_swap(n);
  
  sendlist = (int **)
    memory->srealloc(sendlist,n*sizeof(int *),"comm:sendlist");
  memory->grow(maxsendlist,n,"comm:maxsendlist");
  for (int i = maxswap; i < n; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i],BUFMIN,"comm:sendlist[i]");
  } 
  maxswap = n;
} 

/* ----------------------------------------------------------------------
   allocation of swap info
------------------------------------------------------------------------- */

void CommMosaic::allocate_swap(int n)
{
  memory->create(sendnum,n,"comm:sendnum");
  memory->create(recvnum,n,"comm:recvnum");
  memory->create(sendproc,n,"comm:sendproc");
  memory->create(recvproc,n,"comm:recvproc");
  memory->create(size_forward_recv,n,"comm:size");
  memory->create(size_reverse_send,n,"comm:size");
  memory->create(size_reverse_recv,n,"comm:size");
  memory->create(firstrecv,n,"comm:firstrecv");
  memory->create(pbc_flag,n,"comm:pbc_flag");
  memory->create(pbc,n,6,"comm:pbc");
}

/* ----------------------------------------------------------------------
   free memory for swaps
------------------------------------------------------------------------- */

void CommMosaic::free_swap()
{
  memory->destroy(sendnum);
  memory->destroy(recvnum);
  memory->destroy(sendproc);
  memory->destroy(recvproc);
  memory->destroy(size_forward_recv);
  memory->destroy(size_reverse_send);
  memory->destroy(size_reverse_recv);
  memory->destroy(firstrecv);
  memory->destroy(pbc_flag);
  memory->destroy(pbc);
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

bigint CommMosaic::memory_usage()
{
  bigint bytes = 0;
  for (int i = 0; i < nswap; i++)
    bytes += memory->usage(sendlist[i],maxsendlist[i]);
  bytes += memory->usage(buf_send,maxsend+bufextra);
  bytes += memory->usage(buf_recv,maxrecv);
  return bytes;
}

/* ---------------------------------------------------------------------- 
  read the binning partition file by the zeroth processor and 
  send necessary data to all cores
---------------------------------------------------------------------- */

void CommMosaic::read_user_file()
{
  int i, j, k, l, me_h, pr_tot, neigh_max, send_max, bnm, nn[3], ll[3];
  int **c_neigh, *nneighh, *dth;
  ifstream fr;
  string str;
  double *prd = domain->prd;
  char strh[128];

  bin_max = 100;
  neigh_max = 10;
  c_neigh = NULL;
  nneighh = dth = NULL;

  // read data by the zeroth core

  if (me == 0) {
    fr.open(customfile);
    if(fr.is_open()){
      fr >> pr_tot >> nbp[0] >> nbp[1] >> nbp[2];
      getline(fr,str);
      bnm = nbp[0]*nbp[1]*nbp[2];
      if (pr_tot != nprocs)
        error->one(FLERR,"Bad number of processors in user grid file!");

      memory->create(nbinh_orig,nprocs,"comm:nbinh_orig");
      memory->create(nneighh,nprocs,"comm:nneighh");
      memory->create(own_bin_orig,bin_max,nprocs,"comm:own_bin_orig");
      memory->create(c_neigh,neigh_max,nprocs,"comm:c_neigh");
      for (i = 0; i < nprocs; i++){
        fr >> me_h;
        if (me_h < 0 || me_h >= nprocs) { 
          sprintf(strh,"Problem with proc ID in mosaic partitioning file - id=%d with nprocs=%d!",me_h,nprocs);
          error->one(FLERR,strh);
        } 
        getline(fr,str);
        fr >> nneighh[me_h];
        if (nneighh[me_h] < 0 || nneighh[me_h] >= nprocs) { 
          sprintf(strh,"The number of neighbors for the core %d is not correct in mosaic partitioning file - nneigh=%d with nprocs=%d!",me_h,nneighh[me_h],nprocs);
          error->one(FLERR,strh);
        }
        if (nneighh[me_h] > neigh_max){
          neigh_max = nneighh[me_h] + EXT_ARRAY_SMALL;
          memory->grow(c_neigh,neigh_max,nprocs,"comm:c_neigh");
        }
        for (j = 0; j < nneighh[me_h]; j++){
          fr >> c_neigh[j][me_h];
          if (c_neigh[j][me_h] < 0 || c_neigh[j][me_h] >= nprocs || c_neigh[j][me_h] == me_h) {
            sprintf(strh,"The neighbor %d for the core %d is not correct in mosaic partitioning file - c_neigh=%d with nprocs=%d!",j,me_h,c_neigh[j][me_h],nprocs);
            error->one(FLERR,strh);
          }
        }
        getline(fr,str);
        fr >> nbinh_orig[me_h];
        if (nbinh_orig[me_h] <= 0 || nbinh_orig[me_h] > bnm) {
          sprintf(strh,"The number of bins for the core %d is not correct in mosaic partitioning file - nbinh=%d with tot_bins=%d!",me_h,nbinh_orig[me_h],bnm);
          error->one(FLERR,strh);
        }        
        if (nbinh_orig[me_h] > bin_max){
          bin_max = nbinh_orig[me_h] + EXT_ARRAY;
          memory->grow(own_bin_orig,bin_max,nprocs,"comm:own_bin_orig");
        }
        for (j = 0; j < nbinh_orig[me_h]; j++){
          fr >> own_bin_orig[j][me_h];
          if (own_bin_orig[j][me_h] < 0 || own_bin_orig[j][me_h] >= bnm) {
            sprintf(strh,"A bin ID number %d for the core %d is not correct in mosaic partitioning file - bin_id=%d with tot_bins=%d!",j,me_h,own_bin_orig[j][me_h],bnm);
            error->one(FLERR,strh);
          }
        }
        getline(fr,str);
      }
      fr.close();

      send_max = -1;
      for (i = 0; i < nprocs; i++){
        k = (nneighh[i]+1)*2 + nbinh_orig[i] + 1;
        for (j = 0; j < nneighh[i]; j++){
          me_h = c_neigh[j][i];
          k +=  nbinh_orig[me_h];
        }
        if (send_max < k)
          send_max = k;
      }

    } else
      error->one(FLERR,"Cannot open user grid file");
  }

  // send necessary data to all cores

  if (nprocs > 1){
    MPI_Bcast(&nbp[0],3,MPI_INT,0,world);
    MPI_Bcast(&send_max,1,MPI_INT,0,world);
  }

  memory->create(dth,send_max+1,"comm:dth");
  if (me > 0)
    MPI_Recv(dth,send_max,MPI_INT,0,me,world,MPI_STATUS_IGNORE);
  if (me == 0) {
    for (i = nprocs-1; i >= 0; i--){
      dth[0] = nneighh[i];
      dth[1] = i;
      dth[2] = nbinh_orig[i];
      for (j = 0; j < nneighh[i]; j++){
        k = c_neigh[j][i];
        dth[2*j + 3] = k;
        dth[2*j + 4] = nbinh_orig[k];
      }
      k = (nneighh[i]+1)*2 + 1;
      for (j = 0; j < nbinh_orig[i]; j++)
        dth[k + j] = own_bin_orig[j][i];
      k +=  nbinh_orig[i];
      for (j = 0; j < nneighh[i]; j++){
        me_h = c_neigh[j][i];
        for (l = 0; l < nbinh_orig[me_h]; l++)
          dth[k + l] = own_bin_orig[l][me_h];
        k +=  nbinh_orig[me_h];
      }

      if (i) MPI_Send(dth,send_max,MPI_INT,i,i,world);
    }
    memory->destroy(nbinh_orig);
    memory->destroy(nneighh);
    memory->destroy(c_neigh);
    memory->destroy(own_bin_orig);
    nbinh_orig = nneighh = NULL;
    c_neigh = own_bin_orig = NULL;
  }

  // unpack received data

  n_neigh_orig = dth[0] + 1;
  memory->create(my_neigh_orig,n_neigh_orig,"comm:my_neigh_orig");
  memory->create(nbinh_orig,n_neigh_orig,"comm:nbinh_orig");
  bin_max = -1;
  for (i = 0; i < n_neigh_orig; i++){
    my_neigh_orig[i] = dth[2*i + 1];
    nbinh_orig[i] = dth[2*i + 2];
    if (nbinh_orig[i] > bin_max)
      bin_max = nbinh_orig[i];
  }
  memory->create(own_bin_orig,n_neigh_orig,bin_max,"comm:own_bin_orig");
  k = n_neigh_orig*2 + 1;
  for (i = 0; i < n_neigh_orig; i++){
    for (j = 0; j < nbinh_orig[i]; j++)
      own_bin_orig[i][j] = dth[k + j];
    k += nbinh_orig[i];
  }
  memory->destroy(dth);
  dth = NULL;

  // temporary output
  ofstream fw;
  char  fname_wr[FILENAME_MAX];
  sprintf(fname_wr,"initial_out_%d.dat",me);
  fw.open(fname_wr);
  fw << nbp[0] << " " << nbp[1] << " " << nbp[2] << " \n";
  fw << n_neigh_orig << " ";
  for (i = 0; i < n_neigh_orig; i++)
    fw << my_neigh_orig[i] << " ";
  fw << " \n";
  for (i = 0; i < n_neigh_orig; i++){
    fw << my_neigh_orig[i] << " \n";
    fw << nbinh_orig[i] << " ";
    for (j = 0; j < nbinh_orig[i]; j++)
      fw << own_bin_orig[i][j] << " ";
    fw << " \n";
  }
  fw.close();
  // temporary output 

  // calculate min and max local bins

  for (i = 0; i < 3; i++){
    nbp_max[i] = -1;
    nbp_min[i] = 999999999;
  }
  for (i = 0; i < nbinh_orig[0]; i++){
    bin_coords(nn,own_bin_orig[0][i],nbp);
    for (j = 0; j < 3; j++){
      nbp_min[j] = MIN(nbp_min[j],nn[j]);
      nbp_max[j] = MAX(nbp_max[j],nn[j]);
    }
  }

  // calculate bin basic variables

  bin_diag = 0.0;
  for (j = 0; j < 3; j++){
    bin_size[j] = prd[j]/nbp[j];
    bin_size_sq[j] = bin_size[j]*bin_size[j];
    bin_size_inv[j] = 1.0/bin_size[j];
    nbp_orig[j] = nbp[j];
    bin_diag += bin_size[j]*bin_size[j];
    box_min[j] = domain->boxlo[j];
    nbp_loc[j] = nbp_max[j] - nbp_min[j] + 1;
  }
  bin_diag = sqrt(bin_diag);

  // temporary output 
  //printf("READ: me = %d: bin_size: %f %f %f; nbp_orig: %d %d %d; box_min: %f %f %f; nbp_min: %d %d %d; nbp_max: %d %d %d; bin_diag: %f \n",me,bin_size[0],bin_size[1],bin_size[2],nbp_orig[0],nbp_orig[1],nbp_orig[2],box_min[0],box_min[1],box_min[2],nbp_min[0],nbp_min[1],nbp_min[2],nbp_max[0],nbp_max[1],nbp_max[2],bin_diag);
  // temporary output 

  // create preliminary bit matrix for local particles

  memory->create(m_comm,nbp_loc[0],nbp_loc[1],nbp_loc[2],"comm:m_comm");

  for (i = 0; i < nbp_loc[0]; i++)
    for (j = 0; j < nbp_loc[1]; j++)
      for (k = 0; k < nbp_loc[2]; k++)
        m_comm[i][j][k] = 0;

  local_bit = 1 << 0;

  for (j = 0; j < nbinh_orig[0]; j++){
    bin_coords_local(nn,own_bin_orig[0][j],nbp);
    m_comm[nn[0]][nn[1]][nn[2]] |= local_bit;
  }

  // temporary output 
  sprintf(fname_wr,"initial_matrix_%d.plt",me);
  fw.open(fname_wr);
  fw << "VARIABLES=\"x\",\"y\",\"z\",\"ind\" \n";
  fw << "ZONE I=" << nbp_loc[0] << ", J=" << nbp_loc[1] << ", K=" << nbp_loc[2] << ", F=POINT \n";
  for (k = 0; k < nbp_loc[2]; k++)
    for (j = 0; j < nbp_loc[1]; j++)
      for (i = 0; i < nbp_loc[0]; i++){
        fw << i+nbp_min[0] << " " << j+nbp_min[1]  << " " << k+nbp_min[2]  << " ";
        if (m_comm[i][j][k] & local_bit)
          fw << "1 \n";
        else
          fw << "0 \n";
      }
  fw.close();
  // temporary output 
}

/* ----------------------------------------------------------------------
  process the bin data: 
  1) check whether some core neighbors can be deleted
  2) gather all updated neighbor lists to the zeroth core and 
     reorder them for a proper communication pattern 
  3) send ordered lists back to all cores, 
     reorder the bin arrays accordingly, and 
     extend the global binning
---------------------------------------------------------------------- */

void CommMosaic::process_bin_data(int &n_neigh_h, int &n_own_h, int *nbp_h, int *bin_ext_h, int *nbp_max_h, int *period_own_h, double *box_min_h, double *cut_h)
{
  int i, j, k, l, m, kb, kn, neigh_max, ind, ind1, ind2, n_min, n_max, k_max, ll[3], nn[3], pr[3];
  int *nneighh, *dth, *dth1, *p_ind;
  int *displs, *rcounts;
  double rr;

  n_neigh_h = n_neigh_orig;
  memory->create(my_neigh,n_neigh_h,"comm:my_neigh");
  memory->create(nbinh,n_neigh_h,"comm:nbinh");
  memory->create(own_bin,n_neigh_h,bin_max,"comm:own_bin");
  memory->create(twice_comm,n_neigh_h,"comm:twice_comm");
  for (i = 0; i < n_neigh_h; i++){
    my_neigh[i] = my_neigh_orig[i];
    nbinh[i] = nbinh_orig[i];
    twice_comm[i] = 0;
    for (j = 0; j < nbinh[i]; j++)
      own_bin[i][j] = own_bin_orig[i][j]; 
  }

  nneighh = dth = dth1 = p_ind = displs = rcounts = NULL;

  for (j = 0; j < 3; j++){
    period_own_h[j] = 0;
    bin_ext_h[j] = static_cast<int>(cut_h[j]*bin_size_inv[j]) + 1;
    box_min_h[j] -= bin_ext_h[j]*bin_size[j];
  }

  // temporary output 
  //printf("PROCESSED: me = %d: bin_ext_h: %d %d %d; box_min_h: %f %f %f \n",me,bin_ext_h[0],bin_ext_h[1],bin_ext_h[2],box_min_h[0],box_min_h[1],box_min_h[2]);
  // temporary output

  // check the neighbors, first myself for periodic BCs, then all others

  nn[0] = nn[1] = nn[2] = 0;
  for (i = 0; i < 3; i++)
    if (periodicity[i])
      for (j = 0; j < nbinh[0]; j++){
        for (k = j+1; k < nbinh[0]; k++){
          bin_dist_periodic(ll,own_bin[0][j],own_bin[0][k],i,nbp_h);
          if (((ll[0]-1)*bin_size[0] < cut_h[0]) && ((ll[1]-1)*bin_size[1] < cut_h[1]) && ((ll[2]-1)*bin_size[2] < cut_h[2])){
            rr = sqrt(ll[0]*ll[0]*bin_size_sq[0] + ll[1]*ll[1]*bin_size_sq[1] + ll[2]*ll[2]*bin_size_sq[2]);
            if (rr < bin_diag + cut_h[0]){
              nn[i] = 2;
              period_own_h[i] = 1;
              break;
            }
          }
        }
        if (nn[i]) break;
      }
  n_own_h = nn[0] + nn[1] + nn[2];

  if (n_own_h == 0)
    my_neigh[0] = -1;

  for (i = 1; i < n_neigh_h; i++){
    ind = ind1 = 0;
    for (j = 0; j < nbinh[0]; j++)
      for (k = 0; k < nbinh[i]; k++){
        bin_dist(ll,own_bin[0][j],own_bin[i][k],pr,nbp_h);
        if (((ll[0]-1)*bin_size[0] < cut_h[0]) && ((ll[1]-1)*bin_size[1] < cut_h[1]) && ((ll[2]-1)*bin_size[2] < cut_h[2])){
          rr = sqrt(ll[0]*ll[0]*bin_size_sq[0] + ll[1]*ll[1]*bin_size_sq[1] + ll[2]*ll[2]*bin_size_sq[2]);
          if (rr < bin_diag + cut_h[0]){
            if (pr[0] || pr[1] || pr[2]){
              for (m = 0; m < 3; m++)
                if (pr[m] && period_own_h[m] == 0)
                  ind1 = 1;
            } else
              ind = 1;
          }
        }
      }

    if (ind == 0 && ind1 == 0){
      my_neigh[i] = -1;
    } else{
      if (ind == 1 && ind1 == 1)
        twice_comm[i] = 1;
    }
  }

  neigh_max = n_neigh_h;
  memory->create(dth1,2*neigh_max,"comm:dth1");
  n_neigh_h = 0;
  for (i = 1; i <neigh_max; i++)
    if (my_neigh[i] > -1){
      dth1[n_neigh_h] = my_neigh[i];
      n_neigh_h++;
      if (twice_comm[i]){
        dth1[n_neigh_h] = my_neigh[i];
        n_neigh_h++;
      }
    }

  /*printf("00 me=%d neigh: %d - ",me,n_neigh_h);
  for (m = 0; m < n_neigh_h; m++)
    printf("%d ",dth1[m]);
  printf("\n");
  sleep(2);*/

  // temporary output 
  //printf("PROCESSED: me = %d: n_own_h: %d; n_neigh_h: %d; neigh_max: %d \n",me,n_own_h,n_neigh_h,neigh_max);
  // temporary output  

  // gather all updated neighbor lists to the zeroth core

  if (me == 0){
    memory->create(rcounts,nprocs,"comm:rcounts");
    memory->create(displs,nprocs,"comm:displs");
    memory->create(nneighh,nprocs,"comm:nneighh");
    memory->create(p_ind,nprocs,"comm:p_ind");
  }

  if (nprocs > 1)
    MPI_Gather(&n_neigh_h, 1, MPI_INT, rcounts, 1, MPI_INT, 0, world);

  int cc[nprocs];
  if (me == 0){
    n_max = 0;
    n_min = 999999;
    kn = 0;
    for (i = 0; i < nprocs; i++){
      p_ind[i] = -1;
      cc[i] = 0;
      displs[i] = kn;
      kn += rcounts[i];
      n_max = MAX(n_max,rcounts[i]);
      n_min = MIN(n_min,rcounts[i]);
    }
    memory->create(dth,kn,"comm:dth");
    l = 0;
    for (i = n_max; i >= n_min; i--)
      for (j = 0; j < nprocs; j++)
        if (rcounts[j] == i){
          nneighh[l] = j;
          l++;
        }

    if (l != nprocs)
      error->one(FLERR,"Something went wrong in sorting cores according to the number of neighbors!");
  }
  if (nprocs > 1)
    MPI_Gatherv(dth1, n_neigh_h, MPI_INT, dth, rcounts, displs, MPI_INT, 0, world);

  // reorder neighbors for a proper communication pattern

  if (me == 0){
    ind = 1;
    kb = 0;
    while (ind){
      for (j = 0; j < nprocs; j++){
        l = nneighh[j];
        if (cc[l] == rcounts[l])
          p_ind[l] = kb;
        if (p_ind[l] < kb){
          for (m = displs[l] + cc[l]; m < displs[l] + rcounts[l]; m++){
            ind1 = ind2 = 0;
            k = dth[m];
            if (p_ind[k] < kb) ind1 = 1;
            if (ind1)
              for (i = displs[k] + cc[k]; i < displs[k] + rcounts[k]; i++)
                if (dth[i] == l){
                  ind2 = 1;
                  break;
                }
            if (ind2){
              k_max = dth[m];
              dth[m] = dth[displs[l] + cc[l]];
              dth[displs[l] + cc[l]] = k_max;
              p_ind[l] = kb;
              cc[l]++;
              k_max = dth[i];
              dth[i] = dth[displs[k] + cc[k]];
              dth[displs[k] + cc[k]] = k_max;
              p_ind[k] = kb;
              cc[k]++;
              break;
            }
          }
          if (!ind2)
            p_ind[l] = kb;
        }
      }
      kb++;
      if (kb >= n_max){
        k_max = 0;
        for (j = 0; j < nprocs; j++)
          k_max += cc[j];
        if (k_max == kn) ind = 0;
      }
    }
  }

  // send ordered lists back to all cores and reorder the bin arrays accordingly

  if (nprocs > 1)
    MPI_Scatterv(dth, rcounts, displs, MPI_INT, dth1, n_neigh_h, MPI_INT, 0, world);

  if (me == 0){
    memory->destroy(dth);
    memory->destroy(rcounts);
    memory->destroy(displs);
    memory->destroy(nneighh);
    memory->destroy(p_ind);
    dth = rcounts = displs = nneighh = p_ind = NULL;
  }

  /*printf("me=%d here 00 %d %d %d \n",me,n_neigh_h,neigh_max,bin_max);
  sleep(6);

  printf("me=%d neigh: %d - ",me,n_neigh_h);
  for (m = 0; m < n_neigh_h; m++)
    printf("%d ",dth1[m]);
  printf("\n");
  sleep(2);

  printf("me=%d my_neigh: %d - ",me,neigh_max);
  for (m = 0; m < neigh_max; m++)
    printf("%d ",my_neigh[m]);
  printf("\n");
  sleep(2);

  printf("me=%d twice_comm: %d - ",me,neigh_max);
  for (m = 0; m < neigh_max; m++)
    printf("%d ",twice_comm[m]);
  printf("\n");
  sleep(2);*/

  int c_ind[2*neigh_max];
  memory->create(p_ind,bin_max,"comm:p_ind");
  memory->create(neigh_ind,n_neigh_h,"comm:neigh_ind");
  j = 1;
  k_max = 0;
  for (i = 0; i < n_neigh_h; i++){
    ind = 1;
    for (m = 0; m < k_max; m++)
      if (dth1[i] == c_ind[m]){
        ind = 0;
        break;
      }
    //printf("me=%d i=%d %d %d \n",me,i,j,ind);
    //sleep(2);

    if (ind){
      k = j;
      while (k < neigh_max){
        if (dth1[i] == my_neigh[k])
          break;
        k++;
      }
      if (k >= neigh_max)
        error->one(FLERR,"The neighbor has not been found in my_neigh array!");
      neigh_ind[i] = j;
      if (k > j){
        //printf("me=%d k=%d, %d %d, nj=%d, nk=%d \n",me,k,my_neigh[j],my_neigh[k],nbinh[j],nbinh[k]); 
        //sleep(2); 
        if (my_neigh[j] == -1){
          my_neigh[j] = my_neigh[k];
          nbinh[j] = nbinh[k];
          twice_comm[j] = twice_comm[k];
          for (m = 0; m < nbinh[k]; m++)
            own_bin[j][m] = own_bin[k][m];
          my_neigh[k] = -1;
          twice_comm[k] = 0;
        } else{
          for (m = 0; m < nbinh[j]; m++)
            p_ind[m] = own_bin[j][m];
          for (m = 0; m < nbinh[k]; m++)
            own_bin[j][m] = own_bin[k][m];
          for (m = 0; m < nbinh[j]; m++)
            own_bin[k][m] = p_ind[m];
          m = my_neigh[j];
          my_neigh[j] = my_neigh[k];
          my_neigh[k] = m;
          m = twice_comm[j];
          twice_comm[j] = twice_comm[k];
          twice_comm[k] = m;
          m = nbinh[j];
          nbinh[j] = nbinh[k];
          nbinh[k] = m;
        }
      }
      c_ind[k_max] = my_neigh[j];
      k_max++;
      j++;
    } else{
      ind1 = 1;
      for (m = 1; m <= j; m++)
        if (dth1[i] == my_neigh[m]){
          neigh_ind[i] = m;
          ind1 = 0;
          break;
        }
      if (ind1)
        error->one(FLERR,"The twice neighbor has not been found in my_neigh array!");
    }
  }
  neigh_max = j;
  memory->destroy(p_ind);
  memory->destroy(dth1);
  p_ind = dth1 = NULL;

  //printf("me=%d here 0 \n",me);
  //sleep(6);

  // extend the global binning

  for (j = 0; j < 3; j++){
    ll[j] = nbp_h[j];
    ll[j] += 2*bin_ext_h[j];
  }

  for (i = 0; i < neigh_max; i++)
    for (j = 0; j < nbinh[i]; j++){
      bin_coords(nn,own_bin[i][j],nbp_h);
      for (k = 0; k < 3; k++)
        nn[k] += bin_ext_h[k];
      own_bin[i][j] = nn[2]*ll[1]*ll[0] + nn[1]*ll[0] + nn[0];
    }

  for (j = 0; j < 3; j++){
    nbp_h[j] = ll[j];
    nbp_max_h[j] += 2*bin_ext_h[j];
  }

  //printf("me=%d here 1 \n",me);
  //sleep(3);

  // temporary output
  ofstream fw;
  char  fname_wr[FILENAME_MAX];
  sprintf(fname_wr,"processed_data_%d.dat",me);
  fw.open(fname_wr);
  fw << nbp_h[0] << " " << nbp_h[1] << " " << nbp_h[2] << " \n";
  fw << n_neigh_h + 1 << " ";
  fw << my_neigh[0] << " ";
  for (i = 0; i < n_neigh_h; i++)
    fw << my_neigh[neigh_ind[i]] << " ";
  fw << " \n";
  fw << my_neigh[0] << " \n";
  fw << nbinh[0] << " ";
  for (j = 0; j < nbinh[0]; j++)
    fw << own_bin[0][j] << " ";
  fw << " \n";
  for (i = 0; i < n_neigh_h; i++){
    fw << my_neigh[neigh_ind[i]] << " \n";
    fw << nbinh[neigh_ind[i]] << " ";
    for (j = 0; j < nbinh[neigh_ind[i]]; j++)
      fw << own_bin[neigh_ind[i]][j] << " ";
    fw << " \n";
  }
  fw.close();
  // temporary output 

  //printf("me=%d here 2 \n",me);
  //sleep(3);  
}

/* ---------------------------------------------------------------------- 
  build communication matrix which marks various bins: to send, to receive, to exchange
---------------------------------------------------------------------- */

void CommMosaic::build_comm_matrix(int n_neigh_h, bigint ***m_comm_h, bigint *send_bit_h, int *nbp_h, int *nbp_max_h, int *nbp_loc_h, int *bin_ext_h, int *period_own_h, double *cut_h, int *sendproc_h, int *recvproc_h, int *pbc_flag_h, int **pbc_h, int flag)
{
  int i, j, k, l, m, ii[3], ll[3], nn[3], pr[3], nl_min[3], nl_max[3], k1, k2, ind, ind1, ind2, k_max;
  bigint inv_mask;
  double rr;

  // create bit masks

  if (flag){
    for (i = 0; i < n_neigh_h; i++){
      send_bit_h[i] = (bigint)1 << i;
      k = n_neigh_h + i;
      exchange_bit[i] = (bigint)1 << k;
    }
    k = 2*n_neigh_h;
    local_bit = (bigint)1 << k;
    k++;
    for (i = 0; i < n_own; i++){
      j = k + i;
      own_bit[i] = (bigint)1 << j;
    }
  } else {
    for (i = 0; i < n_neigh_h; i++)
      send_bit_h[i] = (bigint)1 << i;
  }
  /*printf("BITS_TEST: me = %d; test_bit = %ld \n",me,(bigint)1 << 39);

  printf("BITS: me = %d; send_bit_h = ",me);
  for (i = 0; i < n_neigh_h; i++)
    printf("%ld ",send_bit_h[i]);
  printf("; exchange_bit = ");
  for (i = 0; i < n_neigh_h; i++)
    printf("%ld ",exchange_bit[i]);
  printf("; local_bit = %ld; ",local_bit);
  printf("own_bit = ");
  for (i = 0; i < n_own; i++)
    printf("%ld ",own_bit[i]);
    printf("\n");*/

  // assign local atom bits

  if (flag){
    for (j = 0; j < nbinh[0]; j++){
      bin_coords_local(nn,own_bin[0][j],nbp_h);
      m_comm_h[nn[0]][nn[1]][nn[2]] |= local_bit;
    }

    //printf("me=%d here 1 \n",me);
    //sleep(3); 

    // assign bits for sending atoms within the same core due to periodic BCs 

    if (my_neigh[0] > -1){
      l = 0;
      for (i = 0; i < 3; i++)
        if (periodicity[i]){
          if (l == n_own)
            break;
          ind = 0;
          k1 = n_neigh_h + l;
          k2 = k1 + 1;
          pbc_flag_h[k1] = 0;
          pbc_flag_h[k2] = 0;
          sendproc_h[k1] = recvproc_h[k1] = me;
          sendproc_h[k2] = recvproc_h[k2] = me;
          for (j = 0; j < 6; j++){
            pbc_h[k1][j] = 0;
            pbc_h[k2][j] = 0;
          }

          for (j = 0; j < nbinh[0]; j++){
            bin_coords(nn,own_bin[0][j],nbp_h);
            if (nn[i] > nbp_max_h[i] - 2*bin_ext_h[i]){

              for (ii[0] = nbp_min[0]; ii[0] <= nbp_max_h[0]; ii[0]++)
                for (ii[1] = nbp_min[1]; ii[1] <= nbp_max_h[1]; ii[1]++)
                  for (ii[2] = nbp_min[2]; ii[2] <= nbp_max_h[2]; ii[2]++)
                    if (ii[i] >= nbp_min[i] + bin_ext_h[i] && ii[i] < nbp_min[i] + 2*bin_ext_h[i]){
                      k = ii[2]*nbp_h[0]*nbp_h[1] + ii[1]*nbp_h[0] + ii[0];
                      bin_dist_periodic(ll,own_bin[0][j],k,i,nbp_h);
                      if (((ll[0]-1)*bin_size[0] < cut_h[0]) && ((ll[1]-1)*bin_size[1] < cut_h[1]) && ((ll[2]-1)*bin_size[2] < cut_h[2])){
                        rr = sqrt(ll[0]*ll[0]*bin_size_sq[0] + ll[1]*ll[1]*bin_size_sq[1] + ll[2]*ll[2]*bin_size_sq[2]);
                        if (rr < bin_diag + cut_h[0]){
                          /*for (m = 0; m < 3; m++)
                            if (ii[m] - nbp_min[m] < 0 || ii[m] - nbp_min[m] >= nbp_loc_h[m]) {
                              printf("me=%d: %d %d %d %d \n",me,ii[m],nbp_min[m],nbp_max_h[m],nbp_loc_h[m]);
                              sleep(2);
                            }
                          if (l >= n_own){
                            printf("0 me=%d: %d %d \n",me,l,n_own);
                            sleep(2);
                            }*/
                          m_comm_h[ii[0]-nbp_min[0]][ii[1]-nbp_min[1]][ii[2]-nbp_min[2]] |= own_bit[l];
                          pbc_flag_h[k1] = 1;
                          pbc_h[k1][i] = 1;
                          ind = 1;
                        }
                      }
                    }

            } else if (nn[i] < nbp_min[i] + 2*bin_ext_h[i]) {

              for (ii[0] = nbp_min[0]; ii[0] <= nbp_max_h[0]; ii[0]++)
                for (ii[1] = nbp_min[1]; ii[1] <= nbp_max_h[1]; ii[1]++)
                  for (ii[2] = nbp_min[2]; ii[2] <= nbp_max_h[2]; ii[2]++)
                    if (ii[i] > nbp_max_h[i] - 2*bin_ext_h[i] && ii[i] <= nbp_max_h[i] - bin_ext_h[i]){
                      k = ii[2]*nbp_h[0]*nbp_h[1] + ii[1]*nbp_h[0] + ii[0];
                      bin_dist_periodic(ll,own_bin[0][j],k,i,nbp_h);
                      if (((ll[0]-1)*bin_size[0] < cut_h[0]) && ((ll[1]-1)*bin_size[1] < cut_h[1]) && ((ll[2]-1)*bin_size[2] < cut_h[2])){
                        rr = sqrt(ll[0]*ll[0]*bin_size_sq[0] + ll[1]*ll[1]*bin_size_sq[1] + ll[2]*ll[2]*bin_size_sq[2]);
                        if (rr < bin_diag + cut_h[0]){
                          /*for (m = 0; m < 3; m++)
                            if (ii[m] - nbp_min[m] < 0 || ii[m] - nbp_min[m] >= nbp_loc_h[m]) { 
                              printf("me=%d: %d %d %d %d \n",me,ii[m],nbp_min[m],nbp_max_h[m],nbp_loc_h[m]);
                              sleep(2);
                            }
                          if (l+1 >= n_own){
                            printf("1 me=%d: %d %d \n",me,l+1,n_own);
                            sleep(2);
                            }*/
                          m_comm_h[ii[0]-nbp_min[0]][ii[1]-nbp_min[1]][ii[2]-nbp_min[2]] |= own_bit[l+1];
                          pbc_flag_h[k2] = 1;
                          pbc_h[k2][i] = -1;
                          ind = 1;
                        }
                      }
                    }

            }
          }
          if (ind) l += 2;
        }
      if (l != n_own)
        error->one(FLERR,"Something went wrong in marking own bits for the communication matrix!");

      ind2 = 0;
      for (i = 0; i < n_own; i++){
        l = n_neigh_h + i;
        inv_mask = own_bit[i] ^ ~(bigint)0;
        for (j = 0; j < nbinh[0]; j++){
          bin_coords_local(nn,own_bin[0][j],nbp_h);
          bin_coords(ll,own_bin[0][j],nbp_h);
          if (m_comm_h[nn[0]][nn[1]][nn[2]] & own_bit[i]){
            ind = 0;
            for (m = 0; m < 3; m++){
              ll[m] += pbc_h[l][m]*nbp_orig[m];
              if (ll[m] < nbp_min[m] || ll[m] > nbp_max_h[m]){
                ind = 1;
                break;
              }
            }
            if (ind){
              m_comm_h[nn[0]][nn[1]][nn[2]]  &= inv_mask;
              ind2 = 1;
            }
          }
        }
      }
      if (ind2) printf("CORRECTION OF MY_OWN COMMUNICATION MATRIX: proc = %d! \n",me);
    }
  }

  //printf("me=%d here 2 \n",me);
  //sleep(3); 

  // assign bits for the communication with neighbors

  int c_ind[n_neigh_h+1],cc[4],kk;
  int per_check = 0;
  ind2 = 0;
  k_max = 0;
  for (i = 0; i < n_neigh_h; i++){
    for (m = 0; m < 4; m++)
      cc[m] = 0;
    l = neigh_ind[i];
    pbc_flag_h[i] = 0;
    for (j = 0; j < 6; j++)
      pbc_h[i][j] = 0;
    sendproc_h[i] = my_neigh[l];
    recvproc_h[i] = my_neigh[l];

    ind = 0;
    for (m = 0; m < k_max; m++)
      if (my_neigh[l] == c_ind[m]){
        ind = 1;
        break;
      }
    if (twice_comm[l] && ind == 0){
      c_ind[k_max] = my_neigh[l];
      k_max++;
    }

    for (m = 0; m < 3; m++){
      nl_max[m] = -1;
      nl_min[m] = 999999999;
    }
    for (m = 0; m < nbinh[l]; m++){
      bin_coords(nn,own_bin[l][m],nbp_h);
      for (j = 0; j < 3; j++){
        nl_min[j] = MIN(nl_min[j],nn[j]);
        nl_max[j] = MAX(nl_max[j],nn[j]);
      }
    }
    for (m = 0; m < 3; m++){
      nl_min[m] -= bin_ext_h[m];
      nl_max[m] += bin_ext_h[m];
      //if (nl_min[m] < 0 || nl_max[m] >= nbp_h[m]) 
      //  printf("PROBLEM HERE: me=%d; to=%d; m=%d; nl: %d %d \n",me,sendproc_h[i],m,nl_min[m],nl_max[m]);
    }

    for (j = 0; j < nbinh[0]; j++)
      for (k = 0; k < nbinh[l]; k++){
        bin_dist(ll,own_bin[0][j],own_bin[l][k],pr,nbp_h);
        if (((ll[0]-1)*bin_size[0] < cut_h[0]) && ((ll[1]-1)*bin_size[1] < cut_h[1]) && ((ll[2]-1)*bin_size[2] < cut_h[2])){
          rr = sqrt(ll[0]*ll[0]*bin_size_sq[0] + ll[1]*ll[1]*bin_size_sq[1] + ll[2]*ll[2]*bin_size_sq[2]);
          if (rr < bin_diag + cut_h[0]){
            bin_coords_local(nn,own_bin[0][j],nbp_h);
            bin_coords_local(ll,own_bin[l][k],nbp_h);
            if (twice_comm[l]){
              if (ind){
                if (pr[0] || pr[1] || pr[2]){
                  ind1 = 0;
                  for (m = 0; m < 3; m++)
                    if (pr[m] && period_own_h[m] == 0)
                      ind1 = 1;
                  if (ind1){
                    kk = 0;
                    m_comm_h[nn[0]][nn[1]][nn[2]] |= send_bit_h[i];
                    for (m = 0; m < 3; m++)
                      if (pr[m] && period_own_h[m] == 0){
                        kk++;
                        pbc_flag_h[i] = 1;
                        if (ll[m] < nn[m]){
                          pbc_h[i][m] = -1;
                          ll[m] += nbp_orig[m];
                        } else {
                          pbc_h[i][m] = 1;
                          ll[m] -= nbp_orig[m];
                        }
                      }
                    if (flag) m_comm_h[ll[0]][ll[1]][ll[2]] |= exchange_bit[i];
                    cc[kk] = 1;
                  }
                }
              } else{
                if (pr[0] == 0 && pr[1] == 0 && pr[2] == 0){
                  m_comm_h[nn[0]][nn[1]][nn[2]] |= send_bit_h[i];
                  if (flag) m_comm_h[ll[0]][ll[1]][ll[2]] |= exchange_bit[i];
                }
              }
            } else{
              m_comm_h[nn[0]][nn[1]][nn[2]] |= send_bit_h[i];
              kk = 0;
              for (m = 0; m < 3; m++)
                if (pr[m] && period_own_h[m] == 0){
                  kk++;
                  pbc_flag_h[i] = 1;
                  if (ll[m] < nn[m]){
                    pbc_h[i][m] = -1;
                    ll[m] += nbp_orig[m];
                  } else {
                    pbc_h[i][m] = 1;
                    ll[m] -= nbp_orig[m];
                  }
                }
              if (flag) m_comm_h[ll[0]][ll[1]][ll[2]] |= exchange_bit[i];
              cc[kk] = 1;
            }
          }
        }
      }

    inv_mask = send_bit_h[i] ^ ~(bigint)0;
    for (j = 0; j < nbinh[0]; j++){
      bin_coords_local(nn,own_bin[0][j],nbp_h);
      bin_coords(ll,own_bin[0][j],nbp_h);
      if (m_comm_h[nn[0]][nn[1]][nn[2]] & send_bit_h[i]){
        ind = 0;
        for (m = 0; m < 3; m++){
          ll[m] += pbc_h[i][m]*nbp_orig[m];
          if (ll[m] < nl_min[m] || ll[m] > nl_max[m]){
            ind = 1;
            break;
          }
        }
        if (ind){
          m_comm_h[nn[0]][nn[1]][nn[2]]  &= inv_mask;
          ind2 = 1;
        }
      }
    }

    kk = 0;
    for (m = 0; m < 4; m++)
      kk += cc[m];
    if (kk > 1) {
      per_check = 1;
      //printf("per_check me=%d, i=%d, pbc_h=%d %d %d %d \n",me,i,pbc_flag_h[i],pbc_h[i][0],pbc_h[i][1],pbc_h[i][2]);
    }
  }

  if (ind2) printf("CORRECTION OF NEIGHBOR COMMUNICATION MATRIX: proc = %d! \n",me);

  //printf("me=%d here 3 \n",me);
  //sleep(3); 

  if (per_check) error->warning(FLERR,"The simulation may not function properly due to complicated periodic BCs!!!");

  // temporary output 
  //printf("PERIODIC: me = %d: nneigh=%d; ",me,n_neigh_h);
  //for (l = 0; l < n_neigh_h; l++)
  //  printf("to=%d pbc_h: %d - %d %d %d; ",sendproc_h[l],pbc_flag_h[l],pbc_h[l][0],pbc_h[l][1],pbc_h[l][2]);
  //printf(" \n");

  //printf("BUILD: me = %d: cut_h: %f %f %f; nbp_min: %d %d %d; nbp_max_h: %d %d %d \n",me,cut_h[0],cut_h[1],cut_h[2],nbp_min[0],nbp_min[1],nbp_min[2],nbp_max_h[0],nbp_max_h[1],nbp_max_h[2]);
  ofstream fw;
  char  fname_wr[FILENAME_MAX];

  if (flag){
    sprintf(fname_wr,"local_matrix_%d.plt",me);
    fw.open(fname_wr);
    fw << "VARIABLES=\"x\",\"y\",\"z\",\"ind\" \n";
    fw << "ZONE I=" << nbp_loc_h[0] << ", J=" << nbp_loc_h[1] << ", K=" << nbp_loc_h[2] << ", F=POINT \n";
    for (k = 0; k < nbp_loc_h[2]; k++)
      for (j = 0; j < nbp_loc_h[1]; j++)
        for (i = 0; i < nbp_loc_h[0]; i++){
          fw << i+nbp_min[0] << " " << j+nbp_min[1]  << " " << k+nbp_min[2]  << " ";
          if (m_comm_h[i][j][k] & local_bit)
            fw << "1 \n";
          else
            fw << "0 \n";
        }
    fw.close();
  }

  if (n_neigh_h > 0){
    sprintf(fname_wr,"neigh_matrix_%d.plt",me);
    fw.open(fname_wr);
    fw << "VARIABLES=\"x\",\"y\",\"z\"";
    for (l = 0; l < n_neigh_h; l++)
      fw << ",\"s" << l << "\"";
    for (l = 0; l < n_neigh_h; l++)
      fw << ",\"e" << l << "\"";
    fw << " \n";
    fw << "ZONE I=" << nbp_loc_h[0] << ", J=" << nbp_loc_h[1] << ", K=" << nbp_loc_h[2] << ", F=POINT \n";
    for (k = 0; k < nbp_loc_h[2]; k++)
      for (j = 0; j < nbp_loc_h[1]; j++)
        for (i = 0; i < nbp_loc_h[0]; i++){
          fw << i+nbp_min[0] << " " << j+nbp_min[1]  << " " << k+nbp_min[2]  << " ";
          for (l = 0; l < n_neigh_h; l++){
            if (m_comm_h[i][j][k] & send_bit_h[l])
              fw << "1 ";
            else
              fw << "0 ";
          }
          if (flag)
            for (l = 0; l < n_neigh_h; l++){
              if (m_comm_h[i][j][k] & exchange_bit[l])
                fw << "1 ";
              else
                fw << "0 ";
            }
            fw << " \n";
        }
    fw.close();
  }

  if (flag)
    if (n_own > 0){
      sprintf(fname_wr,"own_matrix_%d.plt",me);
      fw.open(fname_wr);
      fw << "VARIABLES=\"x\",\"y\",\"z\"";
      for (l = 0; l < n_own; l++)
        fw << ",\"o" << l << "\"";
      fw << " \n";
      fw << "ZONE I=" << nbp_loc_h[0] << ", J=" << nbp_loc_h[1] << ", K=" << nbp_loc_h[2] << ", F=POINT \n";
      for (k = 0; k < nbp_loc_h[2]; k++)
        for (j = 0; j < nbp_loc_h[1]; j++)
          for (i = 0; i < nbp_loc_h[0]; i++){
            fw << i+nbp_min[0] << " " << j+nbp_min[1]  << " " << k+nbp_min[2]  << " ";
            for (l = 0; l < n_own; l++){
              if (m_comm_h[i][j][k] & own_bit[l])
                fw << "1 ";
              else
                fw << "0 ";
            }
            fw << " \n";
          }
      fw.close();
    }
  // temporary output

  if (flag) processed_data_ind = 1;

  memory->destroy(own_bin);
  memory->destroy(nbinh);
  memory->destroy(my_neigh);
  memory->destroy(twice_comm);
  memory->destroy(neigh_ind);
  own_bin = NULL;
  nbinh = my_neigh = twice_comm = neigh_ind = NULL;
}

/* ---------------------------------------------------------------------- 
  find the distance (ll[0],ll[1],ll[2]) between bins b1 and b2
---------------------------------------------------------------------- */

void CommMosaic::bin_dist(int *ll, int &b1, int &b2, int *pr, int *nbp_h)
{
  int i, n1[3], n2[3];

  bin_coords(n1,b1,nbp_h);
  bin_coords(n2,b2,nbp_h);

  for (i = 0; i < 3; i++){
    pr[i] = 0;
    ll[i] = abs(n1[i]-n2[i]);
    if (periodicity[i] && ll[i] > 0.5*nbp_orig[i]){
      ll[i] = nbp_orig[i] - ll[i];
      pr[i] = 1;
    }
  }
}

/* ---------------------------------------------------------------------- 
  find the distance (ll[0],ll[1],ll[2]) between bins b1 and b2
  used only for processors which may have periodicity with themselves
---------------------------------------------------------------------- */

void CommMosaic::bin_dist_periodic(int *ll, int &b1, int &b2, int prd_ind, int *nbp_h)
{
  int i, n1[3], n2[3], prd[3];

  for (i = 0; i < 3; i++)
    prd[i] = 0;
  prd[prd_ind] = 1;

  bin_coords(n1,b1,nbp_h);
  bin_coords(n2,b2,nbp_h);

  for (i = 0; i < 3; i++){
    ll[i] = abs(n1[i]-n2[i]);
    if (periodicity[i] && prd[i])
      ll[i] = nbp_orig[i] - ll[i];
  }
}

/* ---------------------------------------------------------------------- 
  convert the bin number bn to bin coordinates (nn[0],nn[1],nn[2])
---------------------------------------------------------------------- */

void CommMosaic::bin_coords(int *nn, int &bn, int *nbp_h)
{
  int i;

  nn[2] = static_cast<int>(bn/nbp_h[0]/nbp_h[1]);
  i = bn - nn[2]*nbp_h[0]*nbp_h[1];
  nn[1] = static_cast<int>(i/nbp_h[0]);
  nn[0] = i - nn[1]*nbp_h[0];
}

/* ---------------------------------------------------------------------- 
  convert the bin number bn to local bin coordinates (nn[0],nn[1],nn[2])
---------------------------------------------------------------------- */

void CommMosaic::bin_coords_local(int *nn, int &bn, int *nbp_h)
{
  int i;

  bin_coords(nn,bn,nbp_h);
  for (i = 0; i < 3; i++)
    nn[i] -= nbp_min[i];
}

/* ----------------------------------------------------------------------
  convert coordinates to local bin coordinates (nn[0],nn[1],nn[2])
---------------------------------------------------------------------- */

int CommMosaic::coords_to_bin(double *xx, int *nn, int *nbp_loc_h, double *box_min_h)
{
  int i,ind;

  ind = 1;
  for (i = 0; i < 3; i++){
    nn[i] = static_cast<int>((xx[i] - box_min_h[i])*bin_size_inv[i]) - nbp_min[i];
    if (nn[i] < 0 || nn[i] >= nbp_loc_h[i]){
      ind = 0;
      break;
    }
  }
  return ind;
}

/* ----------------------------------------------------------------------
  convert coordinates to local bin coordinates (nn[0],nn[1],nn[2]) and 
  check for periodicity
---------------------------------------------------------------------- */

int CommMosaic::coords_to_bin_exchange(double *xx, int *nn)
{
  int i,ind;

  ind = 1;
  for (i = 0; i < 3; i++){
    nn[i] = static_cast<int>((xx[i] - box_min[i])*bin_size_inv[i]);
    if (periodicity[i]){
      if (nn[i] > nbp_max[i])
        nn[i] -= nbp_orig[i];
      if (nn[i] < nbp_min[i])
        nn[i] += nbp_orig[i];
    }
    nn[i] -= nbp_min[i];
    if (nn[i] < 0 || nn[i] >= nbp_loc[i]){
      ind = 0;
      break;
    }
  }
  return ind;
}

/* ---------------------------------------------------------------------- */

