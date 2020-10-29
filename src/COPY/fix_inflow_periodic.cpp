/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "fix_inflow_periodic.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "group.h"
#include "error.h"
#include "memory.h"
#include "update.h"
#include "neighbor.h"
#include "domain.h"
#include "modify.h"

#define  NUM 1000
#define  EPS 1e-6
#define  BFF 50

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixInflowPeriodic::FixInflowPeriodic(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  int i,j,l,igroup,nms;
  double *rdata;
  char grp[50];
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  FILE *f_read;

  bc_send_comm = bc_recv_comm = NULL;
  bc_send_comm_rank = bc_send_comm_size = bc_recv_comm_rank = bc_recv_comm_size = NULL;
  rdata = NULL;
  rot = NULL;
  x0 = aa = NULL;
  send_exist = send_new = NULL;
  list_new = sp_list = send_counts = NULL;
  rcounts = displs = buf_send_back = sp_recv = NULL;
  buf_send = buf_recv = NULL;
  send_partner = recv_partner = NULL;
  xp_save = NULL;

  nms = NUM;
  groupbit_reflect = 0;
  exist_max = new_max = 0; 
  new_list_max = 0;
  send_max = recv_max = 0;
  sp_recv_max = send_back_max = 0;
  sp_max = BFF;
  part_size = 9;
  spin_ind = 0;
  save_max = 0;

  if (!(strcmp(atom->atom_style,"molecular") == 0) && !(strcmp(atom->atom_style,"sdpd") == 0))
    error->all(FLERR,"The fix_inflow_periodic can only be used with atom_style molecular and sdpd!");

  if (strcmp(atom->atom_style,"sdpd") == 0){
    spin_ind = 1;
    part_size += 3;
  }
  part_size_track = part_size - 1;
  part_size_vel = part_size - 5;

  if (narg != 4) error->all(FLERR,"Illegal fix_inflow_periodic command");
  sprintf(fname,arg[3]);
  memory->create(rdata,nms,"fix_inflow_periodic:rdata");

  if (comm->me == 0){  // read the data by the root
    l = 0;
    f_read = fopen(fname,"r");
    if (f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open input boundary file from fix_inflow_periodic.");
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&num_inflow);
    rdata[l++] = static_cast<double> (num_inflow);
    //rdata[l++] = ubuf(num_inflow).d;
    if (num_inflow <= 0)
      error->one(FLERR,"The number of inflows in fix_inflow_periodic cannot be a non-positive number!!"); 

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %s",&binsize,&skin,&grp[0]);
    rdata[l++] = binsize;
    rdata[l++] = skin;
    igroup = group->find(grp);
    if (igroup == -1) error->one(FLERR,"Group ID for group_reflect does not exist");
    groupbit_reflect = group->bitmask[igroup];
    rdata[l++] = static_cast<double> (groupbit_reflect);
    //rdata[l++] = ubuf(groupbit_reflect).d;

    for (i = 0; i < num_inflow; ++i){
      fgets(buf,BUFSIZ,f_read);
      if (l + 7 > nms){
        nms += NUM;
        memory->grow(rdata,nms,"fix_inflow_periodic:rdata");
      }
      sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6]);
      l += 7; 
    }
    fclose(f_read);
  }

  MPI_Bcast(&l,1,MPI_INT,0,world);
  if (l > NUM)
    memory->grow(rdata,l,"fix_inflow_periodic:rdata");
  MPI_Bcast(rdata,l,MPI_DOUBLE,0,world);  // send the read data to other cores

  l = 0;
  num_inflow = static_cast<int> (rdata[l++]);
  //num_inflow = (int) ubuf(rdata[l++]).i;

  binsize = rdata[l++];
  skin = rdata[l++]; 
  groupbit_reflect = static_cast<int> (rdata[l++]);
  //groupbit_reflect = (int) ubuf(rdata[l++]).i;

  memory->create(x0,num_inflow,3,"fix_inflow_periodic:x0");       // allocate memory for all arrays
  memory->create(aa,num_inflow,5,"fix_inflow_periodic:aa");
  memory->create(rot,num_inflow,3,3,"fix_inflow_periodic:rot");
  memory->create(bc_send_comm,num_inflow,"fix_inflow_periodic:bc_send_comm");
  memory->create(bc_recv_comm,num_inflow,"fix_inflow_periodic:bc_recv_comm");
  memory->create(bc_send_comm_rank,num_inflow,"fix_inflow_periodic:bc_send_comm_rank");
  memory->create(bc_send_comm_size,num_inflow,"fix_inflow_periodic:bc_send_comm_size");
  memory->create(bc_recv_comm_rank,num_inflow,"fix_inflow_periodic:bc_recv_comm_rank");
  memory->create(bc_recv_comm_size,num_inflow,"fix_inflow_periodic:bc_recv_comm_size"); 
  memory->create(send_partner,num_inflow,"fix_inflow_periodic:send_partner");
  memory->create(recv_partner,num_inflow,"fix_inflow_periodic:recv_partner");   

  for (i = 0; i < num_inflow; ++i){
    send_partner[i] = -1;
    recv_partner[i] = -1;
    for (j = 0; j < 3; ++j){
      x0[i][j] = rdata[l+j];
      aa[i][j] = rdata[l+j+3] - rdata[l+j];
    }
    aa[i][3] = rdata[l+6]; 
    setup_rot(i);
    l += 7;
  }

  memory->destroy(rdata);
}

/* ---------------------------------------------------------------------- */

FixInflowPeriodic::~FixInflowPeriodic()
{
  memory->destroy(rot);
  memory->destroy(x0);
  memory->destroy(aa); 
  memory->destroy(bc_send_comm);
  memory->destroy(bc_recv_comm);
  memory->destroy(bc_send_comm_rank);
  memory->destroy(bc_send_comm_size);
  memory->destroy(bc_recv_comm_rank);
  memory->destroy(bc_recv_comm_size); 
  memory->destroy(send_exist);
  memory->destroy(send_new);
  memory->destroy(list_new);
  memory->destroy(sp_list);
  memory->destroy(send_counts);
  memory->destroy(buf_send);
  memory->destroy(buf_recv);
  memory->destroy(rcounts);
  memory->destroy(displs);
  memory->destroy(buf_send_back);
  memory->destroy(sp_recv);
  memory->destroy(send_partner);
  memory->destroy(recv_partner); 
  memory->destroy(xp_save);
}

/* ---------------------------------------------------------------------- */

int FixInflowPeriodic::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */


void FixInflowPeriodic::setup(int vflag)
{
  int i,ifix,ifix0;

  if (neighbor->every > 1 || neighbor->delay > 1) 
    error->all(FLERR,"Reneighboring should be done every time step when using fix_inflow_periodic!");

  ifix0 = modify->find_fix("inflow/periodic");
  ifix = modify->find_fix("solid/bound");
  if (ifix > -1 && ifix0 < ifix) error->warning(FLERR,"Fix inflow/periodic should normally come after the fix solid/bound! Something may go wrong!");
  ifix = modify->find_fix("force/bound");
  if (ifix > -1 && ifix0 < ifix) error->warning(FLERR,"Fix inflow/periodic should normally come after the fix force/bound! Something may go wrong!");   

  cut = neighbor->cutneighmax;
  for (i = 0; i < 3; ++i){
    subhi[i] = domain->subhi[i];
    sublo[i] = domain->sublo[i];
  }

  if (binsize <= 0.0)
    binsize = 0.5*cut;

  set_comm_groups();   // setup communication groups
}

/* ---------------------------------------------------------------------- */

void FixInflowPeriodic::post_integrate()
{
  int i,j,k,l,ii,kk,offset,ind,st;
  int l_e, l_n, l_loc, l_en, ll[3], nn[3];
  double dl[3],xp[3],xh[3],vh[3];
  double tt,rr;
  double **x = atom->x;
  double **v = atom->v;
  double **omega = atom->omega;
  int *sp = atom->spin;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double lm = cut+skin;
  bigint ntemp;

  // reflections of particles at both surfaces if needed

  for (ii = 0; ii < num_inflow; ++ii){

    if (bc_send_comm_rank[ii] > -1)   // reflections at the end of a cylindrical cap
      for (i = 0; i < nlocal; ++i)
        if (mask[i] & groupbit_reflect)
          if (sp[i] < -1){
            for (j = 0; j < 3; ++j){
              xh[j] = x[i][j] - x0[ii][j];
              xp[j] = xp_save[i][j] - x0[ii][j];
              vh[j] = v[i][j];
            }
            rot_forward(xh[0],xh[1],xh[2],ii);
            rot_forward(xp[0],xp[1],xp[2],ii);
            rot_forward(vh[0],vh[1],vh[2],ii);
            rr = xh[0]*xh[0] + xh[1]*xh[1];  
            if (xh[2]-aa[ii][4] < 0.0 && xp[2]-aa[ii][4] > 0.0 && rr < aa[ii][3]*aa[ii][3]){
              if (vh[2] < 0.0)
                vh[2] = -vh[2];
              tt = vh[0]*xh[0] + vh[1]*xh[1];
              if (tt > 0.0){
                vh[0] = -vh[0];
                vh[1] = -vh[1];
              }  
              rot_back(vh[0],vh[1],vh[2],ii);
              for (j = 0; j < 3; ++j){
                x[i][j] = xp_save[i][j];
                v[i][j] = vh[j];
	      }             
	    }
	  }
      
    if (bc_recv_comm_rank[ii] > -1) // reflections at the beginning of a cylindrical cap
      for (i = 0; i < nlocal; ++i)
        if (mask[i] & groupbit_reflect)
          if (sp[i] < -1){
            for (j = 0; j < 3; ++j){
              xh[j] = x[i][j] - x0[ii][j];
              xp[j] = xp_save[i][j] - x0[ii][j];
              vh[j] = v[i][j];
            }
            rot_forward(xh[0],xh[1],xh[2],ii);
            rot_forward(xp[0],xp[1],xp[2],ii);
            rot_forward(vh[0],vh[1],vh[2],ii);
            rr = xh[0]*xh[0] + xh[1]*xh[1];
            if (xh[2] < 0.0 && xp[2] > 0.0 && rr < aa[ii][3]*aa[ii][3]){
              if (vh[2] < 0.0)
                vh[2] = -vh[2];
              tt = vh[0]*xh[0] + vh[1]*xh[1];
              if (tt > 0.0){
                vh[0] = -vh[0];
                vh[1] = -vh[1];
              }
              rot_back(vh[0],vh[1],vh[2],ii);
              for (j = 0; j < 3; ++j){
                x[i][j] = xp_save[i][j];
                v[i][j] = vh[j];
              }
            }
          }
  }

  // exchange of data between beginnings and ends of periodical sections
  for (ii = 0; ii < num_inflow; ++ii){

    if (bc_send_comm_rank[ii] > -1){  // only cores which participate in sending the data
      l_e = l_n = l_loc = l_en = 0;
      for (i = 0; i < nlocal; ++i)
        if (mask[i] & groupbit) { 
          for(j = 0; j < 3; ++j){
            xh[j] = x[i][j] - x0[ii][j];
            xp[j] = xp_save[i][j] - x0[ii][j];
          }
          rot_forward(xh[0],xh[1],xh[2],ii);
          rot_forward(xp[0],xp[1],xp[2],ii);

          xh[2] -= aa[ii][4];
          xp[2] -= aa[ii][4]; 
          rr = xh[0]*xh[0] + xh[1]*xh[1];
          if (l_e + BFF >= exist_max){   // grow sending arrays if needed
            exist_max += NUM;
            memory->grow(send_exist,exist_max,"fix_inflow_periodic:send_exist"); 
          }
          if (l_n + BFF >= new_max){     // grow sending arrays if needed
            new_max += NUM;
            memory->grow(send_new,new_max,"fix_inflow_periodic:send_new"); 
          }          
          if (rr < aa[ii][3]*aa[ii][3] && xh[2] >= 0.0 && xh[2] < lm){      // check if a particle is in the sending cylinder 
            if (sp[i] > -1){          // already tracked - track further 
              send_exist[l_e++] = static_cast<double> (sp[i]);  
              //send_exist[l_e++] = ubuf(sp[i]).d;
              send_exist[l_e++] = 1.0;
              for (j = 0; j < 3; ++j)
                send_exist[l_e++] = x[i][j] - aa[ii][j];
              for (j = 0; j < 3; ++j)
                send_exist[l_e++] = v[i][j];
              if (spin_ind)
                for (j = 0; j < 3; ++j)
                  send_exist[l_e++] = omega[i][j];  
              l_en++; 
            } else {
              if (xp[2] < 0.0){       // new particle - need to track 
                send_new[l_n++] = 0.0;
                send_new[l_n++] = static_cast<double> (type[i]);
                //send_new[l_n++] = ubuf(type[i]).d;
                send_new[l_n++] = static_cast<double> (mask[i]);
                //send_new[l_n++] = ubuf(mask[i]).d;
                for (j = 0; j < 3; ++j)
                  send_new[l_n++] = x[i][j] - aa[ii][j];
                for (j = 0; j < 3; ++j)
                  send_new[l_n++] = v[i][j]; 
                if (spin_ind)
                  for (j = 0; j < 3; ++j)
                    send_new[l_n++] = omega[i][j]; 
  
                if (l_loc >= new_list_max){    // grow array of local IDs for returning valid tracking IDs for new particles
                  new_list_max += BFF;
                  memory->grow(list_new,new_list_max,"fix_inflow_periodic:list_new"); 
		}
                list_new[l_loc++] = i;
	      } 
            }                        
	  } else{
            if (sp[i] > -1){
              if (xh[2] < 0.0){        // the particle went back - need to remove it at the receiving partner
                send_exist[l_e++] = static_cast<double> (sp[i]);
                //send_exist[l_e++] = ubuf(sp[i]).d; 
                send_exist[l_e++] = -1.0;
                sp[i] = -2;
	      } else{                  // stop particle tracking 
                send_exist[l_e++] = static_cast<double> (sp[i]);
                //send_exist[l_e++] = ubuf(sp[i]).d; 
                send_exist[l_e++] = -2.0;
                sp[i] = -2;               
	      }
              l_en++;    
            }
	  }
	}

      if (l_loc >= sp_recv_max){          // grow array for receiving valid tracking IDs for new particles 
        sp_recv_max = l_loc + BFF;
        memory->grow(sp_recv,sp_recv_max,"fix_inflow_periodic:sp_recv"); 
      }

      ll[0] = l_n;
      ll[1] = l_e;
      ll[2] = 0;
      MPI_Gather(&ll[0],2,MPI_INT,send_counts,2,MPI_INT,0,bc_send_comm[ii]); // gather information about sizes
      MPI_Reduce(&l_en,&ll[2],1,MPI_INT,MPI_SUM,0,bc_send_comm[ii]); 
      ll[0] = ll[1] = 0;
      if (bc_send_comm_rank[ii] == 0)
        for (i = 0; i < bc_send_comm_size[ii]; ++i){           // calculate the total number of new and old particles which need to be tracked
          ll[0] += send_counts[2*i];
          ll[1] += send_counts[2*i+1];
	}
      MPI_Bcast(&ll[0],3,MPI_INT,0,bc_send_comm[ii]);     // broadcast basic information within a sending group 
     
      if (bc_send_comm_rank[ii] == 0){                 
        MPI_Send(&ll[0],3,MPI_INT,recv_partner[ii],send_partner[ii],world);      // send basic information about the volume of data exchange
        i = MAX(ll[0],ll[1]) + 1;
        if (i >= send_max){                              // grow sending buffer if needed  
          send_max = i + BFF; 
          memory->grow(buf_send,send_max,"fix_inflow_periodic:buf_send");
	}         
      }  

      if (ll[0]){
        if (bc_send_comm_rank[ii] == 0){
          offset = 0;
          for (i = 0; i < bc_send_comm_size[ii]; ++i){       // create rcounts and displs for coming MPI_Gatherv to collect new particles
            displs[i] = offset;
            rcounts[i] = send_counts[2*i];
            offset += rcounts[i];
          } 
        }
        MPI_Gatherv(send_new,l_n,MPI_DOUBLE,buf_send,rcounts,displs,MPI_DOUBLE,0,bc_send_comm[ii]); // gather new particles

        if (bc_send_comm_rank[ii] == 0){                 
          offset = 0;
          for (i = 0; i < bc_send_comm_size[ii]; ++i){       // create rcounts and displs for coming MPI_Scatterv to send valid tracking IDs for new particles 
            displs[i] = offset;
            rcounts[i] = static_cast<int> ((send_counts[2*i]+1)/part_size);
            offset += rcounts[i];
          } 
          kk = offset;
          if (offset >= send_back_max){                  // grow array for sending back valid tracking IDs for new particles if needed
            send_back_max = offset + BFF; 
            memory->grow(buf_send_back,send_back_max,"fix_inflow_periodic:buf_send_back");
  	  }        
          k = 0;  
          for (i = 0; i < sp_max; ++i){                 // create new particle IDs for tracking
            if (sp_list[i] < 0){
              buf_send[k*part_size] = static_cast<double> (i);
              //buf_send[k*part_size] = ubuf(i).d;
              sp_list[i] = 1;
              buf_send_back[k] = i; 
              k++; 
	    }
            if (k == kk) break;            
          }
          if (k < kk){            // create new particle IDs for tracking
            j = sp_max;
            st = j+kk-k;
            sp_max = st + BFF; 
            memory->grow(sp_list,sp_max,"fix_inflow_periodic:sp_list");
            for (i = st; i < sp_max; ++i)
              sp_list[i] = -1;
            for (i = j; i < st; ++i){
              buf_send[k*part_size] = static_cast<double> (i);
              //buf_send[k*part_size] = ubuf(i).d;
              sp_list[i] = 1;
              buf_send_back[k] = i; 
              k++;   
	    }  
	  }
	}
        MPI_Scatterv(buf_send_back,rcounts,displs,MPI_INT,sp_recv,l_loc,MPI_INT,0,bc_send_comm[ii]);    // send valid tracking IDs for new particles 
        for (i = 0; i < l_loc; ++i)
          sp[list_new[i]] = sp_recv[i];

        if (bc_send_comm_rank[ii] == 0){ 
          buf_send[ll[0]] = static_cast<double> (sp_max);               
          MPI_Send(buf_send,ll[0]+1,MPI_DOUBLE,recv_partner[ii],send_partner[ii],world);   // send the data about new particles 
        }
      }

      if (ll[1]){
        if (bc_send_comm_rank[ii] == 0){
          offset = 0;
          for (i = 0; i < bc_send_comm_size[ii]; ++i){       // create rcounts and displs for coming MPI_Gatherv to collect already tracked particles
            displs[i] = offset;
            rcounts[i] = send_counts[2*i+1];
            offset += rcounts[i];
          } 
        }
        MPI_Gatherv(send_exist,l_e,MPI_DOUBLE,buf_send,rcounts,displs,MPI_DOUBLE,0,bc_send_comm[ii]); // gather already tracked particles

        if (bc_send_comm_rank[ii] == 0){
          k = 0;
          for (i = 0; i < ll[2]; ++i){              // refresh the local list of tracked particles
            l = static_cast<int>(buf_send[k]);
            //l = (int) ubuf(buf_send[k]).i;            
            if (buf_send[k+1] < 0.0){
              sp_list[l] = -1; 
              k += 2;
	    } else 
              k += part_size_track;   
	  }
          MPI_Send(buf_send,ll[1],MPI_DOUBLE,recv_partner[ii],recv_partner[ii],world);   // send the data about already tracked particles  
	}
      }
 
      if (bc_send_comm_rank[ii] == 0){     // used for consistency checks
        tot_tracked_part = 0;
        for (i = 0; i < sp_max; ++i)
          if (sp_list[i] > 0)
            tot_tracked_part++;
      }
    }

    if (bc_recv_comm_rank[ii] > -1){  // only cores which participate in receiving the data

      if (bc_recv_comm_rank[ii] == 0)
        MPI_Recv(&ll[0],3,MPI_INT,send_partner[ii],send_partner[ii],world,MPI_STATUS_IGNORE);    // receive basic information about the volume of data exchange

      MPI_Bcast(&ll[0],3,MPI_INT,0,bc_recv_comm[ii]);    // broadcast basic information within a receiving group 
      i = MAX(ll[0],ll[1]) + 1;
      if (i >= recv_max){
        recv_max = i + BFF; 
        memory->grow(buf_recv,recv_max,"fix_inflow_periodic:buf_recv");
      }
 
      if (ll[0]){
        if (bc_recv_comm_rank[ii] == 0)
          MPI_Recv(buf_recv,ll[0]+1,MPI_DOUBLE,send_partner[ii],send_partner[ii],world,MPI_STATUS_IGNORE);    // receive the data about new particles 

        MPI_Bcast(buf_recv,ll[0]+1,MPI_DOUBLE,0,bc_recv_comm[ii]);    // broadcast the data for new particles  
        l_n = static_cast<int> ((ll[0]+1)/part_size); 
        for (i = 0; i < l_n; ++i){                      // process the data for new particles 
          k = i*part_size;
          for (j = 0; j < 3; ++j)
            xh[j] = buf_recv[k + 3 + j];

          ind = 0;
          if (comm->user_part){                        // check if a new particle is mine and has to be inserted 
            l = comm->coords_to_bin(xh,nn,comm->nbp_loc,comm->box_min);
            if (l)
              if (comm->m_comm[nn[0]][nn[1]][nn[2]] & comm->local_bit) ind = 1; 
          } else{
            if (xh[0] >= sublo[0] && xh[0] < subhi[0] && xh[1] >= sublo[1] && xh[1] < subhi[1] && xh[2] >= sublo[2] && xh[2] < subhi[2]) ind = 1;
	  }

          if (ind){                             // insert a new particle
            l = static_cast<int>(buf_recv[k + 1]);
            //l = (int) ubuf(buf_recv[k + 1]).i;
            atom->avec->create_atom(l,xh);
            nlocal = atom->nlocal;
            kk = nlocal - 1;             
            sp[kk] = static_cast<int>(buf_recv[k]); 
            //sp[kk] = (int) ubuf(buf_recv[k]).i;
            mask[kk] = static_cast<int>(buf_recv[k + 2]);  
            //mask[kk] = (int) ubuf(buf_recv[k + 2]).i;
            for (j = 0; j < 3; ++j)
              v[kk][j] = buf_recv[k + 6 + j];
            if (spin_ind)
              for (j = 0; j < 3; ++j)
                omega[kk][j] = buf_recv[k + 9 + j];
	  }
	}
      
        l_loc = static_cast<int>(buf_recv[ll[0]]); 
        if (l_loc > sp_max){ 
          j = sp_max;
          sp_max = l_loc;
          memory->grow(sp_list,sp_max,"fix_inflow_periodic:sp_list"); 
          for (i = j; i < sp_max; ++i)
            sp_list[i] = -2;
	}
      }

      if (ll[1]){
        if (bc_recv_comm_rank[ii] == 0)
          MPI_Recv(buf_recv,ll[1],MPI_DOUBLE,send_partner[ii],recv_partner[ii],world,MPI_STATUS_IGNORE);    // receive the data about new particles

        MPI_Bcast(buf_recv,ll[1],MPI_DOUBLE,0,bc_recv_comm[ii]);    // broadcast the data for already tracked particles 

        k = 0;
        for (i = 0; i < ll[2]; ++i){              // unpack and assign the data of tracked particles
          l = static_cast<int>(buf_recv[k]);
          //l = (int) ubuf(buf_recv[k]).i;
          kk = sp_list[l];
          if (kk > -1){
            ind = static_cast<int>(buf_recv[k+1]); 
            //ind = (int) ubuf(buf_recv[k+1]).i;
            if (ind == 1){                       // assign the data of a tracked particle
              for (j = 0; j < 3; ++j)
                x[kk][j] = buf_recv[k + 2 + j];           
              for (j = 0; j < 3; ++j)
                v[kk][j] = buf_recv[k + 5 + j];
              if (spin_ind)
                for (j = 0; j < 3; ++j)
                  omega[kk][j] = buf_recv[k + 8 + j];
              k += part_size_track;               
	    } else if (ind == -1){               // remove the particle
              if (nlocal > 1){
                atom->avec->copy(nlocal-1,kk,1);
                j = sp[kk];
                if (j > -1)
                  if (sp_list[j] > -1)
                    sp_list[j] = kk; 
              } else if (nlocal <= 0)
                error->one(FLERR,"nlocal is non-positive, but we still try to remove an atom!!!");                 
              atom->nlocal--;
              nlocal = atom->nlocal;
              sp_list[l] = -2;
              k += 2;
	    } else {                           // do not track the particle anymore
              sp[kk] = -2;
              sp_list[l] = -2;
              k += 2;
	    }
	  } else{
            if (buf_recv[k+1] < 0.0) 
              k += 2;
	    else          
              k += part_size_track; 
	  }
	}
      }
    }
  }

  ntemp = atom->nlocal;
  MPI_Allreduce(&ntemp,&atom->natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (atom->natoms < 0 || atom->natoms >= MAXBIGINT)
    error->all(FLERR,"Too many total atoms");

  if (atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }

  //consistency_check(1);    
}

/* ---------------------------------------------------------------------- */

void FixInflowPeriodic::end_of_step()
{
  int i,j,k,l,ii,kk,offset;
  int l_e, ll;
  double **x = atom->x;
  double **v = atom->v;
  double **omega = atom->omega;
  int *sp = atom->spin;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // exchange of data between beginnings and ends of periodical sections
  for (ii = 0; ii < num_inflow; ++ii){
    if (bc_send_comm_rank[ii] > -1){  // only cores which participate in sending the data
      if (nlocal >= save_max){
        save_max = nlocal + NUM;
        memory->grow(xp_save,save_max,3,"fix_inflow_periodic:xp_save");
      }
      l_e = 0;
      for (i = 0; i < nlocal; ++i){
        for (j = 0; j < 3; ++j)
          xp_save[i][j] = x[i][j];
        if (mask[i] & groupbit)
          if (sp[i] > -1){              // only already tracked particles  
            if (l_e + BFF > exist_max){   // grow sending arrays if needed
              exist_max += NUM;
              memory->grow(send_exist,exist_max,"fix_inflow_periodic:send_exist"); 
	    } 
            send_exist[l_e++] = static_cast<double> (sp[i]);           
            //send_exist[l_e++] = ubuf(sp[i]).d;
            for (j = 0; j < 3; ++j)
              send_exist[l_e++] = v[i][j];
            if (spin_ind)
              for (j = 0; j < 3; ++j)
                send_exist[l_e++] = omega[i][j];      
	  }
      }
      MPI_Gather(&l_e,1,MPI_INT,send_counts,1,MPI_INT,0,bc_send_comm[ii]); // gather information about sizes

      ll = 0; 
      if (bc_send_comm_rank[ii] == 0){
        for (i = 0; i < bc_send_comm_size[ii]; ++i)           // calculate the total data volume which is tracked
          ll += send_counts[i];
        k = static_cast<int> ((ll+1)/part_size_vel);
        if (k != tot_tracked_part) 
          error->warning(FLERR,"Did not pass first consistency check!!! Number of tracked particles in the function end_of_step() is different from tot_tracked_part!");  
      }
      MPI_Bcast(&ll,1,MPI_INT,0,bc_send_comm[ii]);     // broadcast basic information within a sending group  

      if (bc_send_comm_rank[ii] == 0){                 
        MPI_Send(&ll,1,MPI_INT,recv_partner[ii],send_partner[ii],world);      // send basic information about the volume of data exchange
        if (ll >= send_max){                              // grow sending buffer if needed  
          send_max = ll + BFF; 
          memory->grow(buf_send,send_max,"fix_inflow_periodic:buf_send");
	}         
      }  

      if (ll){
        if (bc_send_comm_rank[ii] == 0){
          offset = 0;
          for (i = 0; i < bc_send_comm_size[ii]; ++i){       // create rcounts and displs for coming MPI_Gatherv to collect already tracked particles
            displs[i] = offset;
            rcounts[i] = send_counts[i];
            offset += rcounts[i];
          } 
        }
        MPI_Gatherv(send_exist,l_e,MPI_DOUBLE,buf_send,rcounts,displs,MPI_DOUBLE,0,bc_send_comm[ii]); // gather already tracked particles

        if (bc_send_comm_rank[ii] == 0)
          MPI_Send(buf_send,ll,MPI_DOUBLE,recv_partner[ii],recv_partner[ii],world);   // send the data about already tracked particles  	
      }
    }

    if (bc_recv_comm_rank[ii] > -1){  // only cores which participate in receiving the data
      if (nlocal >= save_max){
        save_max = nlocal + NUM;
        memory->grow(xp_save,save_max,3,"fix_inflow_periodic:xp_save");
      }
      for (i = 0; i < sp_max; ++i)      // refresh local sp_lists
        sp_list[i] = -2;
      k = l = 0;
      for (i = 0; i < nlocal; ++i){      // refresh local sp_lists
        for (j = 0; j < 3; ++j)
          xp_save[i][j] = x[i][j];
        if (mask[i] & groupbit) 
          if (sp[i] > -1){
            j = sp[i];
            if (j >= sp_max)
              error->warning(FLERR,"The tracking sp_list array is not long enough!"); 
            sp_list[j] = i;
            k++;
	  }
      } 
      MPI_Reduce(&k,&l,1,MPI_INT,MPI_SUM,0,bc_recv_comm[ii]);     // used for a consistency check 

      if (bc_recv_comm_rank[ii] == 0){
        MPI_Recv(&ll,1,MPI_INT,send_partner[ii],send_partner[ii],world,MPI_STATUS_IGNORE);    // receive basic information about the volume of data exchange
        k = static_cast<int> ((ll+1)/part_size_vel);
        if (k != l) 
          error->warning(FLERR,"Did not pass second consistency check!!! Number of tracked particles in the function end_of_step() is different at sending and receiving groups!"); 
      }

      MPI_Bcast(&ll,1,MPI_INT,0,bc_recv_comm[ii]);    // broadcast basic information within a receiving group 
      if (ll >= recv_max){
        recv_max = ll + BFF; 
        memory->grow(buf_recv,recv_max,"fix_inflow_periodic:buf_recv");
      }
 
      if (ll){
        if (bc_recv_comm_rank[ii] == 0)
          MPI_Recv(buf_recv,ll,MPI_DOUBLE,send_partner[ii],recv_partner[ii],world,MPI_STATUS_IGNORE);    // receive the data about new particles

        MPI_Bcast(buf_recv,ll,MPI_DOUBLE,0,bc_recv_comm[ii]);    // broadcast the data for already tracked particles 

        l_e = static_cast<int> ((ll+1)/part_size_vel);
        k = 0;
        for (i = 0; i < l_e; ++i){              // unpack and assign the data of tracked particles
          l = static_cast<int> (buf_recv[k]);
          //l = (int) ubuf(buf_recv[k]).i;
          kk = sp_list[l];
          if (kk > -1){            // assign the data of a tracked particle         
            for (j = 0; j < 3; ++j)
              v[kk][j] = buf_recv[k + 1 + j];
            if (spin_ind)
              for (j = 0; j < 3; ++j)
                omega[kk][j] = buf_recv[k + 4 + j];
          }
          k += part_size_vel; 
	}
      }
    }
  }

  //consistency_check(2);
}

/*------------------------------------------------------------------------*/

void FixInflowPeriodic::reset_target(double rho_new)
{
  int i,j,ii;
  double xh[3],rr;

  double **x = atom->x;
  int *mask = atom->mask;
  double *rho = atom->rho;
  int nall = atom->nlocal + atom->nghost;

  for (ii = 0; ii < num_inflow; ++ii)
    if (bc_recv_comm_rank[ii] > -1)
      for (i = 0; i < nall; ++i)
        if (mask[i] & groupbit){
          for (j = 0; j < 3; ++j)
            xh[j] = x[i][j] - x0[ii][j];
          rot_forward(xh[0],xh[1],xh[2],ii);
          rr = xh[0]*xh[0] + xh[1]*xh[1];
          if (xh[2] >= 0.0 && xh[2] < cut && rr < aa[ii][3]*aa[ii][3])
            rho[i] = rho_new;
        }
}

/*------------------------------------------------------------------------*/

void FixInflowPeriodic::set_comm_groups()
{
  int i,j,k,l,ii,nn[3],jj[3],color,ind,offset;
  double xx[3],ddx[3],rr;
  double rad = sqrt(3.0)*binsize;
  double lm = cut+skin+rad;
  int *tmp;  

  for (j = 0; j < 3; ++j){                  // setup bins
    ddx[j] = subhi[j] - sublo[j];
    nn[j] = static_cast<int> (ddx[j]/binsize + 3.0);
  }

  k = 0;
  offset = 0;
  for (ii = 0; ii < num_inflow; ++ii){               
    ind = 0;
    for (jj[0] = 0; jj[0] < nn[0]; ++jj[0]){             // check if certain bins overlap with a core
      for (jj[1] = 0; jj[1] < nn[1]; ++jj[1]){
        for (jj[2] = 0; jj[2] < nn[2]; ++jj[2]){
          for (l = 0; l < 3; ++l)
            xx[l] = sublo[l] + (jj[l]-0.5)*binsize - x0[ii][l]; 
          rot_forward(xx[0],xx[1],xx[2],ii);
          rr = sqrt(xx[0]*xx[0] + xx[1]*xx[1]);
          if (rr < aa[ii][3]+rad && xx[2] > aa[ii][4]-rad && xx[2] < aa[ii][4]+lm){
            ind = 1;
            break;
          }
        }
        if (ind) break;
      }     
      if (ind) break;
    }
    
    if (ind)
      color = 1;
    else
      color = MPI_UNDEFINED;

    MPI_Comm_split(world,color,comm->me,&bc_send_comm[ii]);                     // create new communicators
    if (color == 1){
      MPI_Comm_rank(bc_send_comm[ii],&bc_send_comm_rank[ii]);
      MPI_Comm_size(bc_send_comm[ii],&bc_send_comm_size[ii]);
    } else{
      bc_send_comm[ii] = MPI_COMM_NULL;
      bc_send_comm_rank[ii] = -1;
      bc_send_comm_size[ii] = 0;
    }

    if (bc_send_comm_rank[ii] > -1){
      //printf("sending group: inflow = %d, glob_me=%d, me=%d, size=%d \n",ii,comm->me,bc_send_comm_rank[ii],bc_send_comm_size[ii]);
      k++;
      if (bc_send_comm_rank[ii] == 0){
        offset++;
        if (bc_send_comm_size[ii] <= 0)
          error->one(FLERR,"The size of a sending group cannot be a non-negative number!");
        memory->create(send_counts,2*bc_send_comm_size[ii],"fix_inflow_periodic:send_counts");
        memory->create(rcounts,bc_send_comm_size[ii],"fix_inflow_periodic:rcounts");
        memory->create(displs,bc_send_comm_size[ii],"fix_inflow_periodic:displs");  
        send_partner[ii] = comm->me;
      }
    }

    ind = 0;
    for (jj[0] = 0; jj[0] < nn[0]; ++jj[0]){                             // check if certain bins overlap with a core
      for (jj[1] = 0; jj[1] < nn[1]; ++jj[1]){
        for (jj[2] = 0; jj[2] < nn[2]; ++jj[2]){
          for (l = 0; l < 3; ++l)
            xx[l] = sublo[l] + (jj[l]-0.5)*binsize - x0[ii][l];
          rot_forward(xx[0],xx[1],xx[2],ii);
          rr = sqrt(xx[0]*xx[0] + xx[1]*xx[1]);
          if (rr < aa[ii][3]+rad && xx[2] > -rad && xx[2] < lm){
            ind = 1;
            break;
          }
        }
        if (ind) break;
      }
      if (ind) break;
    }

    if (ind)
      color = 1;
    else
      color = MPI_UNDEFINED;

    MPI_Comm_split(world,color,comm->me,&bc_recv_comm[ii]);              // create new communicators
    if (color == 1){
      MPI_Comm_rank(bc_recv_comm[ii],&bc_recv_comm_rank[ii]);
      MPI_Comm_size(bc_recv_comm[ii],&bc_recv_comm_size[ii]);
    } else{
      bc_recv_comm[ii] = MPI_COMM_NULL;
      bc_recv_comm_rank[ii] = -1;
      bc_recv_comm_size[ii] = 0;
    }

    if (bc_recv_comm_rank[ii] > -1){
      //printf("receiving group: inflow = %d, glob_me=%d, me=%d, size=%d \n",ii,comm->me,bc_recv_comm_rank[ii],bc_recv_comm_size[ii]);
      k++;
      if (bc_recv_comm_rank[ii] == 0){
        offset++;
        if (bc_recv_comm_size[ii] <= 0)
          error->one(FLERR,"The size of a receiving group cannot be a non-negative number!");
        memory->create(send_counts,bc_recv_comm_size[ii],"fix_inflow_periodic:send_counts");
        memory->create(rcounts,bc_recv_comm_size[ii],"fix_inflow_periodic:rcounts");
        memory->create(displs,bc_recv_comm_size[ii],"fix_inflow_periodic:displs");
        recv_partner[ii] = comm->me;
      }
    }
  }

  if (k > 1)
    error->warning(FLERR,"There are cores, which may participate in multiple inflows (or their beginning/ending)! This may not function properly in fix inflow/periodic!");
  if (offset > 1)
    error->one(FLERR,"There are roots of inflow groups, which may participate in multiple inflows (or their beginning/ending)! This will definitely not function properly in fix inflow/periodic!");

  memory->create(tmp,num_inflow,"fix_inflow_periodic:tmp");
  for (ii = 0; ii < num_inflow; ++ii){                   // assign sending partners
    tmp[ii] = send_partner[ii];
    send_partner[ii] = -1;
  }
  MPI_Allreduce(tmp,send_partner,num_inflow,MPI_INT,MPI_MAX,world);
  for (ii = 0; ii < num_inflow; ++ii){                  // assign receiving partners
    tmp[ii] = recv_partner[ii];
    recv_partner[ii] = -1;
  }
  MPI_Allreduce(tmp,recv_partner,num_inflow,MPI_INT,MPI_MAX,world);
  memory->destroy(tmp);
 
  consistency_check(0);
}

/*------------------------------------------------------------------------*/

void FixInflowPeriodic::setup_rot(int id)
{
  int i,j;
  double norm[3],nr,rx;

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      rot[id][i][j] = 0.0;

  norm[0] = aa[id][0];
  norm[1] = aa[id][1];
  norm[2] = aa[id][2];
  nr = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
  rx = sqrt(norm[0]*norm[0] + norm[1]*norm[1]);
  if (rx > EPS){
    rot[id][0][0] = norm[0]*norm[2]/nr/rx;
    rot[id][0][1] = norm[1]*norm[2]/nr/rx;
    rot[id][0][2] = -rx/nr;
    rot[id][1][0] = -norm[1]/rx;
    rot[id][1][1] = norm[0]/rx;
    rot[id][1][2] = 0.0;
    rot[id][2][0] = norm[0]/nr;
    rot[id][2][1] = norm[1]/nr;
    rot[id][2][2] = norm[2]/nr;
  } else {
    rot[id][0][0] = 1.0;
    rot[id][1][1] = 1.0;
    rot[id][2][2] = 1.0;
    if (norm[2] < 0.0)
      rot[id][2][2] = -1.0;
  }
  aa[id][4] = nr;
}

/* ---------------------------------------------------------------------- */

void FixInflowPeriodic::rot_forward(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][0][1]*y1 + rot[id][0][2]*z1;
  y = rot[id][1][0]*x1 + rot[id][1][1]*y1 + rot[id][1][2]*z1;
  z = rot[id][2][0]*x1 + rot[id][2][1]*y1 + rot[id][2][2]*z1;
}

/* ---------------------------------------------------------------------- */

void FixInflowPeriodic::rot_back(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][1][0]*y1 + rot[id][2][0]*z1;
  y = rot[id][0][1]*x1 + rot[id][1][1]*y1 + rot[id][2][1]*z1;
  z = rot[id][0][2]*x1 + rot[id][1][2]*y1 + rot[id][2][2]*z1;
}

/*------------------------------------------------------------------------*/

void FixInflowPeriodic::consistency_check(int id)
{
  int i,j,ii,kk,color,ind,offset,spm;
  int *tmp, *tmp1, *sp_list_tmp;
  double **x = atom->x;
  int *sp = atom->spin;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int step = update->ntimestep;
  char str1[50],str[200];

  if (id == 0)
    sprintf(str1,"Initial consistency check");
  else if (id == 1)
    sprintf(str1,"Post_integrate consistency check");
  else
    sprintf(str1,"End_of_step consistency check");

  for (ii = 0; ii < num_inflow; ++ii){                // consistency checks

    if (bc_send_comm_rank[ii] > -1){
      if (id == 0){
        save_max = nlocal + NUM;
        memory->create(xp_save,save_max,3,"fix_inflow_periodic:xp_save");       
      }
      ind = 0;
      for (i = 0; i < nlocal; ++i){ // check how many tracked IDs already exist 
        if (id == 0)
          for (j = 0; j < 3; ++j)
            xp_save[i][j] = x[i][j];            
        if (mask[i] & groupbit)
          if (sp[i] > -1)
            ind++;
      }

      MPI_Gather(&ind,1,MPI_INT,send_counts,1,MPI_INT,0,bc_send_comm[ii]); // gather the numbers of tracked IDs 
      if (bc_send_comm_rank[ii] == 0){
        color = 0;
        for (i = 0; i < bc_send_comm_size[ii]; ++i)
          color += send_counts[i];
      }
      MPI_Bcast(&color,1,MPI_INT,0,bc_send_comm[ii]);   // broadcast the total number of tracked IDs

      if (bc_send_comm_rank[ii] == 0){
        MPI_Recv(&kk,1,MPI_INT,recv_partner[ii],recv_partner[ii],world,MPI_STATUS_IGNORE);
        if (kk != color){
          sprintf(str,"%s at step=%d. The total number of already existing tracked IDs at the beginning and end of a cylinder do not match!!",str1,step);
          error->warning(FLERR,str); 
        }
      }

      if (color){
        memory->create(tmp1,ind+1,"fix_inflow_periodic:tmp1");
        ind = 0;
        for (i = 0; i < nlocal; ++i)
          if (mask[i] & groupbit)
            if (sp[i] > -1)
              tmp1[ind++] = sp[i];

        if (bc_send_comm_rank[ii] == 0){
          memory->create(tmp,color,"fix_inflow_periodic:tmp");
          offset = 0;
          for (i = 0; i < bc_send_comm_size[ii]; ++i){
            displs[i] = offset;
            rcounts[i] = send_counts[i];
            offset += rcounts[i];
          }
        }
        MPI_Gatherv(tmp1,ind,MPI_INT,tmp,rcounts,displs,MPI_INT,0,bc_send_comm[ii]); // gather already existing tracked IDs 

        if (bc_send_comm_rank[ii] == 0){
          spm = -1; 
          for (i = 0; i < color; ++i)
            spm = MAX(spm,tmp[i]);
          spm += 5;
          memory->create(sp_list_tmp,spm,"fix_inflow_periodic:sp_list_tmp");       // initialize sp_list_tmp
          for (i = 0; i < spm; ++i)
            sp_list_tmp[i] = -1;
          for (i = 0; i < color; ++i){            // set initial list
            j = tmp[i];
            sp_list_tmp[j] += 2;
          }
          if (id == 0){
            if (spm > sp_max)
              sp_max = spm;
            memory->create(sp_list,sp_max,"fix_inflow_periodic:sp_list");       // initialize sp_list
            for (i = 0; i < sp_max; ++i)
              sp_list[i] = -1;  
            for (i = 0; i < spm; ++i)
              sp_list[i] = sp_list_tmp[i]; 
	  }
          for (i = 0; i < spm; ++i)          // check if IDs are unique    
            if (sp_list_tmp[i] > 1){
              sprintf(str,"%s at step=%d. Some already existing tracked IDs are not unique at a sending group of cores!!",str1,step);
              error->warning(FLERR,str); 
            } 

          MPI_Recv(tmp,kk,MPI_INT,recv_partner[ii],recv_partner[ii],world,MPI_STATUS_IGNORE);

          for (i = 0; i < kk; ++i){
            j = tmp[i];
            if (sp_list_tmp[j] < 0){
              sprintf(str,"%s at step=%d. Some already existing tracked IDs at the beginning of a cylinder are not consistent with those at the end!!",str1,step);
              error->warning(FLERR,str);  
            } 
          }

          memory->destroy(tmp);
          memory->destroy(sp_list_tmp);
        }
        memory->destroy(tmp1);
      } else{
        if (id == 0)
          if (bc_send_comm_rank[ii] == 0){
            memory->create(sp_list,sp_max,"fix_inflow_periodic:sp_list"); // create original sp_list even if there are no already existing tracked IDs 
            for (i = 0; i < sp_max; ++i)
              sp_list[i] = -1;
          } 
      }
    }

    if (bc_recv_comm_rank[ii] > -1){
      if (id == 0){
        save_max = nlocal + NUM;
        memory->create(xp_save,save_max,3,"fix_inflow_periodic:xp_save"); 
      }
      ind = 0;
      for (i = 0; i < nlocal; ++i){
        if (id == 0)
          for (j = 0; j < 3; ++j)
            xp_save[i][j] = x[i][j];
        if (mask[i] & groupbit)
          if (sp[i] > -1){
            if (id == 0)
              sp_max = MAX(sp_max,sp[i]);
            ind++;
	  }
      }
      
      if (id == 0){
        sp_max += 5;
        memory->create(sp_list,sp_max,"fix_inflow_periodic:sp_list");   // create local sp_list for already existing tracked IDs in a receiving group 
        for (i = 0; i < sp_max; ++i)
          sp_list[i] = -2;
      }
      
      MPI_Gather(&ind,1,MPI_INT,send_counts,1,MPI_INT,0,bc_recv_comm[ii]); // gather the numbers of tracked IDs 
      if (bc_recv_comm_rank[ii] == 0){
        color = 0;
        for (i = 0; i < bc_recv_comm_size[ii]; ++i)
          color += send_counts[i];
      }
      MPI_Bcast(&color,1,MPI_INT,0,bc_recv_comm[ii]);   // broadcast the total number of tracked IDs

      if (bc_recv_comm_rank[ii] == 0)
        MPI_Send(&color,1,MPI_INT,send_partner[ii],recv_partner[ii],world);

      if (color){
        memory->create(tmp1,ind+1,"fix_inflow_periodic:tmp1");
        ind = 0;
        for (i = 0; i < nlocal; ++i)
          if (mask[i] & groupbit)
            if (sp[i] > -1){
              j = sp[i];
              tmp1[ind++] = j;
              if (id == 0)
                sp_list[j] = i;  
            }

        if (bc_recv_comm_rank[ii] == 0){
          memory->create(tmp,color,"fix_inflow_periodic:tmp");
          offset = 0;
          for (i = 0; i < bc_recv_comm_size[ii]; ++i){
            displs[i] = offset;
            rcounts[i] = send_counts[i];
            offset += rcounts[i];
          }
        }
        MPI_Gatherv(tmp1,ind,MPI_INT,tmp,rcounts,displs,MPI_INT,0,bc_recv_comm[ii]); // gather already existing tracked IDs  

        if (bc_recv_comm_rank[ii] == 0){
          kk = -1;
          for (i = 0; i < color; ++i)
            kk = MAX(kk,tmp[i]);
          kk += 2;
          memory->grow(tmp1,kk,"fix_inflow_periodic:tmp1");
          for (i = 0; i < kk; ++i)
            tmp1[i] = -1;
          for (i = 0; i < color; ++i){
            j = tmp[i];
            tmp1[j] += 2;
          }
          for (i = 0; i < kk; ++i)          // check if IDs are unique    
            if (tmp1[i] > 1){
              sprintf(str,"%s at step=%d. Some already existing tracked IDs are not unique at a receiving group of cores!!",str1,step);
              error->warning(FLERR,str); 
            }
 
          MPI_Send(tmp,color,MPI_INT,send_partner[ii],recv_partner[ii],world);

          memory->destroy(tmp);
        }
        memory->destroy(tmp1);
      }
    }
  }
}

/*------------------------------------------------------------------------*/

