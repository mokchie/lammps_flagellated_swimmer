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

#include <cstring>
//#include <unistd.h>
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

using namespace LAMMPS_NS;
using namespace FixConst;

#define  NUM 1000
#define  EPS 1e-6
#define  BFF 50
#define  MOLINC 50

/* ---------------------------------------------------------------------- */

FixInflowPeriodic::FixInflowPeriodic(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  int i,j,l,igroup,nms;
  double *rdata = NULL;
  char grp[50];
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  FILE *f_read;

  rot = NULL;
  x0 = aa = NULL;
  send_exist = send_new = NULL;
  list_new = sp_list = send_counts = NULL;
  rcounts = displs = buf_send_back = sp_recv = NULL;
  buf_send = buf_recv = NULL;
  xp_save = NULL;
  num_mols = num_mols_prev = NULL;
  tag_st = tag_st_prev = NULL;
  mol_st = mol_st_prev = NULL;
  mol_list = NULL;
  mol_corresp_list = NULL;
 
  atom->mol_corresp_ind = 1; 
  nms = NUM;
  groupbit_reflect = 0;
  exist_max = new_max = 0; 
  new_list_max = 0;
  send_max = recv_max = 0;
  sp_recv_max = send_back_max = 0;
  sp_max = BFF;
  part_size = 10;
  spin_ind = 0;
  save_max = 0;
  n_mol_max = n_mol_types = 0;
  n_mol_corresp = n_mol_corresp_max = 0;
  
  if (!(strcmp(atom->atom_style,"molecular") == 0) && !(strcmp(atom->atom_style,"sdpd") == 0))
    error->all(FLERR,"The fix_inflow_periodic can only be used with atom_style molecular and sdpd!");

  if (strcmp(atom->atom_style,"sdpd") == 0){
    spin_ind = 1;
    part_size += 3;
  }
  part_size_track = part_size - 2;
  part_size_vel = part_size - 6;

  if (narg != 4) error->all(FLERR,"Illegal fix_inflow_periodic command");
  sprintf(fname,arg[3]);

  if (comm->me == 0){  // read the data by the root
    l = 0;
    f_read = fopen(fname,"r");
    if (f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open input boundary file from fix_inflow_periodic.");
    memory->create(rdata,nms,"fix_inflow_periodic:rdata");
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&num_inflow);
    rdata[l++] = static_cast<double>(num_inflow); 
    //rdata[l++] = ubuf(num_inflow).d;
    if (num_inflow <= 0)
      error->one(FLERR,"The number of inflows in fix_inflow_periodic cannot be a non-positive number!!"); 

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %d %s",&binsize,&skin,&mol_adapt_iter,&grp[0]);
    rdata[l++] = binsize;
    rdata[l++] = skin;
    rdata[l++] = static_cast<double>(mol_adapt_iter);
    //rdata[l++] = ubuf(mol_adapt_iter).d;
    igroup = group->find(grp);
    if (igroup == -1) error->one(FLERR,"Group ID for group_reflect does not exist");
    groupbit_reflect = group->bitmask[igroup];
    rdata[l++] = static_cast<double>(groupbit_reflect);
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
  if (comm->me)
    memory->create(rdata,l,"fix_inflow_periodic:rdata");
  MPI_Bcast(rdata,l,MPI_DOUBLE,0,world);  // send the read data to other cores

  l = 0;
  num_inflow = static_cast<int>(rdata[l++]);  
  //num_inflow = (int) ubuf(rdata[l++]).i;

  binsize = rdata[l++];
  skin = rdata[l++];
  mol_adapt_iter = static_cast<int>(rdata[l++]);
  //mol_adapt_iter = (int) ubuf(rdata[l++]).i; 
  groupbit_reflect = static_cast<int>(rdata[l++]);
  //groupbit_reflect = (int) ubuf(rdata[l++]).i;

  memory->create(x0,num_inflow,3,"fix_inflow_periodic:x0");       // allocate memory for all arrays
  memory->create(aa,num_inflow,5,"fix_inflow_periodic:aa");
  memory->create(rot,num_inflow,3,3,"fix_inflow_periodic:rot");   
  
  for (i = 0; i < num_inflow; ++i){
    for (j = 0; j < 3; ++j){
      x0[i][j] = rdata[l+j];
      aa[i][j] = rdata[l+j+3] - rdata[l+j];
    }
    aa[i][3] = rdata[l+6]; 
    setup_rot(i);
    l += 7;
  }
  send_partner = -1;
  recv_partner = -1;  

  memory->destroy(rdata);
}

/* ---------------------------------------------------------------------- */

FixInflowPeriodic::~FixInflowPeriodic()
{
  memory->destroy(rot);
  memory->destroy(x0);
  memory->destroy(aa); 
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
  memory->destroy(xp_save);
  memory->destroy(num_mols);
  memory->destroy(num_mols_prev);
  memory->destroy(tag_st);
  memory->destroy(tag_st_prev);
  memory->destroy(mol_st);
  memory->destroy(mol_st_prev);
  memory->destroy(mol_list);
  memory->destroy(mol_corresp_list);
}

/* ---------------------------------------------------------------------- */

int FixInflowPeriodic::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixInflowPeriodic::setup_pre_exchange()
{
  int i;

  if (neighbor->every > 1 || neighbor->delay > 1) 
    error->all(FLERR,"Reneighboring should be done every time step when using fix_inflow_periodic!");

  cut = neighbor->cutneighmax;
  for (i = 0; i < 3; ++i){
    subhi[i] = domain->subhi[i];
    sublo[i] = domain->sublo[i];
  }

  if (binsize <= 0.0)
    binsize = 0.5*cut;

  set_comm_groups();   // setup communication groups

  if (atom->n_mol_max)
    initial_mol_order_setup();  

  if (atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }
}

/* ---------------------------------------------------------------------- */

void FixInflowPeriodic::pre_exchange()
{
  int i,j,k,l,m,kk,offset,ind,st;
  int l_e, l_n, l_loc, l_en, l_nn, ll[4], nn[3];
  tagint ml;
  double dl[3],xp[3],xh[3],vh[3];
  double tt,rr;
  double **x = atom->x;
  double **v = atom->v;
  double **omega = atom->omega;
  int *sp = atom->spin;
  int *mask = atom->mask;
  int *type = atom->type;
  tagint *molecule = atom->molecule;
  tagint *tag = atom->tag;  
  int nlocal = atom->nlocal;
  double lm = cut+skin;
  bigint ntemp;
  bigint step_t = update->ntimestep;

  // reflections of particles at both surfaces if needed

  if (bc_send_comm_rank > -1)   // reflections at the end of a cylindrical cap
    for (i = 0; i < nlocal; ++i)
      if (mask[i] & groupbit_reflect)
	if (sp[i] < -1){
	  for (j = 0; j < 3; ++j){
	    xh[j] = x[i][j] - x0[ninf][j];
	    xp[j] = xp_save[i][j] - x0[ninf][j];
	    vh[j] = v[i][j];
	  }
	  rot_forward(xh[0],xh[1],xh[2],ninf);
	  rot_forward(xp[0],xp[1],xp[2],ninf);
	  rot_forward(vh[0],vh[1],vh[2],ninf);
	  rr = xh[0]*xh[0] + xh[1]*xh[1]; 
	  if (xh[2]-aa[ninf][4] < 0.0 && xp[2]-aa[ninf][4] > 0.0 && rr < aa[ninf][3]*aa[ninf][3]){
	    if (vh[2] < 0.0)
	      vh[2] = -vh[2];
	    tt = vh[0]*xh[0] + vh[1]*xh[1];
	    if (tt > 0.0){
	      vh[0] = -vh[0];
	      vh[1] = -vh[1];
	    }  
	    rot_back(vh[0],vh[1],vh[2],ninf);
	    for (j = 0; j < 3; ++j){
	      x[i][j] = xp_save[i][j];
	      v[i][j] = vh[j];
	    } 
	  }
	}

  if (bc_recv_comm_rank > -1) // reflections at the beginning of a cylindrical cap
    for (i = 0; i < nlocal; ++i)
      if (mask[i] & groupbit_reflect)
	if (sp[i] < -1){
	  for (j = 0; j < 3; ++j){
	    xh[j] = x[i][j] - x0[ninf][j];
	    xp[j] = xp_save[i][j] - x0[ninf][j];
	    vh[j] = v[i][j];
	  }
	  rot_forward(xh[0],xh[1],xh[2],ninf);
	  rot_forward(xp[0],xp[1],xp[2],ninf);
	  rot_forward(vh[0],vh[1],vh[2],ninf);
	  rr = xh[0]*xh[0] + xh[1]*xh[1];
	  if (xh[2] < 0.0 && xp[2] > 0.0 && rr < aa[ninf][3]*aa[ninf][3]){
	    if (vh[2] < 0.0)
	      vh[2] = -vh[2];
	    tt = vh[0]*xh[0] + vh[1]*xh[1];
	    if (tt > 0.0){
	      vh[0] = -vh[0];
	      vh[1] = -vh[1];
	    }
	    rot_back(vh[0],vh[1],vh[2],ninf);
	    for (j = 0; j < 3; ++j){
	      x[i][j] = xp_save[i][j];
	      v[i][j] = vh[j];
	    }
	  }
	}

  // exchange of data between beginnings and ends of periodical sections

  if (bc_send_comm_rank > -1){  // only cores which participate in sending the data
    l_e = l_n = l_loc = l_en = l_nn = 0;
    for (i = 0; i < nlocal; ++i)
      if (mask[i] & groupbit) { 
	for (j = 0; j < 3; ++j){
	  xh[j] = x[i][j] - x0[ninf][j];
	  xp[j] = xp_save[i][j] - x0[ninf][j];
	}
	rot_forward(xh[0],xh[1],xh[2],ninf);
	rot_forward(xp[0],xp[1],xp[2],ninf);

	xh[2] -= aa[ninf][4];
	xp[2] -= aa[ninf][4]; 
	rr = xh[0]*xh[0] + xh[1]*xh[1];
	if (l_e + BFF >= exist_max){   // grow sending arrays if needed
	  exist_max += NUM;
	  memory->grow(send_exist,exist_max,"fix_inflow_periodic:send_exist"); 
	}
	if (l_n + BFF >= new_max){     // grow sending arrays if needed
	  new_max += NUM;
	  memory->grow(send_new,new_max,"fix_inflow_periodic:send_new"); 
	}          
	if (rr < aa[ninf][3]*aa[ninf][3] && xh[2] >= 0.0 && xh[2] < lm){      // check if a particle is in the sending cylinder 
	  if (sp[i] > -1){          // already tracked - track further 
	    send_exist[l_e++] = static_cast<double> (sp[i]);  
	    //send_exist[l_e++] = ubuf(sp[i]).d;
	    send_exist[l_e++] = 1.0;
	    for (j = 0; j < 3; ++j)
	      send_exist[l_e++] = x[i][j] - aa[ninf][j];
	    for (j = 0; j < 3; ++j)
	      send_exist[l_e++] = v[i][j];
	    if (spin_ind)
	      for (j = 0; j < 3; ++j)
		send_exist[l_e++] = omega[i][j];  
	    l_en++; 
	  } else {
	    if (xp[2] < 0.0){       // new particle - need to track 
	      send_new[l_n++] = 0.0;        // save the required data to the sending buffer        
	      send_new[l_n++] = static_cast<double> (molecule[i]);
	      //send_new[l_n++] = ubuf(molecule[i]).d;
	      send_new[l_n++] = static_cast<double> (type[i]);
	      //send_new[l_n++] = ubuf(type[i]).d;
	      send_new[l_n++] = static_cast<double> (mask[i]);
	      //send_new[l_n++] = ubuf(mask[i]).d;
	      for (j = 0; j < 3; ++j)
		send_new[l_n++] = x[i][j] - aa[ninf][j];
	      for (j = 0; j < 3; ++j)
		send_new[l_n++] = v[i][j]; 
	      if (spin_ind)
		for (j = 0; j < 3; ++j)
		  send_new[l_n++] = omega[i][j];

	      if (molecule[i]){  // save more data into the buffer if this particle is a part of a molecule
		if (atom->individual)
		  j = 8 + 3*atom->num_bond[i] + 5*atom->num_angle[i] + 6*atom->num_dihedral[i] + 5*atom->num_improper[i] + atom->nspecial[i][2];
		else
		  j = 8 + 2*atom->num_bond[i] + 4*atom->num_angle[i] + 5*atom->num_dihedral[i] + 5*atom->num_improper[i] + atom->nspecial[i][2];

		if (l_n + j + BFF >= new_max){     // grow sending arrays if needed
	          new_max += NUM + j;
	          memory->grow(send_new,new_max,"fix_inflow_periodic:send_new"); 
	        }
	
                send_new[l_n++] = static_cast<double> (tag[i]); 	
		//send_new[l_n++] = ubuf(tag[i]).d;
                send_new[l_n++] = static_cast<double> (atom->num_bond[i]); 
                //send_new[l_n++] = ubuf(atom->num_bond[i]).d; // save bond data
                for (m = 0; m < atom->num_bond[i]; m++){
                  send_new[l_n++] = static_cast<double> (atom->bond_type[i][m]);
		  //send_new[l_n++] = ubuf(atom->bond_type[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->bond_atom[i][m]); 
		  //send_new[l_n++] = ubuf(atom->bond_atom[i][m]).d;
		}
		
                send_new[l_n++] = static_cast<double> (atom->num_angle[i]);
		//send_new[l_n++] = ubuf(atom->num_angle[i]).d; // save angle data
                for (m = 0; m < atom->num_angle[i]; m++){
                  send_new[l_n++] = static_cast<double> (atom->angle_type[i][m]);
		  //send_new[l_n++] = ubuf(atom->angle_type[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->angle_atom1[i][m]);
		  //send_new[l_n++] = ubuf(atom->angle_atom1[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->angle_atom2[i][m]);
                  //send_new[l_n++] = ubuf(atom->angle_atom2[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->angle_atom3[i][m]); 
		  //send_new[l_n++] = ubuf(atom->angle_atom3[i][m]).d;
		}
		
                send_new[l_n++] = static_cast<double> (atom->num_dihedral[i]);
		//send_new[l_n++] = ubuf(atom->num_dihedral[i]).d; // save dihedral data
                for (m = 0; m < atom->num_dihedral[i]; m++){
                  send_new[l_n++] = static_cast<double> (atom->dihedral_type[i][m]);
		  //send_new[l_n++] = ubuf(atom->dihedral_type[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->dihedral_atom1[i][m]);
		  //send_new[l_n++] = ubuf(atom->dihedral_atom1[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->dihedral_atom2[i][m]);
                  //send_new[l_n++] = ubuf(atom->dihedral_atom2[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->dihedral_atom3[i][m]); 
		  //send_new[l_n++] = ubuf(atom->dihedral_atom3[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->dihedral_atom4[i][m]);
		  //send_new[l_n++] = ubuf(atom->dihedral_atom4[i][m]).d;
		}
		
                send_new[l_n++] = static_cast<double> (atom->num_improper[i]);
		//send_new[l_n++] = ubuf(atom->num_improper[i]).d; // save improper data
                for (m = 0; m < atom->num_improper[i]; m++){
                  send_new[l_n++] = static_cast<double> (atom->improper_type[i][m]); 
		  //send_new[l_n++] = ubuf(atom->improper_type[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->improper_atom1[i][m]);
		  //send_new[l_n++] = ubuf(atom->improper_atom1[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->improper_atom2[i][m]); 
                  //send_new[l_n++] = ubuf(atom->improper_atom2[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->improper_atom3[i][m]);
		  //send_new[l_n++] = ubuf(atom->improper_atom3[i][m]).d;
                  send_new[l_n++] = static_cast<double> (atom->improper_atom4[i][m]);
		  //send_new[l_n++] = ubuf(atom->improper_atom4[i][m]).d;
		}
		
                if (atom->individual){ // save data for individual properties
                  for (m = 0; m < atom->num_bond[i]; m++)
                    send_new[l_n++] = atom->bond_length[i][m];
                  for (m = 0; m < atom->num_angle[i]; m++)
                    send_new[l_n++] = atom->angle_area[i][m];
                  for (m = 0; m < atom->num_dihedral[i]; m++)
                    send_new[l_n++] = atom->dihedral_angle[i][m];
		}
		
                send_new[l_n++] = static_cast<double> (atom->nspecial[i][0]);
                //send_new[l_n++] = ubuf(atom->nspecial[i][0]).d;
                send_new[l_n++] = static_cast<double> (atom->nspecial[i][1]);
                //send_new[l_n++] = ubuf(atom->nspecial[i][1]).d;
                send_new[l_n++] = static_cast<double> (atom->nspecial[i][2]);  
                //send_new[l_n++] = ubuf(atom->nspecial[i][2]).d;
                for (m = 0; m < atom->nspecial[i][2]; m++){
                  send_new[l_n++] = static_cast<double> (atom->special[i][m]);
		  //send_new[l_n++] = ubuf(atom->special[i][m]).d;
                } 
	      }

	      if (l_loc >= new_list_max){    // grow array of local IDs for returning valid tracking IDs for new particles
		new_list_max += BFF;
		memory->grow(list_new,new_list_max,"fix_inflow_periodic:list_new"); 
	      }
	      list_new[l_loc++] = i;
	      l_nn++;   // increment the number of new particles
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
    ll[2] = l_nn;
    ll[3] = 0;
    MPI_Gather(&ll[0],3,MPI_INT,send_counts,3,MPI_INT,0,bc_send_comm); // gather information about sizes
    MPI_Reduce(&l_en,&ll[3],1,MPI_INT,MPI_SUM,0,bc_send_comm);
    ll[0] = ll[1] = ll[2] = 0;
    if (bc_send_comm_rank == 0)
      for (i = 0; i < bc_send_comm_size; ++i){           // calculate the total volume of data corresponding to new and old particles which need to be tracked
	ll[0] += send_counts[3*i];
	ll[1] += send_counts[3*i+1];
	ll[2] += send_counts[3*i+2];
      }
    MPI_Bcast(&ll[0],4,MPI_INT,0,bc_send_comm);     // broadcast basic information within a sending group 

    if (bc_send_comm_rank == 0){                 
      MPI_Send(&ll[0],4,MPI_INT,recv_partner,send_partner,world);      // send basic information about the volume of data exchange
      i = MAX(ll[0],ll[1]) + 1;
      if (i >= send_max){                              // grow sending buffer if needed  
	send_max = i + BFF; 
	memory->grow(buf_send,send_max,"fix_inflow_periodic:buf_send");
      }         
    }  

    if (ll[0]){
      if (bc_send_comm_rank == 0){
	offset = 0;
	for (i = 0; i < bc_send_comm_size; ++i){       // create rcounts and displs for coming MPI_Gatherv to collect new particles
	  displs[i] = offset;
	  rcounts[i] = send_counts[3*i];
	  offset += rcounts[i];
	} 
      }
      MPI_Gatherv(send_new,l_n,MPI_DOUBLE,buf_send,rcounts,displs,MPI_DOUBLE,0,bc_send_comm); // gather new particles

      if (bc_send_comm_rank == 0){                 
	offset = 0;
	for (i = 0; i < bc_send_comm_size; ++i){       // create rcounts and displs for coming MPI_Scatterv to send valid tracking IDs for new particles 
	  displs[i] = offset;
	  rcounts[i] = send_counts[3*i+2];
	  offset += rcounts[i];
	} 
	kk = offset;
	if (offset >= send_back_max){                  // grow array for sending back valid tracking IDs for new particles if needed
	  send_back_max = offset + BFF; 
	  memory->grow(buf_send_back,send_back_max,"fix_inflow_periodic:buf_send_back");
	}        
	k = 0;
	m = 0;
	for (i = 0; i < sp_max; ++i){                 // create new particle IDs for tracking
	  if (sp_list[i] < 0){
            ml = static_cast<tagint>(buf_send[m+1]);
	    //ml = (tagint) ubuf(buf_send[m+1]).i;
            buf_send[m] = static_cast<double> (i);
	    //buf_send[m] = ubuf(i).d; 
	    if (ml)
	      m += correct_mol_tags(m,buf_send);
            else
	      m += part_size;
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
            ml = static_cast<tagint>(buf_send[m+1]); 
	    //ml = (tagint) ubuf(buf_send[m+1]).i;
            buf_send[m] = static_cast<double> (i);
	    //buf_send[m] = ubuf(i).d;
	    if (ml)
	      m += correct_mol_tags(m,buf_send);
            else
	      m += part_size;
	    sp_list[i] = 1;
	    buf_send_back[k] = i; 
	    k++;   
	  }  
	}
      }
      MPI_Scatterv(buf_send_back,rcounts,displs,MPI_INT,sp_recv,l_loc,MPI_INT,0,bc_send_comm);    // send valid tracking IDs for new particles 
      for (i = 0; i < l_loc; ++i)
	sp[list_new[i]] = sp_recv[i];

      if (bc_send_comm_rank == 0){ 
	buf_send[ll[0]] = static_cast<double> (sp_max);
	//buf_send[ll[0]] = ubuf(sp_max).d;
	MPI_Send(buf_send,ll[0]+1,MPI_DOUBLE,recv_partner,send_partner,world);   // send the data about new particles 
      }
    }

    if (ll[1]){
      if (bc_send_comm_rank == 0){
	offset = 0;
	for (i = 0; i < bc_send_comm_size; ++i){       // create rcounts and displs for coming MPI_Gatherv to collect already tracked particles
	  displs[i] = offset;
	  rcounts[i] = send_counts[3*i+1];
	  offset += rcounts[i];
	} 
      }
      MPI_Gatherv(send_exist,l_e,MPI_DOUBLE,buf_send,rcounts,displs,MPI_DOUBLE,0,bc_send_comm); // gather already tracked particles

      if (bc_send_comm_rank == 0){
	k = 0;
	for (i = 0; i < ll[3]; ++i){              // refresh the local list of tracked particles
	  l = static_cast<int>(buf_send[k]);
	  //l = (int) ubuf(buf_send[k]).i;            
	  if (buf_send[k+1] < 0.0){
	    sp_list[l] = -1; 
	    k += 2;
	  } else 
	    k += part_size_track;   
	}
	MPI_Send(buf_send,ll[1],MPI_DOUBLE,recv_partner,recv_partner,world);   // send the data about already tracked particles  
      }
    }

    if (bc_send_comm_rank == 0){     // used for consistency checks
      tot_tracked_part = 0;
      for (i = 0; i < sp_max; ++i)
	if (sp_list[i] > 0)
	  tot_tracked_part++;
    }
  }

  if (bc_recv_comm_rank > -1){  // only cores which participate in receiving the data

    if (bc_recv_comm_rank == 0)
      MPI_Recv(&ll[0],4,MPI_INT,send_partner,send_partner,world,MPI_STATUS_IGNORE);    // receive basic information about the volume of data exchange

    MPI_Bcast(&ll[0],4,MPI_INT,0,bc_recv_comm);    // broadcast basic information within a receiving group 
    i = MAX(ll[0],ll[1]) + 1;
    if (i >= recv_max){
      recv_max = i + BFF; 
      memory->grow(buf_recv,recv_max,"fix_inflow_periodic:buf_recv");
    }

    if (ll[0]){
      if (bc_recv_comm_rank == 0)
	MPI_Recv(buf_recv,ll[0]+1,MPI_DOUBLE,send_partner,send_partner,world,MPI_STATUS_IGNORE);    // receive the data about new particles 

      MPI_Bcast(buf_recv,ll[0]+1,MPI_DOUBLE,0,bc_recv_comm);    // broadcast the data for new particles  
      k = 0; 
      for (i = 0; i < ll[2]; ++i){                      // process the data for new particles
	for (j = 0; j < 3; ++j)
	  xh[j] = buf_recv[k + 4 + j];

	ind = 0;
	if (comm->user_part){                        // check if a new particle is mine and has to be inserted 
	  l = comm->coords_to_bin(xh,nn,comm->nbp_loc,comm->box_min);
	  if (l)
	    if (comm->m_comm[nn[0]][nn[1]][nn[2]] & comm->local_bit) ind = 1; 
	} else{
	  if (xh[0] >= sublo[0] && xh[0] < subhi[0] && xh[1] >= sublo[1] && xh[1] < subhi[1] && xh[2] >= sublo[2] && xh[2] < subhi[2]) ind = 1;
	}

	if (ind){                             // insert a new particle
	  l = static_cast<int>(buf_recv[k+2]);
	  //l = (int) ubuf(buf_recv[k+2]).i;
	  atom->avec->create_atom(l,xh);  // create new atom
	  nlocal = atom->nlocal;
	  kk = nlocal - 1;             
	  sp[kk] = static_cast<int>(buf_recv[k]); 
	  //sp[kk] = (int) ubuf(buf_recv[k]).i;  // assign tracking id
          molecule[kk] = static_cast<tagint>(buf_recv[k+1]);
          //molecule[kk] = (tagint) ubuf(buf_recv[k+1]).i;  // assign molecule id
	  mask[kk] = static_cast<int>(buf_recv[k+3]);  
	  //mask[kk] = (int) ubuf(buf_recv[k+3]).i;  
	  for (j = 0; j < 3; ++j)      // assign velocities
	    v[kk][j] = buf_recv[k + 7 + j];
	  if (spin_ind)  // assign spin velocities if needed
	    for (j = 0; j < 3; ++j)
	      omega[kk][j] = buf_recv[k + 10 + j];
 
          k += part_size;
          if (molecule[kk]){  // assign molecular data
            tag[kk] = static_cast<tagint>(buf_recv[k++]);
            //tag[kk] = (tagint) ubuf(buf_recv[k++]).i;   // assign tag
            atom->num_bond[kk] = static_cast<int>(buf_recv[k++]);
            //atom->num_bond[kk] = (int) ubuf(buf_recv[k++]).i;  // assign bond data
            for (m = 0; m < atom->num_bond[kk]; m++){
              atom->bond_type[kk][m] = static_cast<int>(buf_recv[k++]); 
              //atom->bond_type[kk][m] = (int) ubuf(buf_recv[k++]).i;
              atom->bond_atom[kk][m] = static_cast<tagint>(buf_recv[k++]);
              //atom->bond_atom[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
            }
            atom->num_angle[kk] = static_cast<int>(buf_recv[k++]);
            //atom->num_angle[kk] = (int) ubuf(buf_recv[k++]).i;  // assign angle data
            for (m = 0; m < atom->num_angle[kk]; m++){
              atom->angle_type[kk][m] = static_cast<int>(buf_recv[k++]);
              //atom->angle_type[kk][m] = (int) ubuf(buf_recv[k++]).i;
              atom->angle_atom1[kk][m] = static_cast<tagint>(buf_recv[k++]); 
              //atom->angle_atom1[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
              atom->angle_atom2[kk][m] = static_cast<tagint>(buf_recv[k++]);
              //atom->angle_atom2[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
              atom->angle_atom3[kk][m] = static_cast<tagint>(buf_recv[k++]); 
              //atom->angle_atom3[kk][m] = (tagint) ubuf(buf_recv[k++]).i;     
            }
            atom->num_dihedral[kk] = static_cast<int>(buf_recv[k++]);
            //atom->num_dihedral[kk] = (int) ubuf(buf_recv[k++]).i;  // assign dihedral data
            for (m = 0; m < atom->num_dihedral[kk]; m++){
              atom->dihedral_type[kk][m] = static_cast<int>(buf_recv[k++]);
              //atom->dihedral_type[kk][m] = (int) ubuf(buf_recv[k++]).i;
              atom->dihedral_atom1[kk][m] = static_cast<tagint>(buf_recv[k++]);
              //atom->dihedral_atom1[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
              atom->dihedral_atom2[kk][m] = static_cast<tagint>(buf_recv[k++]);
              //atom->dihedral_atom2[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
              atom->dihedral_atom3[kk][m] = static_cast<tagint>(buf_recv[k++]);
              //atom->dihedral_atom3[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
              atom->dihedral_atom4[kk][m] = static_cast<tagint>(buf_recv[k++]);
              //atom->dihedral_atom4[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
            }  
            atom->num_improper[kk] = static_cast<int>(buf_recv[k++]); 
            //atom->num_improper[kk] = (int) ubuf(buf_recv[k++]).i; // assign improper data
            for (m = 0; m < atom->num_improper[kk]; m++){
              atom->improper_type[kk][m] = static_cast<int>(buf_recv[k++]);
              //atom->improper_type[kk][m] = (int) ubuf(buf_recv[k++]).i;
              atom->improper_atom1[kk][m] = static_cast<tagint>(buf_recv[k++]); 
              //atom->improper_atom1[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
              atom->improper_atom2[kk][m] = static_cast<tagint>(buf_recv[k++]); 
              //atom->improper_atom2[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
              atom->improper_atom3[kk][m] = static_cast<tagint>(buf_recv[k++]); 
              //atom->improper_atom3[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
              atom->improper_atom4[kk][m] = static_cast<tagint>(buf_recv[k++]);
              //atom->improper_atom4[kk][m] = (tagint) ubuf(buf_recv[k++]).i;
            }
            if (atom->individual){  // assign individual data
              for (m = 0; m < atom->num_bond[kk]; m++)
                atom->bond_length[kk][m] = buf_recv[k++];
              for (m = 0; m < atom->num_angle[kk]; m++)
                atom->angle_area[kk][m] = buf_recv[k++];
              for (m = 0; m < atom->num_dihedral[kk]; m++)
                atom->dihedral_angle[kk][m] = buf_recv[k++];
            }
            atom->nspecial[kk][0] = static_cast<int>(buf_recv[k++]);
            //atom->nspecial[kk][0] = (int) ubuf(buf_recv[k++]).i;
            atom->nspecial[kk][1] = static_cast<int>(buf_recv[k++]);
            //atom->nspecial[kk][1] = (int) ubuf(buf_recv[k++]).i;
            atom->nspecial[kk][2] = static_cast<int>(buf_recv[k++]);
            //atom->nspecial[kk][2] = (int) ubuf(buf_recv[k++]).i;
            for (m = 0; m < atom->nspecial[kk][2]; m++){
              atom->special[kk][m] = static_cast<tagint>(buf_recv[k++]);
              //atom->special[kk][m] = (tagint) ubuf(buf_recv[k++]).i;  
            }
          } 
	} else{ // skip particle data if it is not mine
          ml = static_cast<tagint>(buf_recv[k+1]);
          //ml = (tagint) ubuf(buf_recv[k+1]).i; 
          k += part_size;
          if (ml){
            k++;
            j = static_cast<int>(buf_recv[k++]);
            //j = (int) ubuf(buf_recv[k++]).i;
            m = j;
            k += 2*j;
            j = static_cast<int>(buf_recv[k++]);
            //j = (int) ubuf(buf_recv[k++]).i;
            m += j;
            k += 4*j;
            j = static_cast<int>(buf_recv[k++]);
            //j = (int) ubuf(buf_recv[k++]).i;
            m += j;
            k += 5*j;
            j = static_cast<int>(buf_recv[k++]);
            //j = (int) ubuf(buf_recv[k++]).i;
            k += 5*j;
            if (atom->individual)
              k += m;
            k += 2;
            j = static_cast<int>(buf_recv[k++]);
            //j = (int) ubuf(buf_recv[k++]).i;
            k += j;            
          }
        }
      }

      l_loc = static_cast<int>(buf_recv[ll[0]]);
      //l_loc = (int) ubuf(buf_recv[ll[0]]).i; 
      if (l_loc > sp_max){ 
	j = sp_max;
	sp_max = l_loc;
	memory->grow(sp_list,sp_max,"fix_inflow_periodic:sp_list"); 
	for (i = j; i < sp_max; ++i)
	  sp_list[i] = -2;
      }
    }

    if (ll[1]){
      if (bc_recv_comm_rank == 0)
	MPI_Recv(buf_recv,ll[1],MPI_DOUBLE,send_partner,recv_partner,world,MPI_STATUS_IGNORE);    // receive the data about new particles

      MPI_Bcast(buf_recv,ll[1],MPI_DOUBLE,0,bc_recv_comm);    // broadcast the data for already tracked particles 

      k = 0;
      for (i = 0; i < ll[3]; ++i){              // unpack and assign the data of tracked particles
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

  ntemp = atom->nlocal;
  MPI_Allreduce(&ntemp,&atom->natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (atom->natoms < 0 || atom->natoms >= MAXBIGINT)
    error->all(FLERR,"Too many total atoms");

  if (n_mol_types > 0)
    if (step_t > step_start && step_t%mol_adapt_iter == 0)
      periodic_mol_reordering();

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
  int i,j,k,l,kk,offset;
  int l_e, ll;
  double **x = atom->x;
  double **v = atom->v;
  double **omega = atom->omega;
  int *sp = atom->spin;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // exchange of data between beginnings and ends of periodical sections

  if (bc_send_comm_rank > -1){  // only cores which participate in sending the data
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
    MPI_Gather(&l_e,1,MPI_INT,send_counts,1,MPI_INT,0,bc_send_comm); // gather information about sizes

    ll = 0; 
    if (bc_send_comm_rank == 0){
      for (i = 0; i < bc_send_comm_size; ++i)           // calculate the total data volume which is tracked
	ll += send_counts[i];
      k = static_cast<int>((ll+1)/part_size_vel);
      if (k != tot_tracked_part) 
	error->warning(FLERR,"Did not pass first consistency check!!! Number of tracked particles in the function end_of_step() is different from tot_tracked_part!");  
    }
    MPI_Bcast(&ll,1,MPI_INT,0,bc_send_comm);     // broadcast basic information within a sending group  

    if (bc_send_comm_rank == 0){                 
      MPI_Send(&ll,1,MPI_INT,recv_partner,send_partner,world);      // send basic information about the volume of data exchange
      if (ll >= send_max){                              // grow sending buffer if needed  
	send_max = ll + BFF; 
	memory->grow(buf_send,send_max,"fix_inflow_periodic:buf_send");
      }         
    }  

    if (ll){
      if (bc_send_comm_rank == 0){
	offset = 0;
	for (i = 0; i < bc_send_comm_size; ++i){       // create rcounts and displs for coming MPI_Gatherv to collect already tracked particles
	  displs[i] = offset;
	  rcounts[i] = send_counts[i];
	  offset += rcounts[i];
	} 
      }
      MPI_Gatherv(send_exist,l_e,MPI_DOUBLE,buf_send,rcounts,displs,MPI_DOUBLE,0,bc_send_comm); // gather already tracked particles

      if (bc_send_comm_rank == 0)
	MPI_Send(buf_send,ll,MPI_DOUBLE,recv_partner,recv_partner,world);   // send the data about already tracked particles  	
    }
  }

  if (bc_recv_comm_rank > -1){  // only cores which participate in receiving the data
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
    MPI_Reduce(&k,&l,1,MPI_INT,MPI_SUM,0,bc_recv_comm);     // used for a consistency check 

    if (bc_recv_comm_rank == 0){
      MPI_Recv(&ll,1,MPI_INT,send_partner,send_partner,world,MPI_STATUS_IGNORE);    // receive basic information about the volume of data exchange
      k = static_cast<int>((ll+1)/part_size_vel);
      if (k != l) 
	error->warning(FLERR,"Did not pass second consistency check!!! Number of tracked particles in the function end_of_step() is different at sending and receiving groups!"); 
    }

    MPI_Bcast(&ll,1,MPI_INT,0,bc_recv_comm);    // broadcast basic information within a receiving group 
    if (ll >= recv_max){
      recv_max = ll + BFF; 
      memory->grow(buf_recv,recv_max,"fix_inflow_periodic:buf_recv");
    }

    if (ll){
      if (bc_recv_comm_rank == 0)
	MPI_Recv(buf_recv,ll,MPI_DOUBLE,send_partner,recv_partner,world,MPI_STATUS_IGNORE);    // receive the data about new particles

      MPI_Bcast(buf_recv,ll,MPI_DOUBLE,0,bc_recv_comm);    // broadcast the data for already tracked particles 

      l_e = static_cast<int>((ll+1)/part_size_vel);
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

  //consistency_check(2);
}

/*------------------------------------------------------------------------*/

void FixInflowPeriodic::reset_target(double rho_new)
{
  int i,j;
  double xh[3],rr;

  double **x = atom->x;
  int *mask = atom->mask;
  double *rho = atom->rho;
  int nall = atom->nlocal + atom->nghost;

  if (bc_recv_comm_rank > -1)
    for (i = 0; i < nall; ++i)
      if (mask[i] & groupbit){
        for (j = 0; j < 3; ++j)
          xh[j] = x[i][j] - x0[ninf][j];
        rot_forward(xh[0],xh[1],xh[2],ninf);
        rr = xh[0]*xh[0] + xh[1]*xh[1];
        if (xh[2] >= 0.0 && xh[2] < cut && rr < aa[ninf][3]*aa[ninf][3])
          rho[i] = rho_new;
      }
}

/*------------------------------------------------------------------------*/

void FixInflowPeriodic::set_comm_groups()
{
  int i,j,k,l,ii,nn[3],jj[3],color,ind;
  double xx[3],ddx[3],rr;
  double rad = sqrt(3.0)*binsize;
  double lm = cut+skin+rad;
  int *tmp,*tmp1;  

  tmp = tmp1 = NULL;

  for (j = 0; j < 3; ++j){                  // setup bins
    ddx[j] = subhi[j] - sublo[j];
    nn[j] = static_cast<int>(ddx[j]/binsize + 3.0);
  }
  if (domain->dimension == 2)
    nn[2] = 1;

  ninf = -1;
  k = 0;
  color = 0;
  for (ii = 0; ii < num_inflow; ++ii){               
    ind = 0;
    for (jj[0] = 0; jj[0] < nn[0]; ++jj[0]){             // check if certain bins overlap with a core
      for (jj[1] = 0; jj[1] < nn[1]; ++jj[1]){
        for (jj[2] = 0; jj[2] < nn[2]; ++jj[2]){
          for (l = 0; l < 3; ++l)
            xx[l] = sublo[l] + (jj[l]-0.5)*binsize - x0[ii][l]; 
          if (domain->dimension == 2)
            xx[2] = sublo[2] - x0[ii][2];
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
    
    if (ind){
      color = ii+1;
      ninf = ii;
      k++;
    }
  }

  MPI_Comm_split(world,color,comm->me,&bc_send_comm);                     // create new communicators
  if (color > 0){
    MPI_Comm_rank(bc_send_comm,&bc_send_comm_rank);
    MPI_Comm_size(bc_send_comm,&bc_send_comm_size);
  } else{
    bc_send_comm = MPI_COMM_NULL;
    bc_send_comm_rank = -1;
    bc_send_comm_size = 0;
  }

  color = 0;
  if (bc_send_comm_rank == 0){ 
    if (bc_send_comm_size <= 0)
      error->one(FLERR,"The size of a sending group cannot be a non-negative number!");    
    memory->create(send_counts,3*bc_send_comm_size,"fix_inflow_periodic:send_counts");
    memory->create(rcounts,bc_send_comm_size,"fix_inflow_periodic:rcounts");
    memory->create(displs,bc_send_comm_size,"fix_inflow_periodic:displs");  
    send_partner = comm->me;
    color = 1;
  }

  MPI_Comm_split(world,color,comm->me,&bc_send_zero_comm);                     // create new communicator from zeroes of sending cores
  if (color > 0){
    MPI_Comm_rank(bc_send_zero_comm,&bc_send_zero_comm_rank);
    MPI_Comm_size(bc_send_zero_comm,&bc_send_zero_comm_size);
  } else{
    bc_send_zero_comm = MPI_COMM_NULL;
    bc_send_zero_comm_rank = -1;
    bc_send_zero_comm_size = 0;
  }

  if (bc_send_zero_comm_rank == 0)
    color = comm->me;
  else
    color = -1;
  MPI_Allreduce(&color,&glob_zero,1,MPI_INT,MPI_MAX,world);

  color = 0;
  for (ii = 0; ii < num_inflow; ++ii){ 
    ind = 0;
    for (jj[0] = 0; jj[0] < nn[0]; ++jj[0]){                             // check if certain bins overlap with a core
      for (jj[1] = 0; jj[1] < nn[1]; ++jj[1]){
        for (jj[2] = 0; jj[2] < nn[2]; ++jj[2]){
          for (l = 0; l < 3; ++l)
            xx[l] = sublo[l] + (jj[l]-0.5)*binsize - x0[ii][l];
          if (domain->dimension == 2)
            xx[2] = sublo[2] - x0[ii][2];
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

    if (ind){
      color = ii+1;
      ninf = ii;
      k++;
    }
  }

  MPI_Comm_split(world,color,comm->me,&bc_recv_comm);              // create new communicators
  if (color > 0){
    MPI_Comm_rank(bc_recv_comm,&bc_recv_comm_rank);
    MPI_Comm_size(bc_recv_comm,&bc_recv_comm_size);
  } else{
    bc_recv_comm = MPI_COMM_NULL;
    bc_recv_comm_rank = -1;
    bc_recv_comm_size = 0;
  }

  if (bc_recv_comm_rank == 0){
    if (bc_recv_comm_size <= 0)
      error->one(FLERR,"The size of a receiving group cannot be a non-negative number!");
    memory->create(send_counts,bc_recv_comm_size,"fix_inflow_periodic:send_counts");
    memory->create(rcounts,bc_recv_comm_size,"fix_inflow_periodic:rcounts");
    memory->create(displs,bc_recv_comm_size,"fix_inflow_periodic:displs");
    recv_partner = comm->me;
  }

  if (k > 1)
    error->one(FLERR,"There are cores, which may participate in multiple inflows (or their beginning/ending)! This will not function properly in fix inflow/periodic!");

  memory->create(tmp,2*num_inflow,"fix_inflow_periodic:tmp");
  memory->create(tmp1,2*num_inflow,"fix_inflow_periodic:tmp1");
  for (ii = 0; ii < 2*num_inflow; ++ii){ 
    tmp[ii] = -1;
    tmp1[ii] = -1;
  }

  if (bc_send_comm_rank == 0)
    tmp[ninf] = send_partner;
  if (bc_recv_comm_rank == 0)  
    tmp[num_inflow + ninf] = recv_partner;
  MPI_Allreduce(tmp,tmp1,2*num_inflow,MPI_INT,MPI_MAX,world);
  if (bc_recv_comm_rank == 0)      // assign sending partners
    send_partner = tmp1[ninf];
  if (bc_send_comm_rank == 0)      // assign receiving partners
    recv_partner = tmp1[num_inflow + ninf];

  memory->destroy(tmp);
  memory->destroy(tmp1);  
 
  consistency_check(0);
}

/*------------------------------------------------------------------------
  initial reodering of molecules and their tags in order to assign all 
  molecules to certain inflows 
  each inflow has lists of molecule ids for different molecule types
------------------------------------------------------------------------*/ 
  
void FixInflowPeriodic::initial_mol_order_setup()
{
  tagint i, l;
  int j, k, ii, m;
  int nlocal = atom->nlocal;
  tagint *mol_size = atom->mol_size;
  tagint *molecule = atom->molecule;
  tagint *tag = atom->tag;
  int *mol_type = atom->mol_type;
  tagint *ll_mol_rest, *mol_tag, *mol_tag_recv;
  int *mol_inf, *mol_inf_recv;
  double **x = atom->x;
  double xx[3], rr;

  mol_inf = mol_inf_recv = NULL;
  mol_tag = mol_tag_recv = ll_mol_rest = NULL;

  step_start = update->ntimestep;

  n_mol_max = atom->n_mol_max; // assign basic parameters
  n_mol_types = atom->n_mol_types;
  if (n_mol_types == 0)
    return; 
 
  memory->create(ll_mol_rest,n_mol_types,"fix_inflow_periodic:ll_mol_rest");
  for (j = 0; j < n_mol_types; ++j)
    ll_mol_rest[j] = 0;
    
  memory->create(mol_tag,n_mol_max,"fix_inflow_periodic:mol_tag");  // initialize arrays for a further use
  memory->create(mol_tag_recv,n_mol_max,"fix_inflow_periodic:mol_tag_recv");
  memory->create(mol_inf,n_mol_max,"fix_inflow_periodic:mol_inf");
  memory->create(mol_inf_recv,n_mol_max,"fix_inflow_periodic:mol_inf_recv");
  for (i = 0; i < n_mol_max; ++i){  // set initial values to -1
    mol_tag[i] = -1;
    mol_tag_recv[i] = -1;
    mol_inf[i] = -1;
    mol_inf_recv[i] = -1;
  }

  for (j = 0; j < nlocal; ++j){  // loop over local particles to find which molecules belong to certain inflows
    i = molecule[j] - 1; 
    if (i > -1){
      
      if (tag[j] > mol_tag[i])
        mol_tag[i] = tag[j]; 
	
      for (ii = 0; ii < num_inflow; ++ii){ // check if a particle belongs to certain inflow              
        for (k = 0; k < 3; ++k)
          xx[k] = x[j][k] - x0[ii][k]; 
        rot_forward(xx[0],xx[1],xx[2],ii);
        rr = sqrt(xx[0]*xx[0] + xx[1]*xx[1]);
        if (rr < aa[ii][3] && xx[2] >= 0.0 && xx[2] < aa[ii][4]){
          mol_inf[i] = ii;
          break;
        }
      }
    } else
      tag[j] = 0; 
  }
  MPI_Allreduce(mol_inf,mol_inf_recv,n_mol_max,MPI_INT,MPI_MAX,world);  // reduce an inflow list
  MPI_Allreduce(mol_tag,mol_tag_recv,n_mol_max,MPI_LMP_TAGINT,MPI_MAX,world); // reduce a max tag list 
 
  memory->create(num_mols,num_inflow,n_mol_types,"fix_inflow_periodic:num_mols"); // initialize arrays used for molecule ids and tags
  memory->create(num_mols_prev,num_inflow,n_mol_types,"fix_inflow_periodic:num_mols_prev");
  memory->create(mol_st,num_inflow,n_mol_types,"fix_inflow_periodic:mol_st");
  memory->create(mol_st_prev,num_inflow,n_mol_types,"fix_inflow_periodic:mol_st_prev");    
  memory->create(tag_st,num_inflow,n_mol_types,"fix_inflow_periodic:tag_st");
  memory->create(tag_st_prev,num_inflow,n_mol_types,"fix_inflow_periodic:tag_st_prev");
  
  for (j = 0; j < num_inflow; ++j)
    for (k = 0; k < n_mol_types; ++k)
      num_mols[j][k] = 0;
  
  for (i = 0; i < n_mol_max; ++i)  // calculate the number of molecules per every inflow and calculate the number of those which were not assigned yet 
    if (mol_tag_recv[i] > -1){
      k = mol_type[i];
      mol_tag[i] = mol_tag_recv[i] - mol_size[k] + 1;
      if (mol_inf_recv[i] == -1)
        ll_mol_rest[k]++;
      else
        num_mols[mol_inf_recv[i]][k]++; 
    }
 
  for (k = 0; k < n_mol_types; ++k){  // distribute the numbers of not assigned molecules among inflows
    i = static_cast<tagint>(ll_mol_rest[k]/num_inflow);
    ii = ll_mol_rest[k] - i*num_inflow;
    for (j = 0; j < num_inflow; ++j){
      num_mols[j][k] += MOLINC; 
      num_mols_prev[j][k] = i; 
      if (j < ii)
        num_mols_prev[j][k]++; 
    }
  }

  for (i = 0; i < n_mol_max; ++i)  // finally assign all molecules to certain inflows
    if (mol_tag_recv[i] > -1 && mol_inf_recv[i] == -1){
      k = mol_type[i];
      for (j = 0; j < num_inflow; ++j)
        if (num_mols_prev[j][k] > 0){
          num_mols_prev[j][k]--;
          num_mols[j][k]++;
          mol_inf_recv[i] = j;
          break; 
        }
    }

  i = l = 1;
  for (j = 0; j < num_inflow; ++j)  // assign the starting tags and molecule ids based on the number of molecules per inflow 
    for (k = 0; k < n_mol_types; ++k){
      mol_st[j][k] = i;
      tag_st[j][k] = l;
      i += num_mols[j][k];
      l += num_mols[j][k]*mol_size[k];
    }
 
  memory->create(mol_list,n_mol_max,"fix_inflow_periodic:mol_list");  
  for (i = 0; i < n_mol_max; ++i){  // define new molecular ids
    mol_tag_recv[i] = mol_type[i];  
    j = mol_inf_recv[i];
    mol_list[i] = j;  
    if (j > -1){
      k = mol_type[i];
      mol_inf[i] = mol_st[j][k] + num_mols_prev[j][k];
      num_mols_prev[j][k]++;
    }
  }

  for (j = 0; j < nlocal; ++j){  // shift molecular ids and tags according to the new distribution of tags
    i = molecule[j] - 1; 
    if (i > -1){
      ii = mol_inf_recv[i]; 
      k = mol_type[i];
      l = tag_st[ii][k] + (mol_inf[i] - mol_st[ii][k])*mol_size[k] - mol_tag[i];  // tag shift
      tag[j] += l;
      for (m = 0; m < atom->num_bond[j]; m++) // all bonds
        atom->bond_atom[j][m] += l;
      for (m = 0; m < atom->num_angle[j]; m++){  // all angles
        atom->angle_atom1[j][m] += l;
        atom->angle_atom2[j][m] += l;
        atom->angle_atom3[j][m] += l;
      }
      for (m = 0; m < atom->num_dihedral[j]; m++){  // all dihedrals
        atom->dihedral_atom1[j][m] += l;
        atom->dihedral_atom2[j][m] += l;
        atom->dihedral_atom3[j][m] += l;
        atom->dihedral_atom4[j][m] += l;
      }
      for (m = 0; m < atom->num_improper[j]; m++){
        atom->improper_atom1[j][m] += l;
        atom->improper_atom2[j][m] += l;
        atom->improper_atom3[j][m] += l;
        atom->improper_atom4[j][m] += l;
      }
      for (m = 0; m < atom->nspecial[j][2]; m++)
	atom->special[j][m] += l;
      molecule[j] = mol_inf[i];  // assign new molecule id
    }
  }

  l = mol_st[num_inflow-1][n_mol_types-1] + num_mols[num_inflow-1][n_mol_types-1] - 1;
  if (l > n_mol_max){  // grow the mol_type and mol_list (inflow id) arrays
    atom->n_mol_max = l;
    memory->grow(atom->mol_type,l,"atom:mol_type");
    memory->grow(mol_list,l,"fix_inflow_periodic:mol_list");  
    for (i = 0; i < l; ++i){ // initialize all values to -1
      atom->mol_type[i] = -1;
      mol_list[i] = -1;    
    }
    for (i = 0; i < n_mol_max; ++i)  // fill in the arrays with shifted mol types and inflow ids 
      if (mol_tag_recv[i] > -1){
        l = mol_inf[i]-1;
        atom->mol_type[l] = mol_tag_recv[i];
        mol_list[l] = mol_inf_recv[i];  
      }
    n_mol_max = atom->n_mol_max;
  }

  memory->destroy(mol_tag);
  memory->destroy(mol_tag_recv);
  memory->destroy(mol_inf);
  memory->destroy(mol_inf_recv);
  memory->destroy(ll_mol_rest);

  if (bc_send_comm_rank == 0){  // initialize correspondence list only at the root of a sending group
    n_mol_corresp_max = MOLINC;
    memory->create(mol_corresp_list,2*n_mol_corresp_max,"fix_inflow_periodic:mol_corresp_list");
  }

  /*char f_name[FILENAME_MAX];
  FILE* out_stat;
  sprintf(f_name,"initial_%d.dat",comm->me);
  out_stat=fopen(f_name,"w");
  fprintf(out_stat,"n_mol_max=%d, n_mol_types=%d, num_inflow=%d  \n", n_mol_max, n_mol_types, num_inflow);
  fprintf(out_stat,"mol types and lists  \n");
  for (i = 0; i < n_mol_max; i++)
    fprintf(out_stat,"%d %d %d \n",i,atom->mol_type[i],mol_list[i]);
  fprintf(out_stat,"numbers  \n");
  for (i = 0; i < num_inflow; ++i)
    for (j = 0; j < n_mol_types; ++j)
      fprintf(out_stat,"%d %d %d %d %d %d \n",i,j,mol_size[j],num_mols[i][j],mol_st[i][j],tag_st[i][j]);
  fprintf(out_stat,"mol data  \n");
  for (j = 0; j < nlocal; ++j){ 
    i = molecule[j] - 1;
    if (i > -1)
      fprintf(out_stat,"%d %d %d x=%f %f %f \n",j,molecule[j],tag[j],x[j][0],x[j][1],x[j][2]);
  }
  fclose(out_stat);*/
}

/*------------------------------------------------------------------------
  reodering of molecules (if needed) in order to extend molecule id arrays 
  renewal of molecule correspondence arrays
------------------------------------------------------------------------*/

void FixInflowPeriodic::periodic_mol_reordering()
{
  tagint i, l, tt, mm;
  int j, k, m, ii, ind;
  int *rcounts1, *displs1;
  int nlocal = atom->nlocal;
  tagint *mol_size = atom->mol_size;
  tagint *molecule = atom->molecule;
  tagint *tag = atom->tag;
  int *mol_type = atom->mol_type;
  int *mol_tag, *mol_tag_recv, *mol_inf, *ind_corr;
  char str[200];
  double **x = atom->x;
  double xx[3], rr;
  double lm = cut+skin;  

  /*char f_name[FILENAME_MAX];
  FILE* out_stat;
  sprintf(f_name,"before_%d_%d.dat",comm->me,update->ntimestep);
  out_stat=fopen(f_name,"w");
  fprintf(out_stat,"n_mol_max=%d, n_mol_types=%d, num_inflow=%d  \n", n_mol_max, n_mol_types, num_inflow);
  fprintf(out_stat,"mol types and lists  \n");
  for (i = 0; i < n_mol_max; i++)
    fprintf(out_stat,"%d %d %d \n",i,atom->mol_type[i],mol_list[i]);
  fprintf(out_stat,"numbers  \n");
  for (i = 0; i < num_inflow; ++i)
    for (j = 0; j < n_mol_types; ++j)
      fprintf(out_stat,"%d %d %d %d %d %d \n",i,j,mol_size[j],num_mols[i][j],mol_st[i][j],tag_st[i][j]);
  fprintf(out_stat,"mol data  \n");
  for (j = 0; j < nlocal; ++j){
    i = molecule[j] - 1;
    if (i > -1)
      fprintf(out_stat,"%d %d %d x=%f %f %f \n",j,molecule[j],tag[j],x[j][0],x[j][1],x[j][2]);
  }
  fprintf(out_stat,"n_mol_corresp=%d  \n",n_mol_corresp);
  for (j = 0; j < n_mol_corresp; ++j)
    fprintf(out_stat,"%d %d %d \n",j,mol_corresp_list[2*j],mol_corresp_list[2*j+1]);
  fprintf(out_stat,"atom->n_mol_corresp_glob=%d  \n",atom->n_mol_corresp_glob);
  for (j = 0; j < atom->n_mol_corresp_glob; ++j)
    fprintf(out_stat,"%d %d %d \n",j,atom->mol_corresp_glob[2*j],atom->mol_corresp_glob[2*j+1]);
    fclose(out_stat);*/

  atom->fix_force_bound_ind = 1;
  mol_tag = mol_tag_recv = mol_inf = ind_corr = NULL;
  rcounts1 = displs1 = NULL;
  
  memory->create(mol_tag,n_mol_max,"fix_inflow_periodic:mol_tag");
  memory->create(mol_tag_recv,n_mol_max,"fix_inflow_periodic:mol_tag_recv");
  memory->create(mol_inf,n_mol_max,"fix_inflow_periodic:mol_inf");  
  for (i = 0; i < n_mol_max; ++i){ // initialize all values to -1
    mol_tag[i] = -1;
    mol_tag_recv[i] = -1;
  }

  for (j = 0; j < nlocal; ++j){ // find all existing molecules
    i = molecule[j] - 1; 
    if (i > -1)
      mol_tag[i] = 1;
  }
  MPI_Allreduce(mol_tag,mol_tag_recv,n_mol_max,MPI_INT,MPI_MAX,world);  // reduce all existing molecules 

  for (i = 0; i < n_mol_max; ++i){  // initialize all values to -1
    mol_tag[i] = -1;
    mol_inf[i] = -1;
  }    
  MPI_Allreduce(mol_type,mol_tag,n_mol_max,MPI_INT,MPI_MAX,world);  // reduce mol types, which may have changed
  MPI_Allreduce(mol_list,mol_inf,n_mol_max,MPI_INT,MPI_MAX,world);  // reduce inflow ids, which may have changed  
  for (i = 0; i < n_mol_max; ++i){  // compare mol types and inflow ids against the existing molecules
    if (mol_tag_recv[i] > -1){
      if (mol_tag[i] < 0 || mol_tag[i] >= n_mol_types){  // possible error check regarding mol types
        sprintf(str,"mol_type = %d is wrong in FixInflowPeriodic::periodic_mol_reordering!!",mol_tag[i]);
        error->warning(FLERR,str);  
      }
      mol_type[i] = mol_tag[i];  
      if (mol_inf[i] < 0 || mol_inf[i] >= num_inflow){  // possible error check regarding inflow ids
        sprintf(str,"mol_inflow = %d is wrong in FixInflowPeriodic::periodic_mol_reordering!!",mol_inf[i]);
        error->warning(FLERR,str);  
      }  
      mol_list[i] = mol_inf[i];
    } else{
      mol_type[i] = -1;  // re-assign -1 to all non-existing molecules
      mol_list[i] = -1;          
    }
  }
    
  l = tt = mm = 0;
  ind = 0;
  for (j = 0; j < num_inflow; ++j)  // calculate molecule and tag shift, whenever an array extension is needed
    for (k = 0; k < n_mol_types; ++k){
      num_mols_prev[j][k] = 0;
      for (i = 0; i < num_mols[j][k]; ++i){  // the current number of molecules per inflow
        if (mol_tag_recv[l] == 1)
          num_mols_prev[j][k]++;
        l++;
      }
      mol_st_prev[j][k] = mm;  
      tag_st_prev[j][k] = tt;    
      mol_st[j][k] += mm;
      tag_st[j][k] += tt;    
      if (num_mols[j][k] - num_mols_prev[j][k] < MOLINC){  // perform an extension of mol numbers if needed
        mm += MOLINC;
        num_mols[j][k] += MOLINC;  
        tt += MOLINC*mol_size[k];
        ind = 1;
      }
    }

  if (ind){  // perform shifts of tags and molecule ids
    for (j = 0; j < nlocal; ++j){
      i = molecule[j] - 1; 
      if (i > -1){
        ii = mol_inf[i]; 
        k = mol_tag[i];
        l = tag_st_prev[ii][k]; // tag shift
        tag[j] += l;
        for (m = 0; m < atom->num_bond[j]; m++) // all bonds
          atom->bond_atom[j][m] += l;
        for (m = 0; m < atom->num_angle[j]; m++){  // all angles
          atom->angle_atom1[j][m] += l;
          atom->angle_atom2[j][m] += l;
          atom->angle_atom3[j][m] += l;
        }
        for (m = 0; m < atom->num_dihedral[j]; m++){  // all dihedrals
          atom->dihedral_atom1[j][m] += l;
          atom->dihedral_atom2[j][m] += l;
          atom->dihedral_atom3[j][m] += l;
          atom->dihedral_atom4[j][m] += l;
        }
        for (m = 0; m < atom->num_improper[j]; m++){
          atom->improper_atom1[j][m] += l;
          atom->improper_atom2[j][m] += l;
          atom->improper_atom3[j][m] += l;
          atom->improper_atom4[j][m] += l;
        }
	for (m = 0; m < atom->nspecial[j][2]; m++)
	  atom->special[j][m] += l;
        molecule[j] += mol_st_prev[ii][k];  // molecule id shift
      }
    }
 
    l = mol_st[num_inflow-1][n_mol_types-1] + num_mols[num_inflow-1][n_mol_types-1] - 1; 
    atom->n_mol_max = l;
    memory->grow(atom->mol_type,l,"atom:mol_type");  // extend mol_type and inflow id arrays
    memory->grow(mol_list,l,"fix_inflow_periodic:mol_list");  
    for (i = 0; i < l; ++i){  // initialize all values to -1
      atom->mol_type[i] = -1;
      mol_list[i] = -1;    
    }

    if (bc_send_comm_rank == 0)
      if (n_mol_corresp){ // initialize ind_corr array if some molecule correspondences exist 
        memory->create(ind_corr,n_mol_corresp,"fix_inflow_periodic:ind_corr");
	for (j = 0; j < n_mol_corresp; ++j)
	  ind_corr[j] = 1;
      }
	
    for (i = 0; i < n_mol_max; ++i) // assign new values for molecule types and inflow ids
      if (mol_tag_recv[i] > -1){    // shift also molecule ids in the correspondence lists
        ii = mol_inf[i]; 
        k = mol_tag[i];         
        l = i + mol_st_prev[ii][k];
        atom->mol_type[l] = k;
        mol_list[l] = ii;
	if (bc_send_comm_rank == 0)
          for (j = 0; j < n_mol_corresp; ++j)
	    if (mol_corresp_list[2*j] == i && ind_corr[j]){
              ind_corr[j] = 0;
              mol_corresp_list[2*j] += mol_st_prev[ii][k];
	      mol_corresp_list[2*j+1] += mol_st_prev[ii][k];
              break;
	    }
      }
    n_mol_max = atom->n_mol_max;
  }

  if (bc_send_comm_rank > -1){  // clean not needed correspondences and create a global correspondence list (e.g., used in angle_area_volume class)
    if (ind){  // extend local arrays if needed
      memory->grow(mol_tag,n_mol_max,"fix_inflow_periodic:mol_tag");
      memory->grow(mol_tag_recv,n_mol_max,"fix_inflow_periodic:mol_tag_recv");
    }
    
    for (i = 0; i < n_mol_max; ++i){ // initialize all values to -1
      mol_tag[i] = -1;
      mol_tag_recv[i] = -1;
    }

    for (j = 0; j < nlocal; ++j){  // loop over local particles to check if some molecules have already moved far enough to remove them from the correspondence list
      i = molecule[j] - 1; 
      if (i > -1){
  	ii = mol_list[i];
	if (ii == ninf){
          for (k = 0; k < 3; ++k)
            xx[k] = x[j][k] - x0[ii][k]; 
          rot_forward(xx[0],xx[1],xx[2],ii);
          rr = sqrt(xx[0]*xx[0] + xx[1]*xx[1]);
          if (rr < aa[ii][3] && xx[2] > aa[ii][4] && xx[2] < aa[ii][4] + lm) // check which molecules are at the end of an inflow cylinder
            mol_tag[i] = 1;
	} else{
          sprintf(str,"The inflow number is not consistent with the inflow proc group in FixInflowPeriodic::periodic_mol_reordering: ii = %d and ninf = %d!",ii,ninf);
          error->warning(FLERR,str); 
	}
      }
    }
    MPI_Reduce(mol_tag,mol_tag_recv,n_mol_max,MPI_INT,MPI_MAX,0,bc_send_comm);  // reduce molecule array for those at the end of an inflow cylinder

    if (bc_send_comm_rank == 0){  // only roots of inflows participate
      if (n_mol_corresp){  
	k = 0;
        while (k < n_mol_corresp){  // remove molecule correspondences which are not needed anymore
          i = mol_corresp_list[2*k] - 1;
  	  if (mol_tag_recv[i] < 0){
            mol_corresp_list[2*k] = mol_corresp_list[2*n_mol_corresp-2];
            mol_corresp_list[2*k+1] = mol_corresp_list[2*n_mol_corresp-1];
	    n_mol_corresp--;
	  } else
	    k++;
        }
      }

      if (bc_send_zero_comm_rank == 0){  // create arrays at the root of all inflow roots
        memory->create(rcounts1,bc_send_zero_comm_size,"fix_inflow_periodic:rcounts1");
        memory->create(displs1,bc_send_zero_comm_size,"fix_inflow_periodic:displs1");	
      }
      MPI_Gather(&n_mol_corresp,1,MPI_INT,rcounts1,1,MPI_INT,0,bc_send_zero_comm); // gather correspondence numbers
      if (bc_send_zero_comm_rank == 0){
        atom->n_mol_corresp_glob = 0;
	for (j = 0; j < bc_send_zero_comm_size; ++j){ // calculate counts and displacements for not-even data gathering
          displs1[j] = 2*atom->n_mol_corresp_glob;
	  atom->n_mol_corresp_glob += rcounts1[j];
	  rcounts1[j] *= 2;
	}
	if (atom->n_mol_corresp_glob > atom->n_mol_corresp_glob_max){  // extend global correspondence array if needed
          atom->n_mol_corresp_glob_max = atom->n_mol_corresp_glob + MOLINC;
	  memory->grow(atom->mol_corresp_glob,2*atom->n_mol_corresp_glob_max,"atom:mol_corresp_glob");
	}
      }
      MPI_Gatherv(mol_corresp_list,2*n_mol_corresp,MPI_LMP_TAGINT,atom->mol_corresp_glob,rcounts1,displs1,MPI_LMP_TAGINT,0,bc_send_zero_comm); // gather all molecule correspondences
    }
  }

  MPI_Bcast(&atom->n_mol_corresp_glob,1,MPI_LMP_TAGINT,glob_zero,world); // broadcast the number of correspondences globally
  if (atom->n_mol_corresp_glob > atom->n_mol_corresp_glob_max){  // extend global correspondence array if needed
    atom->n_mol_corresp_glob_max = atom->n_mol_corresp_glob + MOLINC;
    memory->grow(atom->mol_corresp_glob,2*atom->n_mol_corresp_glob_max,"atom:mol_corresp_glob");
  }
  MPI_Bcast(atom->mol_corresp_glob,2*atom->n_mol_corresp_glob,MPI_LMP_TAGINT,glob_zero,world);  // broadcast all correspondences globally 
       
  memory->destroy(rcounts1); // destroy all local arrays
  memory->destroy(displs1);
  memory->destroy(ind_corr); 
  memory->destroy(mol_tag);
  memory->destroy(mol_tag_recv);
  memory->destroy(mol_inf);

  /*sprintf(f_name,"after_%d_%d.dat",comm->me,update->ntimestep);
  out_stat=fopen(f_name,"w");
  fprintf(out_stat,"n_mol_max=%d, n_mol_types=%d, num_inflow=%d  \n", n_mol_max, n_mol_types, num_inflow);
  fprintf(out_stat,"mol types and lists  \n");
  for (i = 0; i < n_mol_max; i++)
    fprintf(out_stat,"%d %d %d \n",i,atom->mol_type[i],mol_list[i]);
  fprintf(out_stat,"numbers  \n");
  for (i = 0; i < num_inflow; ++i)
    for (j = 0; j < n_mol_types; ++j)
      fprintf(out_stat,"%d %d %d %d %d %d \n",i,j,mol_size[j],num_mols[i][j],mol_st[i][j],tag_st[i][j]);
  fprintf(out_stat,"mol data  \n");
  for (j = 0; j < nlocal; ++j){
    i = molecule[j] - 1;
    if (i > -1)
      fprintf(out_stat,"%d %d %d x=%f %f %f \n",j,molecule[j],tag[j],x[j][0],x[j][1],x[j][2]);
  }
  fprintf(out_stat,"n_mol_corresp=%d  \n",n_mol_corresp);
  for (j = 0; j < n_mol_corresp; ++j)
    fprintf(out_stat,"%d %d %d \n",j,mol_corresp_list[2*j],mol_corresp_list[2*j+1]);
  fprintf(out_stat,"atom->n_mol_corresp_glob=%d  \n",atom->n_mol_corresp_glob);
  for (j = 0; j < atom->n_mol_corresp_glob; ++j)
    fprintf(out_stat,"%d %d %d \n",j,atom->mol_corresp_glob[2*j],atom->mol_corresp_glob[2*j+1]); 
    fclose(out_stat);*/
}

/*------------------------------------------------------------------------
  correct molecule id and tags in the sending buffer
  originally it is a particle being sent
  while the received particle should belong to a different molecule
------------------------------------------------------------------------*/

int FixInflowPeriodic::correct_mol_tags(int m, double *buf)
{
  int i,k,j,ind,ind1;
  tagint ml, mlc, l;
  double sh;
  char str[200];
  
  k = m;
  ml = static_cast<tagint>(buf[m+1]);
  //ml = (tagint) ubuf(buf[m+1]).i;
  ind = 1;
  for (i = 0; i < n_mol_corresp; ++i)   // check if molecule correspondence already exists
    if (mol_corresp_list[2*i] == ml){
      ind = 0;
      mlc = mol_corresp_list[2*i+1];
      break;
    }

  i = atom->mol_type[ml-1];  
  if (ind){             // create new molecule correspondence
    if (n_mol_corresp == n_mol_corresp_max){   // extend mol_corresp_list if needed
      n_mol_corresp_max += MOLINC;
      memory->grow(mol_corresp_list,2*n_mol_corresp_max,"fix_inflow_periodic:mol_corresp_list");
    }
   
    if (mol_list[ml-1] != ninf){         // check for consistency of the inflow number
      sprintf(str,"Something is not consistent in FixInflowPeriodic::correct_mol_tags: ml = %g, mol_list[ml] = %d, and ninf = %d!",ml,mol_list[ml-1],ninf);
      error->one(FLERR,str); 
    }
    
    ind1 = 1;
    for (l = mol_st[ninf][i]; l < mol_st[ninf][i] + num_mols[ninf][i]; ++l)           // find new available molecule ID
      if (mol_list[l-1] == -1){
        mlc = l;
	mol_list[mlc-1] = ninf;
	atom->mol_type[mlc-1] = i;
	mol_corresp_list[2*n_mol_corresp] = ml;
	mol_corresp_list[2*n_mol_corresp+1] = mlc;
	n_mol_corresp++;
	ind1 = 0;
        break;
      }
    if (ind1){
      sprintf(str,"Failed to find a new molecule ID in FixInflowPeriodic::correct_mol_tags: ml = %g and ninf = %d!",ml,ninf);
      error->one(FLERR,str);       
    }
  }
  
  l = (mlc - ml)*atom->mol_size[i];
  sh  = static_cast<double> (l);
  //sh = ubuf(l).d;       // tag shift
  buf[m+1] = static_cast<double> (mlc);
  //buf[m+1] = ubuf(mlc).d;
  m += part_size;
  buf[m++] += sh;      // shift the particle tag

  ind = static_cast<int> (buf[m++]);
  //ind = (int) ubuf(buf[m++]).i;  
  ind1 = ind;
  for (i = 0; i < ind; ++i){   // shift the bond tags
    m++;
    buf[m++] += sh;
  }

  ind = static_cast<int> (buf[m++]);
  //ind = (int) ubuf(buf[m++]).i;
  ind1 += ind;
  for (i = 0; i < ind; ++i){  // shift the angle tags
    m++;
    for (j = 0; j < 3; ++j)
      buf[m++] += sh;
  }

  ind = static_cast<int> (buf[m++]);
  //ind = (int) ubuf(buf[m++]).i;
  ind1 += ind;
  for (i = 0; i < ind; ++i){  // shift the dihedral tags
    m++;
    for (j = 0; j < 4; ++j)
      buf[m++] += sh;
  }

  ind = static_cast<int> (buf[m++]);
  //ind = (int) ubuf(buf[m++]).i;
  for (i = 0; i < ind; ++i){  // shift the improper tags
    m++;
    for (j = 0; j < 4; ++j)
      buf[m++] += sh;
  }
  
  if (atom->individual) // skip individual characteristics
    m += ind1;

  m += 2;
  ind = static_cast<int> (buf[m++]); 
  //ind = (int) ubuf(buf[m++]).i;
  for (i = 0; i < ind; ++i)  // shift the special tags
    buf[m++] += sh;
  
  i = m-k;
  return i;
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
    if (norm[2] < 0.0){
      rot[id][1][1] = -1.0;
      rot[id][2][2] = -1.0;
    } 
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
  int i,j,kk,color,ind,offset,spm;
  int *tmp, *tmp1, *sp_list_tmp;
  double **x = atom->x;
  int *sp = atom->spin;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int step = update->ntimestep;
  char str1[50],str[200];

  tmp = tmp1 = sp_list_tmp = NULL;

  if (id == 0)
    sprintf(str1,"Initial consistency check");
  else if (id == 1)
    sprintf(str1,"Post_integrate consistency check");
  else
    sprintf(str1,"End_of_step consistency check");

  // consistency checks

  if (bc_send_comm_rank > -1){
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

    MPI_Gather(&ind,1,MPI_INT,send_counts,1,MPI_INT,0,bc_send_comm); // gather the numbers of tracked IDs 
    if (bc_send_comm_rank == 0){
      color = 0;
      for (i = 0; i < bc_send_comm_size; ++i)
        color += send_counts[i];
    }
    MPI_Bcast(&color,1,MPI_INT,0,bc_send_comm);   // broadcast the total number of tracked IDs

    if (bc_send_comm_rank == 0){
      MPI_Recv(&kk,1,MPI_INT,recv_partner,recv_partner,world,MPI_STATUS_IGNORE);
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

      if (bc_send_comm_rank == 0){
        memory->create(tmp,color,"fix_inflow_periodic:tmp");
        offset = 0;
        for (i = 0; i < bc_send_comm_size; ++i){
          displs[i] = offset;
          rcounts[i] = send_counts[i];
          offset += rcounts[i];
        }
      }
      MPI_Gatherv(tmp1,ind,MPI_INT,tmp,rcounts,displs,MPI_INT,0,bc_send_comm); // gather already existing tracked IDs 

      if (bc_send_comm_rank == 0){
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

        MPI_Recv(tmp,kk,MPI_INT,recv_partner,recv_partner,world,MPI_STATUS_IGNORE);

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
        if (bc_send_comm_rank == 0){
          memory->create(sp_list,sp_max,"fix_inflow_periodic:sp_list"); // create original sp_list even if there are no already existing tracked IDs 
          for (i = 0; i < sp_max; ++i)
            sp_list[i] = -1;
        } 
    }
  }

  if (bc_recv_comm_rank > -1){
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
      
    MPI_Gather(&ind,1,MPI_INT,send_counts,1,MPI_INT,0,bc_recv_comm); // gather the numbers of tracked IDs 
    if (bc_recv_comm_rank == 0){
      color = 0;
      for (i = 0; i < bc_recv_comm_size; ++i)
        color += send_counts[i];
    }
    MPI_Bcast(&color,1,MPI_INT,0,bc_recv_comm);   // broadcast the total number of tracked IDs

    if (bc_recv_comm_rank == 0)
      MPI_Send(&color,1,MPI_INT,send_partner,recv_partner,world);

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

      if (bc_recv_comm_rank == 0){
        memory->create(tmp,color,"fix_inflow_periodic:tmp");
        offset = 0;
        for (i = 0; i < bc_recv_comm_size; ++i){
          displs[i] = offset;
          rcounts[i] = send_counts[i];
          offset += rcounts[i];
        }
      }
      MPI_Gatherv(tmp1,ind,MPI_INT,tmp,rcounts,displs,MPI_INT,0,bc_recv_comm); // gather already existing tracked IDs  

      if (bc_recv_comm_rank == 0){
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
 
        MPI_Send(tmp,color,MPI_INT,send_partner,recv_partner,world);

        memory->destroy(tmp);
      }
      memory->destroy(tmp1);
    }
  }
}

/*------------------------------------------------------------------------*/

