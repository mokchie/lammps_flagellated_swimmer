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

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include "fix_force_bound_2d.h"
#include "atom.h"
#include "comm.h"
#include "group.h"
#include "error.h"
#include "memory.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "neighbor.h"
#include "domain.h"
#include "force.h"
#include "universe.h"

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 100
#define NUM 10000
#define NUM1 10
#define NUM2 100
#define FACE_INC 5
#define EPS 1e-6
#define EPSSS 0.01 
#define BIG 1.0e20
#define DR 0.01
#define MAX_GBITS 64

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixForceBound2D::FixForceBound2D(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  int i, j, l, ttyp, igroup, nms, nw_max; 
  double *rdata = NULL;
  double dummy,a1[2],rr,wd; 
  char grp[50],grp1[50],grp2[50];
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  char fname1[FILENAME_MAX];
  FILE *f_read, *f_read1;

  if (domain->dimension != 2) error->all(FLERR,"The fix force/bound/2d should be only used in 2D simulations!");

  groupbit_s = groupbit_p = 0;
  f_press = NULL;
  weight = NULL;
  dr_inv = 1.0/DR;
  if (narg != 4) error->all(FLERR,"Illegal fix force/bound/2d command");
  sprintf(fname,arg[3]);
  user_part = comm->user_part;
  send_bit = NULL;
  m_comm = NULL;
  me = comm->me;

  nms = NUM;
  cur_iter = 1;
  size_border = 3;
  size_forward = 4;
  size_reverse = 2; 
  read_restart_ind = 0;
  if (domain->xperiodic)
    prd[0] = domain->xprd;
  else
    prd[0] = 0;
  if (domain->yperiodic)
    prd[1] = domain->yprd;
  else
    prd[1] = 0;
  if (domain->zperiodic)
    prd[2] = domain->zprd;
  else
    prd[2] = 0;
  
  if (me == 0){
    l = 0;
    f_read = fopen(fname,"r");
    if (f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open input boundary file");
    memory->create(rdata,nms,"fix_force_bound_2d:rdata");
    fgets(buf,BUFSIZ,f_read); 
    sscanf(buf,"%d",&num_shapes_tot);
    rdata[l++] = static_cast<double> (num_shapes_tot); 
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %d %s",&r_cut_n,&r_cut_t,&mirror,&grp[0]);
    rdata[l++] = r_cut_n;
    rdata[l++] = r_cut_t;
    rdata[l++] = static_cast<double> (mirror);
    igroup = group->find(grp);
    if (igroup == -1) error->one(FLERR,"Group ID for group_solid does not exist");
    groupbit_solid = group->bitmask[igroup];
    rdata[l++] = static_cast<double> (groupbit_solid);
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d %lf %lf %lf %d %d %s %s %s",&ind_bounce,&d_cut,&comm_cut,&binsize,&max_count,&cell_update,&grp[0],&grp1[0],&grp2[0]);
    rdata[l++] = static_cast<double> (ind_bounce);   // 0 - bounce-back; 1 - bounce-forward
    rdata[l++] = d_cut;
    rdata[l++] = comm_cut; 
    rdata[l++] = binsize;
    rdata[l++] = static_cast<double> (max_count);
    rdata[l++] = static_cast<double> (cell_update);
    igroup = group->find(grp);
    if (igroup == -1) error->one(FLERR,"Group ID for group_inner does not exist");
    groupbit_inner = group->bitmask[igroup];
    rdata[l++] = static_cast<double> (groupbit_inner);
    igroup = group->find(grp1);
    if (igroup == -1) error->one(FLERR,"Group ID for group_comm does not exist");
    groupbit_comm = group->bitmask[igroup];
    rdata[l++] = static_cast<double> (groupbit_comm);
    igroup = group->find(grp2);
    if (igroup == -1) error->one(FLERR,"Group ID for group_no_move does not exist");
    groupbit_no_move = group->bitmask[igroup];
    rdata[l++] = static_cast<double> (groupbit_no_move);  

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&ind_shear); 
    rdata[l++] = static_cast<double> (ind_shear);
    if (ind_shear){
      sscanf(buf,"%d %lf %lf %lf %d %d %d %d %s",&ind_shear,&r_shear,&coeff,&power,&n_per,&iter,&mmax_iter,&s_apply,&grp[0]);
      rdata[l++] = r_shear;
      rdata[l++] = coeff;
      rdata[l++] = power;
      rdata[l++] = static_cast<double> (n_per);
      rdata[l++] = static_cast<double> (iter);
      rdata[l++] = static_cast<double> (mmax_iter);
      rdata[l++] = static_cast<double> (s_apply);
      igroup = group->find(grp);
      if (igroup == -1) error->one(FLERR,"Group ID for group_s does not exist");
      groupbit_s = group->bitmask[igroup];
      rdata[l++] = static_cast<double> (groupbit_s);
    }
    
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&ind_press);
    rdata[l++] = static_cast<double> (ind_press);
    if (ind_press){
      sscanf(buf,"%d %d %lf %d %s %s",&ind_press,&n_press,&r_press,&p_apply,&fname1[0],&grp[0]);
      rdata[l++] = static_cast<double> (n_press);
      rdata[l++] = r_press;
      rdata[l++] = static_cast<double> (p_apply);
      igroup = group->find(grp);
      if (igroup == -1) error->one(FLERR,"Group ID for group_p does not exist");
      groupbit_p = group->bitmask[igroup];
      rdata[l++] = static_cast<double> (groupbit_p);
      f_read1 = fopen(fname1,"r");
      if (f_read1 == (FILE*) NULL)
        error->one(FLERR,"Could not open input pressure file");
      for (j=0; j<n_press; j++){
        fgets(buf,BUFSIZ,f_read1);
        sscanf(buf,"%lf %lf",&dummy,&rdata[l]);
        l++;
      }
      fclose(f_read1);
    }
      
    for (i=0; i<num_shapes_tot; i++){
      fgets(buf,BUFSIZ,f_read);
      sscanf(buf,"%d",&ttyp);
      if (l + 10 > nms){
        nms += NUM;
        memory->grow(rdata,nms,"fix_force_bound_2d:rdata");
      }
      switch (ttyp){
        case 1 :             
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8]);
          l += 9;
          break;
        case 2 :
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7]);
          l += 8;
          break;  
      } 
    }
    fclose(f_read);
  } 
  
  MPI_Bcast(&l,1,MPI_INT,0,world);
  if (me)
    memory->create(rdata,l,"fix_force_bound_2d:rdata");
  MPI_Bcast(rdata,l,MPI_DOUBLE,0,world);

  l = 0;
  num_shapes_tot = static_cast<int> (rdata[l++]);
  r_cut_n = rdata[l++];
  r_cut_t = rdata[l++]; 
  mirror = static_cast<int> (rdata[l++]);
  groupbit_solid = static_cast<int> (rdata[l++]);
  ind_bounce = static_cast<int> (rdata[l++]);
  d_cut = rdata[l++];  
  d_cut_sq = d_cut*d_cut;
  comm_cut = rdata[l++]; 
  binsize = rdata[l++]; 
  max_count = static_cast<int> (rdata[l++]);
  cell_update = static_cast<int> (rdata[l++]);   
  groupbit_inner = static_cast<int> (rdata[l++]);
  groupbit_comm = static_cast<int> (rdata[l++]);
  groupbit_no_move = static_cast<int> (rdata[l++]);
  if (binsize <= 0.0)
    binsize = 0.5*r_cut_n;
  binsizeinv = 1.0/binsize;

  ind_shear = static_cast<int> (rdata[l++]);
  if (ind_shear){
    r_shear = rdata[l++];
    coeff = rdata[l++];
    power = rdata[l++];
    n_per = static_cast<int> (rdata[l++]);
    iter = static_cast<int> (rdata[l++]);
    mmax_iter = static_cast<int> (rdata[l++]);
    s_apply = static_cast<int> (rdata[l++]);
    groupbit_s = static_cast<int> (rdata[l++]);

    nw_max = static_cast<int> (r_shear*dr_inv) + 2;
    memory->create(weight,nw_max,"fix_force_bound_2d:weight");
    for (i = 0; i < nw_max; i++){
      rr = i*DR;
      if (rr > r_shear)
        wd = 0.0;
      else  
        wd = 1.0 - rr/r_shear;
      weight[i] = pow(wd,power);
    }
    restart_global = 1;
  }      
  ind_press = static_cast<int> (rdata[l++]);
  if (ind_press){
    n_press = static_cast<int> (rdata[l++]);
    r_press = rdata[l++];
    p_apply = static_cast<int> (rdata[l++]);
    groupbit_p = static_cast<int> (rdata[l++]);
    memory->create(f_press,n_press+1,"fix_force_bound_2d:f_press");
    for (j=0; j<n_press; j++)
      f_press[j] = rdata[l++];
    f_press[n_press] = 0.0;
  }
        
  numt = 0;
  numt_s = 0;
  num_shapes = 0;
  max_shapes = 0;
  max_faces = 0;
  le_max = le_npart = 0;
  nnmax = atom->nlocal;
  if (nnmax == 0) nnmax = 1;
  bin_max_shape = NUM1;
  memory->create(x0,num_shapes_tot,2,"fix_force_bound_2d:x0");
  memory->create(aa,num_shapes_tot,"fix_force_bound_2d:aa");
  memory->create(vel,num_shapes_tot,2,"fix_force_bound_2d:vel");
  memory->create(rot,num_shapes_tot,2,"fix_force_bound_2d:rot"); 
  memory->create(ndiv,num_shapes_tot,"fix_force_bound_2d:ndiv");
  memory->create(ptype,num_shapes_tot,"fix_force_bound_2d:ptype");
  memory->create(refl,num_shapes_tot,"fix_force_bound_2d:refl");
  memory->create(face_order,num_shapes_tot,"fix_force_bound_2d:face_order");
  memory->create(shapes_local,num_shapes_tot,"fix_force_bound_2d:shapes_local");
  memory->create(shapes_global_to_local,num_shapes_tot,"fix_force_bound_2d:shapes_global_to_local");
  bin_shapes = bin_vertex = bin_stensil = NULL;
  num_bin_shapes = num_bin_vertex = num_bin_stensil = NULL;
  ind_shapes = NULL; 
  tot_shapes = num_faces = NULL;
  le_sh = NULL; 
  memory->create(tot_shapes,nnmax,"fix_force_bound_2d:tot_shapes");
  memory->create(num_faces,nnmax,"fix_force_bound_2d:num_faces");       
        
  for (i=0; i<num_shapes_tot; i++){
    ptype[i] = static_cast<int> (rdata[l]);
    switch (ptype[i]){
      case 1 :
        for (j=0; j<2; j++){
          x0[i][j] = rdata[l+j+1];
          a1[j] = rdata[l+j+3] - rdata[l+j+1];
          vel[i][j] = rdata[l+j+5];
	}       
        ndiv[i] = static_cast<int> (rdata[l+7]);
        refl[i] = static_cast<int> (rdata[l+8]);
        l += 9; 
        break;
      case 2 :           
        for (j=0; j<2; j++){
          x0[i][j] = rdata[l+j+1];
          vel[i][j] = rdata[l+j+4];
	}       
        aa[i] = rdata[l+3];  
        ndiv[i] = static_cast<int> (rdata[l+6]);  
        refl[i] = static_cast<int> (rdata[l+7]);
        l += 8; 
        break;
    }
    setup_rot(i,a1);                    
    rot_forward(vel[i][0],vel[i][1],i);
    face_order[i] = numt;
    numt_s += 2*ndiv[i];
    numt += 3*2*n_per*ndiv[i];
    shapes_global_to_local[i] = -2;
    if (check_shapes(i)){
      shapes_local[num_shapes] = i;
      shapes_global_to_local[i] = num_shapes;
      num_shapes++;
    }
  }
  memory->destroy(rdata);
 
  numt_loc = 0; 
  if (ind_shear && num_shapes){
    velx = new double**[num_shapes];
    vely = new double**[num_shapes];
    num = new double**[num_shapes];
    fsx = new double**[num_shapes];
    fsy = new double**[num_shapes];
    for (i=0; i<num_shapes; i++){
      j = shapes_local[i];  
      memory->create(velx[i],2*n_per,ndiv[j],"fix_force_bound_2d:velx[i]");
      memory->create(vely[i],2*n_per,ndiv[j],"fix_force_bound_2d:vely[i]");
      memory->create(num[i],2*n_per,ndiv[j],"fix_force_bound_2d:num[i]");
      memory->create(fsx[i],2,ndiv[j],"fix_force_bound_2d:fsx[i]");
      memory->create(fsy[i],2,ndiv[j],"fix_force_bound_2d:fsy[i]");
      numt_loc += 4*ndiv[j];
    }
    numt_loc += num_shapes;
  } else{
    velx = vely = NULL;
    num = NULL;
    fsx = fsy = NULL;
  }

  part_degree = tags_part = NULL;
  bond_mol = bond_list = part_list = NULL;
  part_bond = bond_tags = NULL;
  x_part = v_part = v_delt = NULL;
  ind_faces = NULL;
  sendnum = recvnum = size_forward_recv = NULL;
  sendproc = recvproc = NULL;
  size_reverse_send = size_reverse_recv = NULL;
  maxsendlist = NULL;
  sendlist = NULL;
  buf_send = buf_recv = NULL; 
  slablo = slabhi = NULL; 
  pbc_flag = NULL;
  pbc = NULL;
  mol_mass = NULL; 
}

/* ---------------------------------------------------------------------- */

FixForceBound2D::~FixForceBound2D()
{
  int i;

  if (ind_press)
    memory->destroy(f_press);
  if (ind_shear) 
    memory->destroy(weight);
  if (ind_shear && num_shapes){
    for (i=0; i<num_shapes; i++){
      memory->destroy(velx[i]);
      memory->destroy(vely[i]);
      memory->destroy(num[i]);
      memory->destroy(fsx[i]);
      memory->destroy(fsy[i]);
    }
    delete[] velx;
    delete[] vely;
    delete[] num;
    delete[] fsx;
    delete[] fsy;
  }
    
  memory->destroy(x0);
  memory->destroy(aa);
  memory->destroy(vel);
  memory->destroy(rot);
  memory->destroy(ndiv); 
  memory->destroy(ptype);
  memory->destroy(refl);
  memory->destroy(face_order);
  if (max_shapes) memory->destroy(ind_shapes);
  memory->destroy(tot_shapes); 
  memory->destroy(shapes_local);
  memory->destroy(shapes_global_to_local);
  memory->destroy(le_sh); 
  memory->destroy(bin_shapes);
  memory->destroy(num_bin_shapes);
  memory->destroy(bin_vertex);
  memory->destroy(num_bin_vertex);  
  memory->destroy(bin_stensil);
  memory->destroy(num_bin_stensil);

  memory->destroy(part_degree);
  memory->destroy(tags_part);
  memory->destroy(bond_mol);
  memory->destroy(bond_list);
  memory->destroy(part_list);
  memory->destroy(part_bond);
  memory->destroy(bond_tags);
  memory->destroy(x_part);
  memory->destroy(v_part); 
  memory->destroy(v_delt); 

  if (user_part){
    memory->destroy(m_comm);
    memory->destroy(send_bit);
  }
  
  free_swap();
  memory->destroy(buf_send);
  memory->destroy(buf_recv);
 
  if (max_faces) memory->destroy(ind_faces);
  memory->destroy(num_faces);
  memory->destroy(mol_mass);
}

/* ---------------------------------------------------------------------- */

int FixForceBound2D::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::setup(int vflag)
{
  int l,ii;
  int iswap, dim, ineed, offset;
  tagint i,j,k,max_bond,n_bond;
  tagint *tag = atom->tag; 
  int *mask = atom->mask;
  int *type = atom->type;
  tagint *molecule = atom->molecule; 
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int *displs, *rcounts;
  tagint *bff, *bff1;  
  double mss[atom->n_mol_max+1];

  bff = bff1 = NULL;

  n_accum = 0;  
  // zero out initial arrays
  if (ind_shear && num_shapes)
    for (i = 0; i < num_shapes; i++){
      ii = shapes_local[i];    
      for (k = 0; k < ndiv[ii]; k++){
        for (j = 0; j < 2*n_per; j++){
          velx[i][j][k] = 0.0;
          vely[i][j][k] = 0.0;
          num[i][j][k] = 0.0;
        }
        if (!read_restart_ind)
          for (j = 0; j < 2; j++){
            fsx[i][j][k] = 0.0;
            fsy[i][j][k] = 0.0;
	  }
      }
    } 

  // assign atom masses 
  memory->create(mol_mass,atom->n_mol_max+1,"fix_force_bound_2d:mol_mass");
  for (i = 0; i < atom->n_mol_max+1; i++){
    mss[i] = 0.0;
    mol_mass[i] = 0.0; 
  }
  for (i = 0; i < atom->nlocal; i++)  
    if (atom->mass[type[i]] > mss[molecule[i]])
      mss[molecule[i]] = atom->mass[type[i]];
  MPI_Allreduce(&mss,mol_mass,atom->n_mol_max+1,MPI_DOUBLE,MPI_MAX,world); 

  // construct arrays and maps for cell structures
  memory->create(rcounts,comm->nprocs,"fix_force_bound_2d:rcounts");
  memory->create(displs,comm->nprocs,"fix_force_bound_2d:displs");
  n_bond = 0;
  for (i = 0; i < nbondlist; i++)
    if (mask[bondlist[i][0]] & groupbit_comm)
      n_bond++;
   
  l = static_cast<int>(n_bond); 
  if (comm->nprocs > 1){
    MPI_Gather(&l, 1, MPI_INT, rcounts, 1, MPI_INT, 0, world);
    //MPI_Gather(&n_bond, 1, MPI_LMP_TAGINT, rcounts, 1, MPI_LMP_TAGINT, 0, world);
    if (me == 0){
      nbond_tot = 0;  
      offset = 0;
      for (i = 0; i < comm->nprocs; i++) {
        nbond_tot += rcounts[i];  
        displs[i] = offset;
        rcounts[i] *= 3;
        offset += rcounts[i];
      }
    }
    MPI_Bcast(&nbond_tot,1,MPI_LMP_TAGINT,0,world); 
  } else{
    nbond_tot = n_bond; 
  }

  memory->create(bff,nbond_tot*3,"fix_force_bound_2d:bff");
  memory->create(bff1,nbond_tot*3,"fix_force_bound_2d:bff1");
  j = 0;

  if (comm->nprocs > 1){
    for (i = 0; i < nbondlist; i++) 
      if (mask[bondlist[i][0]] & groupbit_comm){
        bff1[3*j] = tag[bondlist[i][0]];
        bff1[3*j+1] = tag[bondlist[i][1]];
        bff1[3*j+2] = molecule[bondlist[i][0]];
        j++;
      }
    MPI_Gatherv(bff1, 3*n_bond, MPI_LMP_TAGINT, bff, rcounts, displs, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(bff,3*nbond_tot,MPI_LMP_TAGINT,0,world);
  } else{
    for (i = 0; i < nbondlist; i++) 
      if (mask[bondlist[i][0]] & groupbit_comm){
        bff[3*j] = tag[bondlist[i][0]];
        bff[3*j+1] = tag[bondlist[i][1]];
        bff[3*j+2] = molecule[bondlist[i][0]];
	j++;
      }
  }

  memory->create(bond_tags,nbond_tot,2,"fix_force_bound_2d:bond_tags");
  memory->create(bond_mol,nbond_tot,"fix_force_bound_2d:bond_mol");
  min_ind = 999999999;
  max_ind = -1;
  for (i = 0; i < nbond_tot; i++){
    bond_mol[i] = bff[3*i+2]; 
    for (j = 0; j < 2; j++){
      bond_tags[i][j] = bff[3*i+j];
      if (bond_tags[i][j] < min_ind) min_ind = bond_tags[i][j];
      if (bond_tags[i][j] > max_ind) max_ind = bond_tags[i][j];   
    } 
  }
  npart_tot = max_ind - min_ind + 1;
  memory->destroy(bff);
  memory->destroy(bff1);

  max_nbond = max_npart = 0;
  npart = nbond = 0;
  memory->create(part_degree,npart_tot,"fix_force_bound_2d:part_degree");
  memory->create(tags_part,npart_tot,"fix_force_bound_2d:tags_part");

  for (i = 0; i < npart_tot; i++){
    part_degree[i] = 0;
    tags_part[i] = -1;
  }

  for (i = 0; i < nbond_tot; i++){
    for (j = 0; j < 2; j++){
      bond_tags[i][j] -= min_ind;
      part_degree[bond_tags[i][j]]++;   
    }
  }
  max_bond = -1;
  for (i = 0; i < npart_tot; i++){
    if (part_degree[i] > max_bond) max_bond = part_degree[i];
    part_degree[i] = 0;
  }

  memory->create(part_bond,npart_tot,max_bond,"fix_force_bound_2d:part_bond"); 
  for (i = 0; i < nbond_tot; i++)
    for (j = 0; j < 2; j++){
      k = bond_tags[i][j];
      part_bond[k][part_degree[k]] = i;
      part_degree[k]++;   
    }  

  memory->destroy(rcounts);
  memory->destroy(displs);

  /*char f_name[FILENAME_MAX]; 
  FILE* out_stat;
  sprintf(f_name,"output_%d.plt",me);
  out_stat=fopen(f_name,"w");
  fprintf(out_stat,"min_ind=%d, max_ind=%d, npart_tot=%d  \n", min_ind, max_ind, npart_tot);
  fprintf(out_stat,"nbond_tot=%d, max_bond=%d  \n", nbond_tot, max_bond); 
  fprintf(out_stat,"bonds  \n"); 
  for (i = 0; i < nbond_tot; i++)
    fprintf(out_stat,"%d    %ld %ld mol=%d \n", i, bond_tags[i][0], bond_tags[i][1], bond_mol[i]);
  fprintf(out_stat,"indexes  \n"); 
  for (i = 0; i < npart_tot; i++){
    fprintf(out_stat,"%d   %ld    ", i, part_degree[i]);
    for (j = 0; j < part_degree[i]; j++)  
      fprintf(out_stat,"%ld ", part_bond[i][j]);
    fprintf(out_stat," \n"); 
  }
  fclose(out_stat);*/  

  // allocate arrays for communications
  maxsend = BUFMIN;
  memory->create(buf_send,maxsend+BUFEXTRA,"fix_force_bound_2d:buf_send");
  maxrecv = BUFMIN;
  memory->create(buf_recv,maxrecv,"fix_force_bound_2d:buf_recv");

  if (user_part){

    double cut_h[3];

    for (j = 0; j < 3; j++){
       cut_h[j] = comm_cut;
       nbp[j] = comm->nbp_orig[j];
       nbp_max[j] = comm->nbp_max[j];
       box_min[j] = domain->boxlo[j];     
    }
    cut_h[0] = cut_h[1] = cut_h[2] = comm_cut;
    comm->process_bin_data(n_neigh,dim,nbp,bin_ext,nbp_max,period_own,box_min,cut_h);

    for (j = 0; j < 3; j++)
      nbp_loc[j] = nbp_max[j] - comm->nbp_min[j] + 1;
  
    memory->create(m_comm,nbp_loc[0],nbp_loc[1],nbp_loc[2],"fix_force_bound_2d:m_comm");
  
    for (i = 0; i < nbp_loc[0]; i++)
      for (j = 0; j < nbp_loc[1]; j++)
        for (k = 0; k < nbp_loc[2]; k++)
          m_comm[i][j][k] = 0;

    // create bit arrays
  
    nswap = n_neigh;
    maxswap = nswap;
    allocate_swap(maxswap);
     
    if (n_neigh > MAX_GBITS)
      error->one(FLERR,"Not enough bits for the communication matrix in fix_force_bound_2d!");
  
    memory->create(send_bit,n_neigh,"fix_force_bound_2d:send_bit");
    
    comm->build_comm_matrix(n_neigh,m_comm,send_bit,nbp,nbp_max,nbp_loc,bin_ext,period_own,cut_h,sendproc,recvproc,pbc_flag,pbc,0);    

  } else {

    int left,right;

    for (i = 0; i < 3; i++)
      maxneed[i] = static_cast<int> (comm_cut * comm->procgrid[i] / domain->prd[i]) + 1;
    if (domain->dimension == 2) maxneed[2] = 0;
    for (i = 0; i < 3; i++)
      if (!domain->periodicity[i]) maxneed[i] = MIN(maxneed[i],comm->procgrid[i]-1);

    for (i = 0; i < 3; i++){
      if (!domain->periodicity[i]) {
        recvneed[i][0] = MIN(maxneed[i],comm->myloc[i]);
        recvneed[i][1] = MIN(maxneed[i],comm->procgrid[i]-comm->myloc[i]-1);
        left = comm->myloc[i] - 1;
        if (left < 0) left = comm->procgrid[i] - 1;
        sendneed[i][0] = MIN(maxneed[i],comm->procgrid[i]-left-1);
        right = comm->myloc[i] + 1;
        if (right == comm->procgrid[i]) right = 0;
        sendneed[i][1] = MIN(maxneed[i],right);
      } else { 
        recvneed[i][0] = recvneed[i][1] =
               sendneed[i][0] = sendneed[i][1] = maxneed[i];
      }
    }

    // allocate comm memory

    nswap = 2 * (maxneed[0]+maxneed[1]+maxneed[2]);
    maxswap = nswap;

    allocate_swap(maxswap);
 
    // setup communication data 
    iswap = 0;
    for (dim = 0; dim < 3; dim++) 
      for (ineed = 0; ineed < 2*maxneed[dim]; ineed++) {
        pbc_flag[iswap] = 0;
        pbc[iswap][0] = pbc[iswap][1] = pbc[iswap][2] =
          pbc[iswap][3] = pbc[iswap][4] = pbc[iswap][5] = 0;

        if (ineed % 2 == 0) {
          sendproc[iswap] = comm->procneigh[dim][0];
          recvproc[iswap] = comm->procneigh[dim][1];
          if (ineed < 2) slablo[iswap] = -BIG;
          else slablo[iswap] = 0.5*(domain->sublo[dim] + domain->subhi[dim]);
          slabhi[iswap] = domain->sublo[dim] + comm_cut;

          if (comm->myloc[dim] == 0) {
            pbc_flag[iswap] = 1;
            pbc[iswap][dim] = 1;
          }
        } else {
          sendproc[iswap] = comm->procneigh[dim][1];
          recvproc[iswap] = comm->procneigh[dim][0];
          slablo[iswap] = domain->subhi[dim] - comm_cut;        
          if (ineed < 2) slabhi[iswap] = BIG;
          else slabhi[iswap] = 0.5*(domain->sublo[dim] + domain->subhi[dim]);

          if (comm->myloc[dim] == comm->procgrid[dim]-1) {
            pbc_flag[iswap] = 1;
            pbc[iswap][dim] = -1;
          }
        }
        iswap++;
      }
  }
  
  // setup bins in the domain

  setup_bins();
  for (i = 0; i < nbin; ++i)
    for (j = 0; j < num_shapes; ++j){
       k = shapes_local[j];   
       if (check_bins(i,k)){
         bin_shapes[num_bin_shapes[i]][i] = k;
         num_bin_shapes[i]++;
         if (num_bin_shapes[i] == bin_max_shape){
           bin_max_shape += NUM1;
           memory->grow(bin_shapes,bin_max_shape,nbin,"fix_force_bound_2d:bin_shapes"); 
	 }
       }  
    }

  bin_max_vertex = bin_max_shape;  
  memory->create(bin_vertex,bin_max_vertex,nbin,"fix_force_bound_2d:bin_vertex");
  memory->create(num_bin_vertex,nbin,"fix_force_bound_2d:num_bin_vertex");  

  post_force(vflag);
}
/* ---------------------------------------------------------------------- */

void FixForceBound2D::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::post_integrate()
{
  int i,j,k,kk,ind,cond,i_x[NUM2],cc,bounce,cplane0,cplane1; 
  int ccount,i0,i1,m0,i_b[NUM2];
  tagint m;
  double dl[2],dl1[2],t_x[NUM2],dd[2],xp[2],xh[2],vh[2],vh1[2],vv[3],norm_h[2],x_x[2]; 
  double xd[2],xc[2],vc[2],cf[4],nnp[3],v_loc[2],xp1[2],shift[NUM2][2],shift1[NUM2][2];
  double tt,dot,d1,d2,dtt,u0,u1,uu,uun,mss,mprt; 
  double dtv = update->dt; 

  double **x = atom->x;
  double **v = atom->v;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *tag = atom->tag; 

  //double vv_mom[2], vv_ang, shf_h[2][2];

  local_faces(1);
  if (comm->nprocs > 1) communicate();

  for (i = 0; i < npart; i++)
    for (j = 0; j < 2; j++)
      v_delt[i][j] = 0.0; 

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit || mask[i] & groupbit_solid) {
      cond = 1;
      ccount = 0;
      dtt = dtv;
      cplane0 = -1;
      cplane1 = -1; 
      while (cond && ccount < max_count){
        ind = 0;
        if (ccount == 0)
          for (j = 0; j < 2; j++)
            v_loc[j] = v[i][j];    
        if (mask[i] & groupbit){
          for (j = 0; j < 2; j++){
            dl1[j] = v_loc[j]*dtt;
            xp1[j] = x[i][j] - dl1[j];
	  }
          for (k = 0; k < num_faces[i]; k++){
            tt = -1.0;
            m0 = ind_faces[i][k];
            if (m0 != cplane0){
              i0 = tags_part[bond_tags[m0][0]];
              i1 = tags_part[bond_tags[m0][1]];
              for (j = 0; j < 2; j++){
                x_x[j] = x_part[i0][j] - xp1[j]; 
                if (fabs(x_x[j]) > 0.5*prd[j]){
                  if (x_x[j] > 0.0)
                    shift[ind][j] = prd[j];
                  else
                    shift[ind][j] = -prd[j];
	        } else
                  shift[ind][j] = 0.0; 
                x_x[j] = x_part[i0][j] - x_part[i1][j];
                if (fabs(x_x[j]) > 0.5*prd[j]){
                  if (x_x[j] > 0.0)
                    shift1[ind][j] = prd[j];
                  else
                    shift1[ind][j] = -prd[j];
                } else
                  shift1[ind][j] = 0.0;           
	      }
              for (j = 0; j < 2; j++){
                xc[j] = x_part[i1][j] + shift1[ind][j] - x_part[i0][j]; 
                vc[j] = v_part[i1][j] - v_part[i0][j]; 
	      }
              nnp[0] = xc[1] - dtt*vc[1];
              nnp[1] = -xc[0] + dtt*vc[0]; 
	      nnp[2] = -nnp[0]*(x_part[i0][0] - v_part[i0][0]*dtt) - nnp[1]*(x_part[i0][1] - v_part[i0][1]*dtt);
              u0 = nnp[0]*(xp1[0]+shift[ind][0]) + nnp[1]*(xp1[1]+shift[ind][1]) + nnp[2];
              if ((u0 <= 0.0 && mask[i] & groupbit_inner) || (u0 >= 0.0 && !(mask[i] & groupbit_inner))){
                d1 = -xc[1]*x_part[i0][0] + xc[0]*x_part[i0][1]; 
                u1 = xc[1]*(x[i][0]+shift[ind][0]) - xc[0]*(x[i][1]+shift[ind][1]) + d1;
                if (u0*u1 <= 0.0){ 
                  for (j = 0; j < 2; j++){
                    vv[j] = dl1[j] - dtt*v_part[i0][j];
                    xd[j] = xp1[j]+shift[ind][j] - x_part[i0][j] + dtt*v_part[i0][j];
	          }
                  cf[1] = dtt*(vc[0]*vv[1] - vc[1]*vv[0]);
                  cf[2] = -nnp[0]*vv[0] - dtt*vc[1]*xd[0] - nnp[1]*vv[1] + dtt*vc[0]*xd[1];
                  cf[3] = -nnp[0]*xd[0] - nnp[1]*xd[1];
                  tt =  solve_quadratic(cf);
	        }
	      }
	      if (tt>=0.0 && tt<=1.0){  
                t_x[ind] = tt;
	        i_x[ind] = m0;
                i_b[ind] = 0; 
	        ind++;  
	      }
	    }
	  }         
	}
	if (mask[i] & groupbit_solid){
          for (m = 0; m < tot_shapes[i]; m++){
            kk = ind_shapes[i][m];
            if (kk != cplane1){
              for (j = 0; j < 2; j++){
                xh[j] = x[i][j] - x0[kk][j];
                vh[j] = v_loc[j];
	      }  
              rot_forward(xh[0],xh[1],kk);
              rot_forward(vh[0],vh[1],kk);  
              for (j = 0; j < 2; j++){
                dl[j] = vh[j]*dtt;
                xp[j] = xh[j] - dl[j]; 
	      }
           
              tt = -1.0;
              if (ptype[kk] == 1){
                if (xh[1]*xp[1] < 0.0)
                  if (refl[kk] == 3 || (xp[1]>0.0 && refl[kk] == 1) || (xp[1]<0.0 && refl[kk] == 2))
	            tt = xp[1]/(xp[1]-xh[1]);  
	      } else if (ptype[kk] == 2){  
                d1 = sqrt(xh[0]*xh[0] + xh[1]*xh[1]);
                d2 = sqrt(xp[0]*xp[0] + xp[1]*xp[1]);
                if ((d1-aa[kk])*(d2-aa[kk]) < 0.0)
                  if (refl[kk] == 3 || (d2>aa[kk] && refl[kk] == 1) || (d2<aa[kk] && refl[kk] == 2)){
                    vv[0] = dl[0]*dl[0] + dl[1]*dl[1];
                    vv[1] = (xp[0]*dl[0] + xp[1]*dl[1])/vv[0];
                    vv[2] = (d2*d2-aa[kk]*aa[kk])/vv[0];
                    dot =  vv[1]*vv[1] - vv[2];
                    if (dot > 0.0){
                      if (vv[2] > 0) 
	                tt = -vv[1] - sqrt(dot);
	              else
	                tt = -vv[1] + sqrt(dot);

                      //if (tt < EPS) tt = -1.0;  
		    }
	          }
	      }
          
	      if (tt>0.0 && tt<=1.0){ 
                  t_x[ind] = tt;
	          i_x[ind] = kk;
                  i_b[ind] = 1; 
	          ind++;
	      }
	    }
	  }
	}
     
        if (ind > 0){
          for (j = 1; j < ind; j++){          
            tt = 2.0; 
            for (k = 0; k < ind-j+1; k++)
              if (t_x[k]<tt) {
                tt = t_x[k];
                kk = k;      
	      }
            cc = i_x[kk];
            m = i_b[kk];
            t_x[kk] = t_x[ind-j];
            i_x[kk] = i_x[ind-j];
            i_b[kk] = i_b[ind-j];
            t_x[ind-j] = tt;
            i_x[ind-j] = cc;
            i_b[ind-j] = m; 
            for (k = 0; k < 2; k++){
              uu = shift[kk][k];
              shift[kk][k] = shift[ind-j][k];
              shift[ind-j][k] = uu;
              uu = shift1[kk][k];
              shift1[kk][k] = shift1[ind-j][k];
              shift1[ind-j][k] = uu;             
	    }  
          }
	  while (ind){
            bounce = 1;
            m = ind - 1;
            tt = t_x[m];
            if (i_b[m]){
	      kk = i_x[m];
              for (k = 0; k < 2; k++){
                xp[k] = x[i][k] - v_loc[k]*dtt - x0[kk][k];
                vh[k] = v_loc[k];
                vh1[k] = v[i][k];
	      }  
              rot_forward(xp[0],xp[1],kk);
              rot_forward(vh[0],vh[1],kk);
              rot_forward(vh1[0],vh1[1],kk);
              for (k = 0; k < 2; k++)
	        dd[k] = xp[k] + tt*vh[k]*dtt;
         
              switch (ptype[kk]){
                case 1 : 
                  u0 = dd[0]/aa[kk];
                  if (u0 < 0.0 || u0 > 1.0)
                    bounce = 0;
                  break;
                case 2 : 
                  norm_h[0] = dd[0]/aa[kk];
                  norm_h[1] = dd[1]/aa[kk]; 
                  break;
	      }
	    } else{
	      m0 = i_x[m];
              i0 = tags_part[bond_tags[m0][0]];
              i1 = tags_part[bond_tags[m0][1]];
              for (k = 0; k < 2; k++){
	        x_x[k] = xp1[k] + shift[m][k] + tt*dl1[k];
                xc[k] = x_part[i1][k] + shift1[m][k] - x_part[i0][k] + dtt*(1.0-tt)*(v_part[i0][k] - v_part[i1][k]);
	        dd[k] = x_x[k] - x_part[i0][k] + dtt*(1.0-tt)*v_part[i0][k];             	              
	      }
              u1 = xc[0]*xc[0] + xc[1]*xc[1]; 
              u0 = (dd[0]*xc[0] + dd[1]*xc[1])/u1;  
	      if (u0 <= 0.0 || u0 >= 1.0)
                bounce = 0;
	    }
            
	    if (bounce){
              ccount++;
              if (i_b[m]){
                cplane1 = kk; 
                if (mirror){
                  if (ptype[kk] == 1){
                    vh[1] = -vh[1];
                    vh1[1] = -vh1[1];
		  } else{
                    d1 = 2.0 * (vh[0]*norm_h[0] + vh[1]*norm_h[1]);
                    d2 = 2.0 * (vh1[0]*norm_h[0] + vh1[1]*norm_h[1]);
                    for (k = 0; k < 2; k++){ 
                      vh[k] += -d1*norm_h[k];     
                      vh1[k] += -d2*norm_h[k];
		    }
		  }
	        }
                else {
                  for (k = 0; k < 2; k++){
                    vh[k] = -vh[k] + 2.0*vel[kk][k];     
                    vh1[k] = -vh1[k] + 2.0*vel[kk][k];
		  }
	        }
                
                for (k = 0; k < 2; k++){
                  xp[k] = dd[k];      
                  xh[k] = xp[k] + (1.0-tt)*dtt*vh[k];      // verlet 
	        }
                rot_back(xh[0],xh[1],kk);
                rot_back(vh[0],vh[1],kk);
                rot_back(vh1[0],vh1[1],kk);
                for (k = 0; k < 2; k++){
                  x[i][k] = xh[k] + x0[kk][k];
                  v_loc[k] = vh[k];
                  v[i][k] = vh1[k];
	        }
	      } else{
                cplane0 = m0; 
                for (k = 0; k < 2; k++){
                  vv[k] = (1.0-u0)*v_part[i0][k] + u0*v_part[i1][k];  
                  xd[k] = 0.5*(x_part[i0][k] + x_part[i1][k] + shift1[m][k]);
                  vc[k] = 0.5*(v_part[i0][k] + v_part[i1][k]);
	        }
                mss = mol_mass[bond_mol[m0]];
                mprt = mass[type[i]];   

                /*for (k = 0; k < 2; k++){
                  shf_h[0][k] = x_part[i0][k] - xp1[k];
                  shf_h[1][k] = x_part[i0][k] - xp1[k];
                  if (fabs(shf_h[0][k]) > 0.5*prd[k]){
                    if (shf_h[0][k] > 0.0)
                      shf_h[0][k] = -prd[k];
                    else
                      shf_h[0][k] = prd[k];
                  } else shf_h[0][k] = 0.0;
                  if (fabs(shf_h[1][k]) > 0.5*prd[k]){
                    if (shf_h[1][k] > 0.0)
                      shf_h[1][k] = -prd[k];
                    else
                      shf_h[1][k] = prd[k];
                  } else shf_h[1][k] = 0.0;  
                  d1 = sqrt((x_part[i0][0]+shf_h[0][0]-x_part[i1][0]-shf_h[1][0])*(x_part[i0][0]+shf_h[0][0]-x_part[i1][0]-shf_h[1][0]) + (x_part[i0][1]+shf_h[0][1]-x_part[i1][1]-shf_h[1][1])*(x_part[i0][1]+shf_h[0][1]-x_part[i1][1]-shf_h[1][1])); 
                  //if (d1 > 3.0) printf("Large distance: %15.13lf \n",d1);
                  vv_mom[k] = 0.0;
                  vv_mom[k] += mprt*v[i][k];
                  vv_mom[k] += mss*(v_part[i0][k] + v_part[i1][k]); 
                }
                vv_ang = mprt*(xp1[0]*v[i][1] - xp1[1]*v[i][0]) + mss*((x_part[i0][0]+shf_h[0][0]-dtv*v_part[i0][0])*v_part[i0][1] - (x_part[i0][1]+shf_h[0][1]-dtv*v_part[i0][1])*v_part[i0][0] + (x_part[i1][0]+shf_h[1][0]-dtv*v_part[i1][0])*v_part[i1][1] - (x_part[i1][1]+shf_h[1][1]-dtv*v_part[i1][1])*v_part[i1][0]);*/

                if (ind_bounce == 1){
                  nnp[0] = xc[1];
                  nnp[1] = -xc[0]; 
                  uun = sqrt(u1);
                  uu = 2.0 * ((v_loc[0]-vv[0])*nnp[0] + (v_loc[1]-vv[1])*nnp[1])/uun;
                  for (k = 0; k < 2; k++)        
                    v_loc[k] -= uu*nnp[k]/uun;
	        }  
                else{
                  for (k = 0; k < 2; k++)
                    v_loc[k] = -v_loc[k] + 2.0*vv[k];         
	        }
                  
                for (k = 0; k < 2; k++){
                  //dd[k] = -xp1[k];
                  xp1[k] = x_x[k] - shift[m][k];      
                  x[i][k] = xp1[k] + (1.0-tt)*dtt*v_loc[k];      // verlet
                  //dd[k] += x[i][k];
                  //cf[k] = x[i][k] - xd[k] + shift[m][k]; 
                  vv[k] = -2.0*mss*(v[i][k]-vc[k])/(mprt+2.0*mss); 
	        } 

                /*u0 = sqrt(cf[0]*cf[0] + cf[1]*cf[1]);   
                u1 = sqrt(vv[0]*vv[0] + vv[1]*vv[1]);   
                if (u0*u1 > EPS){                 
                  uun = (-dd[0]*v[i][1] + dd[1]*v[i][0])/(u0*u1);
                  if (uun <= 1.0 && uun >= -1.0){
                    uu = asin(uun); 
                    
		    //  if (uun > 1.0){
		    //   uu = 0.5*M_PI;
		    // u1 *= uun;
		    //} else if (uun < -1.0){
		    // uu = -0.5*M_PI;
		    //  u1 *= -uun;
		    //} else {
		    // uu = asin(uun); 
		    // }                 
                  d1 = cos(uu);
                  d2 = sin(uu); 
                  vv[0] = u1*(d1*cf[0] - d2*cf[1])/u0;
                  vv[1] = u1*(d2*cf[0] + d1*cf[1])/u0;  
		  }
		} */                

                uu = -0.5*mprt/mss;  
                for (k = 0; k < 2; k++){
                  v[i][k] += vv[k];  
                  v_delt[i0][k] += uu*vv[k];
                  v_delt[i1][k] += uu*vv[k];
                }

                /*for (k = 0; k < 2; k++){
                  vv_mom[k] -= mprt*v[i][k];
                  vv_mom[k] -= mss*(v_part[i0][k] + uu*vv[k] + v_part[i1][k] + uu*vv[k]); 
                }
                vv_ang -= mprt*(x[i][0]*v[i][1] - x[i][1]*v[i][0]) + mss*((x_part[i0][0]+shf_h[0][0])*(v_part[i0][1]+uu*vv[1]) - (x_part[i0][1]+shf_h[0][1])*(v_part[i0][0]+uu*vv[0]) + (x_part[i1][0]+shf_h[1][0])*(v_part[i1][1]+uu*vv[1]) - (x_part[i1][1]+shf_h[1][1])*(v_part[i1][0]+uu*vv[0])); 
                if (fabs(vv_mom[0]) > EPS || fabs(vv_mom[1]) > EPS || fabs(vv_ang) > EPS)
		printf("Momentum change: %15.13lf %15.13lf; Ang. momentum change: %15.13lf \n",vv_mom[0],vv_mom[1],vv_ang);*/

                if (cell_update && mask[i] & groupbit_comm){
                  m = tag[i] - min_ind;
                  m0 = tags_part[m];
                  for (k = 0; k < 2; k++){
                    x_part[m0][k] = x[i][k];
                    v_part[m0][k] = v[i][k];         
	          } 
	        }
	      }
              dtt = (1.0-tt)*dtt;  
              ind = 0; 
	    }
	    else{
              ind--;
              if (ind == 0) 
                cond = 0;
	    }
	  }
	}
        else{
          cond = 0;
        }
      }
    } 

  if (comm->nprocs > 1) reverse_communicate();

  k = 0; 
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit_comm) {
      if (!(mask[i] & groupbit_no_move))
        for (j = 0; j < 2; j++){
          v[i][j] += v_delt[k][j];     
        }        
      k++;             
    }

  /*double vv_tot[2], vv_get[2], vv_ang_get;
  double vv_ang;
  for (j = 0; j < 2; j++){
    vv_tot[j] = 0.0;
    vv_get[j] = 0.0;
  }
  vv_ang = 0.0;
  vv_ang_get = 0.0;

  for (i = 0; i < nlocal; i++){
    domain->unmap(x[i],atom->image[i],cf);  
    vv_ang += mass[type[i]]*(cf[0]*v[i][1] - cf[1]*v[i][0]); 
    for (j = 0; j < 2; j++)
      vv_tot[j] += mass[type[i]]*v[i][j];
  }
  if (comm->nprocs > 1){
     MPI_Reduce(&vv_tot,&vv_get,2,MPI_DOUBLE,MPI_SUM,0,world);
     MPI_Reduce(&vv_ang,&vv_ang_get,1,MPI_DOUBLE,MPI_SUM,0,world);
  } else{
    for (j = 0; j < 2; j++)
      vv_get[j] = vv_tot[j];
    vv_ang_get = vv_ang; 
  }
  if (me == 0) printf("Momentum: %15.13lf %15.13lf; Ang. momentum: %15.13lf \n",vv_get[0],vv_get[1],vv_ang_get);*/
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::post_force(int vflag)
{
  int i,j,k,m,kk,ix,iy,kk1;
  double xh[2],vh[2],ff[2];
  double weight_h,rr,theta,theta1,hh;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  bigint step_t = update->ntimestep;

  if (atom->nlocal > nnmax){
    while (nnmax < atom->nlocal)               
      nnmax += BUFMIN;
    if (max_shapes) 
      memory->grow(ind_shapes,nnmax,max_shapes,"fix_force_bound_2d:ind_shapes");
    memory->grow(tot_shapes,nnmax,"fix_force_bound_2d:tot_shapes");
    if (max_faces)
      memory->grow(ind_faces,nnmax,max_faces,"fix_force_bound_2d:ind_faces");
    memory->grow(num_faces,nnmax,"fix_force_bound_2d:num_faces");
  }  
 
  if (neighbor->ago == 0){
    shape_decide();
    local_faces(0);
    if (comm->nprocs > 1) borders();
    bin_vertices();
    face_decide();
  }  

  n_accum++;
  if (ind_shear || ind_press)
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit_s || mask[i] & groupbit_p)
        for (m = 0; m < tot_shapes[i]; m++){
          kk = ind_shapes[i][m];
          kk1 = shapes_global_to_local[kk];
          for (j = 0; j < 2; j++){
            xh[j] = x[i][j] - x0[kk][j];
            vh[j] = v[i][j];
	  }
          rot_forward(xh[0],xh[1],kk);
          if (cur_iter)
            rot_forward(vh[0],vh[1],kk);  
          switch (ptype[kk]){
            case 1 :
              if (xh[0] >= 0.0 && xh[0] < aa[kk]){
                ix = static_cast<int> (xh[0]*ndiv[kk]/aa[kk]);
                if (ind_shear && (mask[i] & groupbit_s)){ 
                  if (xh[1]>0.0 && xh[1]<r_shear && (s_apply == 1 || s_apply == 0)){
                    if (cur_iter){
                      iy = static_cast<int> (xh[1]*n_per/r_shear);
                      velx[kk1][iy][ix] += vh[0];
                      vely[kk1][iy][ix] += vh[1];
                      num[kk1][iy][ix] += 1.0;
                    }
                    k = static_cast<int> (xh[1]*dr_inv);
                    weight_h = (weight[k+1]-weight[k])*(xh[1]*dr_inv - k) + weight[k];
                    ff[0] = fsx[kk1][0][ix]*weight_h;
                    ff[1] = 0.0;
                    rot_back(ff[0],ff[1],kk);
                    for (j = 0; j < 2; j++)
                      f[i][j] -= ff[j]; 
		  } 
                  if (xh[1]>-r_shear && xh[1]<0.0 && (s_apply == 2 || s_apply == 0)){
                    if (cur_iter){
                      iy = static_cast<int> (-xh[1]*n_per/r_shear);
                      velx[kk1][iy+n_per][ix] += vh[0];
                      vely[kk1][iy+n_per][ix] += vh[1];
                      num[kk1][iy+n_per][ix] += 1.0;
                    }
                    k = static_cast<int> (-xh[1]*dr_inv);
                    weight_h = (weight[k+1]-weight[k])*(-xh[1]*dr_inv - k) + weight[k];
                    ff[0] = fsx[kk1][1][ix]*weight_h;
                    ff[1] = 0.0;  
                    rot_back(ff[0],ff[1],kk);
                    for (j = 0; j < 2; j++)
                      f[i][j] -= ff[j]; 
		  }
	        }
                if (ind_press && (mask[i] & groupbit_p)){
                  if (xh[1]>0.0 && xh[1]<r_press && (p_apply == 1 || p_apply == 0)){
                    iy = static_cast<int> (xh[1]*n_press/r_press);
                    ff[0] = 0.0;
                    ff[1] = f_press[iy];
                    rot_back(ff[0],ff[1],kk);
                    for (j = 0; j < 2; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (xh[1]>-r_press && xh[1]<0.0 && (p_apply == 2 || p_apply == 0)){
                    iy = static_cast<int> (-xh[1]*n_press/r_press);
                    ff[0] = 0.0;
                    ff[1] = -f_press[iy];
                    rot_back(ff[0],ff[1],kk);
                    for (j = 0; j < 2; j++)
                      f[i][j] += ff[j]; 
		  }
	        }
	      }
              break;
            case 2 :
              rr = sqrt(xh[0]*xh[0] + xh[1]*xh[1]); 
              theta = acos(xh[0]/rr);
              theta1 = asin(xh[1]/rr);
              if (theta1 < 0.0)
                theta = 2.0*M_PI - theta;
              ix = static_cast<int> (0.5*theta*ndiv[kk]/M_PI);
              if (ix > ndiv[kk]-1)
                ix = ndiv[kk]-1;
              hh = rr - aa[kk];
              if (ind_shear && (mask[i] & groupbit_s)){
                if (hh>0.0 && hh<r_shear && (s_apply == 1 || s_apply == 0)){
                  if (cur_iter){
                    iy = static_cast<int> (hh*n_per/r_shear);
                    velx[kk1][iy][ix] += vh[0];
                    vely[kk1][iy][ix] += vh[1];
                    num[kk1][iy][ix] += 1.0;
                  }
                  k = static_cast<int> (hh*dr_inv);
                  weight_h = (weight[k+1]-weight[k])*(hh*dr_inv - k) + weight[k];
                  ff[0] = fsx[kk1][0][ix]*weight_h;
                  ff[1] = fsy[kk1][0][ix]*weight_h;   
                  rot_back(ff[0],ff[1],kk);
                  for (j = 0; j < 2; j++)
                    f[i][j] -= ff[j]; 
		} 
                if (hh>-r_shear && hh<0.0 && (s_apply == 2 || s_apply == 0)){
                  if (cur_iter){
                    iy = static_cast<int> (-hh*n_per/r_shear);
                    velx[kk1][iy+n_per][ix] += vh[0];
                    vely[kk1][iy+n_per][ix] += vh[1];
                    num[kk1][iy+n_per][ix] += 1.0;
                  } 
                  k = static_cast<int> (-hh*dr_inv);
                  weight_h = (weight[k+1]-weight[k])*(-hh*dr_inv - k) + weight[k];
                  ff[0] = fsx[kk1][1][ix]*weight_h;
                  ff[1] = fsy[kk1][1][ix]*weight_h;   
                  rot_back(ff[0],ff[1],kk);
                  for (j = 0; j < 2; j++)
                    f[i][j] -= ff[j]; 
		}
	      }
              if (ind_press && (mask[i] & groupbit_p)){
                if (hh>0.0 && hh<r_press && (p_apply == 1 || p_apply == 0)){
                  iy = static_cast<int> (hh*n_press/r_press);
                  ff[0] = f_press[iy]*xh[0]/rr;
                  ff[1] = f_press[iy]*xh[1]/rr;
                  rot_back(ff[0],ff[1],kk);
                  for (j = 0; j < 2; j++)
                    f[i][j] += ff[j]; 
		} 
                if (hh>-r_press && hh<0.0 && (p_apply == 2 || p_apply == 0)){
                  iy = static_cast<int> (-hh*n_press/r_press);
                  ff[0] = -f_press[iy]*xh[0]/rr;
                  ff[1] = -f_press[iy]*xh[1]/rr;
                  rot_back(ff[0],ff[1],kk);
                  for (j = 0; j < 2; j++)
                    f[i][j] += ff[j]; 
	        }
	      }   
              break;
	  }
	}

  if (ind_shear && num_shapes_tot)
    if (cur_iter && step_t%iter == 0 && n_accum > 0.6*iter) 
      recalc_force();
}

/* ---------------------------------------------------------------------- */

int FixForceBound2D::check_shapes(int sh)
{
  int i, ch_ind, nn[2], jj[2];
  double rad, rad_n, rad_t, zz;
  double dd[2], ddx[2];

  for (i = 0; i < 2; ++i){                  // setup bins
    ddx[i] = domain->subhi[i] - domain->sublo[i];
    nn[i] = static_cast<int> (ddx[i]/binsize + 3.0);
  }
  rad = 0.5*sqrt(2.0)*binsize;
  rad_n = rad + r_cut_n; 
  rad_t = rad + r_cut_t;

  ch_ind = 0;
  for (jj[0] = 0; jj[0] < nn[0]; ++jj[0]){             // check if certain bins overlap with a core
    for (jj[1] = 0; jj[1] < nn[1]; ++jj[1]){
      for (i = 0; i < 2; ++i)
        dd[i] = domain->sublo[i] + (jj[i]-0.5)*binsize - x0[sh][i];
      rot_forward(dd[0],dd[1],sh);

      switch (ptype[sh]){
        case 1 :
          if (dd[0] > -rad_t && dd[0] < aa[sh]+rad_t && dd[1] > -rad_n && dd[1] < rad_n)
            ch_ind = 1;
          break;
        case 2 :
          zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
          if (zz > aa[sh]-rad_n && zz < aa[sh]+rad_n)
            ch_ind = 1;
          break;
      }
      if (ch_ind) break;
    }
    if (ch_ind) break;
  }

  return ch_ind;
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::grow_shape_arrays()
{
  int k, l, m;
  int **ind_tmp = NULL;

  if (max_shapes == 0){
    max_shapes = FACE_INC;
    memory->create(ind_shapes,nnmax,max_shapes,"fix_force_bound_2d:ind_shapes");
  } else{
    m = max_shapes;
    memory->create(ind_tmp,nnmax,m,"fix_force_bound_2d:ind_tmp");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_tmp[k][l] = ind_shapes[k][l];
    max_shapes += FACE_INC;
    memory->destroy(ind_shapes);
    memory->create(ind_shapes,nnmax,max_shapes,"fix_force_bound_2d:ind_shapes");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_shapes[k][l] = ind_tmp[k][l];
    memory->destroy(ind_tmp);
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::setup_rot(int id, double a1[])
{
  int i;
  double rx;

  for (i = 0; i < 2; i++)
    rot[id][i] = 0.0;

  if (ptype[id] == 1){
    rx = sqrt(a1[0]*a1[0] + a1[1]*a1[1]);
    rot[id][0] = a1[0]/rx;
    rot[id][1] = a1[1]/rx;  
    aa[id] = rx;
  } else {
    rot[id][0] = 1.0; 
  }
} 

/* ---------------------------------------------------------------------- */

void FixForceBound2D::rot_forward(double &x, double &y, int id)
{
  double x1,y1;

  x1 = x; y1 = y;
  x = rot[id][0]*x1 + rot[id][1]*y1;
  y = -rot[id][1]*x1 + rot[id][0]*y1;
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::rot_back(double &x, double &y, int id)
{
  double x1,y1;

  x1 = x; y1 = y;
  x = rot[id][0]*x1 - rot[id][1]*y1;
  y = rot[id][1]*x1 + rot[id][0]*y1;
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::shape_decide()
{
  int i,sh,k,kk,bin;
  double dd[2], zz;

  double **x = atom->x;
  int *mask = atom->mask;

  for (i=0; i<atom->nlocal; i++){
    tot_shapes[i] = 0;
    if (mask[i] & groupbit_solid){
      bin = coord2bin(x[i]); 
      if (bin == -1){
        fprintf(stderr, "Negative bin in shape_decide: me=%d, step=" BIGINT_FORMAT ", particle - %f %f; box - %f %f %f %f \n",me,update->ntimestep,x[i][0],x[i][1],bboxlo[0],bboxlo[1],bboxhi[0],bboxhi[1]);
      }else{
        for (kk = 0; kk < num_bin_shapes[bin]; kk++){
          sh = bin_shapes[kk][bin];
          for (k = 0; k < 2; k++)
            dd[k] = x[i][k] - x0[sh][k];
          rot_forward(dd[0],dd[1],sh);
          switch (ptype[sh]){
            case 1 :
              if (dd[0] > -r_cut_t && dd[0] < aa[sh] + r_cut_t && dd[1] > -r_cut_n && dd[1] < r_cut_n){
                if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                ind_shapes[i][tot_shapes[i]] = sh;
                tot_shapes[i]++;
	      }
              break;
            case 2 :
              zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
              if (zz > aa[sh] - r_cut_n && zz < aa[sh] + r_cut_n){
                if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                ind_shapes[i][tot_shapes[i]] = sh;
                tot_shapes[i]++;
	      }
              break;
	  }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::setup_bins()
{
  // bbox = size of the bounding box
  // bbox lo/hi = bounding box of my subdomain extended by r_cut_n

  int i, ii, jj, nn;
  int ix, iy, bin, h_bin;
  double bbox[2];
  double r_extent = r_cut_n;

  for (i = 0; i < 2; i++){
    bboxlo[i] = domain->sublo[i] - r_extent;
    bboxhi[i] = domain->subhi[i] + r_extent;
    bbox[i] = bboxhi[i] - bboxlo[i];
  }

  nbinx = static_cast<int> (bbox[0]*binsizeinv + 1.0);
  nbiny = static_cast<int> (bbox[1]*binsizeinv + 1.0);
  nbin = nbinx*nbiny;
  
  memory->create(bin_shapes,bin_max_shape, nbin, "fix_force_bound_2d:bin_shapes");
  memory->create(num_bin_shapes,nbin,"fix_force_bound_2d:num_bin_shapes");
  for (i = 0; i < nbin; ++i)
    num_bin_shapes[i] = 0;
  
  nn = static_cast<int> (d_cut*binsizeinv + 1.0);
  ii = 2*nn + 1;
  h_bin = ii*ii;
  
  memory->create(bin_stensil,h_bin, nbin, "fix_force_bound_2d:bin_stensil");
  memory->create(num_bin_stensil,nbin,"fix_force_bound_2d:num_bin_stensil");
  for (bin = 0; bin < nbin; ++bin){
    num_bin_stensil[bin] = 0;
    iy = static_cast<int> (bin/nbinx);
    ix = bin - iy*nbinx;
    for (ii = ix-nn; ii <= ix+nn; ++ii)
      for (jj = iy-nn; jj <= iy+nn; ++jj){
        h_bin = jj*nbinx + ii;
        if (bin_distance(ii-ix,jj-iy) < d_cut_sq && h_bin > -1 && h_bin < nbin){
          bin_stensil[num_bin_stensil[bin]][bin] = h_bin;
          num_bin_stensil[bin]++;          
        }       
      } 
  }
}

/* ---------------------------------------------------------------------- */

double FixForceBound2D::bin_distance(int i, int j)
{
  double delx,dely;

  if (i > 0) delx = (i-1)*binsize;
  else if (i == 0) delx = 0.0;
  else delx = (i+1)*binsize;

  if (j > 0) dely = (j-1)*binsize;
  else if (j == 0) dely = 0.0;
  else dely = (j+1)*binsize;

  return (delx*delx + dely*dely);
}

/* ---------------------------------------------------------------------- */

int FixForceBound2D::check_bins(int bin, int sh)
{
  int ix, iy, ch_ind;
  double rad, rad_n, rad_t, zz;
  double dd[2];

  ch_ind = 0;
  rad = 0.5*sqrt(2.0)*binsize;
  rad_n = rad + r_cut_n;
  rad_t = rad + r_cut_t;
  iy = static_cast<int> (bin/nbinx);
  ix = bin - iy*nbinx;
  dd[0] = bboxlo[0] + (ix + 0.5)*binsize - x0[sh][0];
  dd[1] = bboxlo[1] + (iy + 0.5)*binsize - x0[sh][1];

  rot_forward(dd[0],dd[1],sh);
  switch (ptype[sh]){
    case 1 :
      if (dd[0] > -rad_t && dd[0] < aa[sh] + rad_t && dd[1] > -rad_n && dd[1] < rad_n)
        ch_ind = 1;
      break;
    case 2 :
      zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
      if (zz > aa[sh]-rad_n && zz < aa[sh]+rad_n)
        ch_ind = 1;
      break;
  }

  return ch_ind;
}     

/* ---------------------------------------------------------------------- */

void FixForceBound2D::bin_vertices()
{
  int i, k, m;
  tagint j; 
 
  for (i = 0; i < nbin; ++i)
    num_bin_vertex[i] = 0;
    
  for (i = 0; i < npart; ++i){
    j = part_list[i];   
    k = tags_part[j];
    m = coord2bin(x_part[k]);
    if (m != -1){
      bin_vertex[num_bin_vertex[m]][m] = i;
      num_bin_vertex[m]++; 
      if (num_bin_vertex[m] == bin_max_vertex){
        bin_max_vertex += NUM1;
        memory->grow(bin_vertex,bin_max_vertex,nbin,"fix_force_bound_2d:bin_vertex"); 
      }     
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixForceBound2D::coord2bin(double *xx)
{
  int ix,iy;

  ix = static_cast<int> ((xx[0]-bboxlo[0])*binsizeinv);
  if (ix > nbinx-1 || ix < 0) return -1;
  iy = static_cast<int> ((xx[1]-bboxlo[1])*binsizeinv);
  if (iy > nbiny-1 || iy < 0) return -1;
  
  return (iy*nbinx + ix);
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::recalc_force()
{
  int i,j,l,m,nn,st,ii;
  double vv,theta,nrm[2];
  double *tmp,*ntmp;

  tmp = ntmp = NULL;

  memory->create(tmp,numt,"fix_force_bound_2d:tmp");
  memory->create(ntmp,numt,"fix_force_bound_2d:ntmp");

  n_accum = 0;
  for (i = 0; i < numt; i++){
    tmp[i] = 0.0;
    ntmp[i] = 0.0;
  }

  for (i = 0; i < num_shapes; i++){
    ii = shapes_local[i];   
    st = face_order[ii]; 
    l = 0; 
    for (j=0; j<ndiv[ii]; j++)
      for (m=0; m<2*n_per; m++){
        nn = st + l;
        tmp[nn] = num[i][m][j];
        tmp[nn+1] = velx[i][m][j];
        tmp[nn+2] = vely[i][m][j];
        num[i][m][j] = 0.0;
        velx[i][m][j] = 0.0;
        vely[i][m][j] = 0.0;
        l += 3;
      }
  }

  MPI_Allreduce(tmp,ntmp,numt,MPI_DOUBLE,MPI_SUM,world);
  
  for (i = 0; i < num_shapes; i++){
    ii = shapes_local[i];    
    st = face_order[ii]; 
    l = 0; 
    for (j=0; j<ndiv[ii]; j++)
      for (m=0; m<2*n_per; m++){
        nn = st + l;
        if (m == 0 && ntmp[nn] > 0.0 && ntmp[nn+3] > 0.0){
          vv=0.5*(3.0*ntmp[nn+1]/ntmp[nn]-ntmp[nn+4]/ntmp[nn+3]);
          fsx[i][0][j] += coeff*(vv-vel[ii][0]);
          vv=0.5*(3.0*ntmp[nn+2]/ntmp[nn]-ntmp[nn+5]/ntmp[nn+3]);
          fsy[i][0][j] += coeff*(vv-vel[ii][1]); 
	}
        if (m == n_per && ntmp[nn] > 0.0 && ntmp[nn+3] > 0.0){
          vv=0.5*(3.0*ntmp[nn+1]/ntmp[nn]-ntmp[nn+4]/ntmp[nn+3]);
          fsx[i][1][j] += coeff*(vv-vel[ii][0]);
          vv=0.5*(3.0*ntmp[nn+2]/ntmp[nn]-ntmp[nn+5]/ntmp[nn+3]);
          fsy[i][1][j] += coeff*(vv-vel[ii][1]); 
	}
	l += 3;
      }
  }

  // projection of the adaptive shear force onto the tangential direction for cylinder and sphere shapes
  for (i = 0; i < num_shapes; i++){
    ii = shapes_local[i];
    if (ptype[ii] == 2){
      for (j=0; j<ndiv[ii]; j++){
        theta = 2*M_PI*(j+0.5)/ndiv[ii];
        nrm[0] = cos(theta);
        nrm[1] = sin(theta);
        for (m=0; m<2; m++){
          vv = nrm[0]*fsx[i][m][j] + nrm[1]*fsy[i][m][j];
          fsx[i][m][j] -= vv*nrm[0];
          fsy[i][m][j] -= vv*nrm[1];
        }
      }
    }
  }

  cur_iter++;
  if (cur_iter > mmax_iter) cur_iter = 0;

  memory->destroy(tmp);
  memory->destroy(ntmp);

  /*char fname[FILENAME_MAX];
  FILE *f_write;
  sprintf(fname,"write_bla_%d.dat",me);
  f_write = fopen(fname,"w");
  fprintf(f_write,"%d \n",cur_iter);
  for (i=0; i<num_shapes; i++){
    m = shapes_local[i];
    fprintf(f_write,"%d \n",m);
    for (j=0; j<ndiv[m]; j++){
      fprintf(f_write,"%lf %lf \n",fsx[i][0][j], fsy[i][0][j]);
      fprintf(f_write,"%lf %lf \n",fsx[i][1][j], fsy[i][1][j]);
    }
  }
  fclose(f_write);*/
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::borders()
{
  int i,j,m,iswap,dim,ineed,nsend,nrecv,nfirst,nlast,sendflag,twoneed,smax,rmax;
  int nn[3],ind;
  double lo,hi,xx_h[3];
  double *buf;
  MPI_Request request;

  iswap = 0;
  smax = rmax = 0;
  xx_h[2] = 0.0; 
  
  if (user_part){

    for (dim = 0; dim < n_neigh; dim++) {

      // find atoms within send bins 
      // check only local atoms 
      // store sent atom indices in list for use in future timesteps

      m = nsend = 0;

      for (i = 0; i < npart; i++){
        for (j = 0; j < 2; j++)
          xx_h[j] = x_part[i][j]; 
        ind = comm->coords_to_bin(&xx_h[j],nn,nbp_loc,box_min);
        if (ind && (m_comm[nn[0]][nn[1]][nn[2]] & send_bit[dim])) {
          m += pack_border_h(i,&buf_send[m],pbc_flag[dim],pbc[dim]);
          if (m > maxsend) grow_send(m);
          sendlist[dim][nsend++] = i;  
        }
      }

      // swap atoms with other proc
      // no MPI calls except SendRecv if nsend/nrecv = 0
      // put incoming ghosts at end of my atom arrays

      MPI_Sendrecv(&nsend,1,MPI_INT,sendproc[dim],me,
                   &nrecv,1,MPI_INT,recvproc[dim],recvproc[dim],world,MPI_STATUS_IGNORE);
      if (nrecv*size_border > maxrecv) grow_recv(nrecv*size_border);
      if (nrecv) MPI_Irecv(buf_recv,nrecv*size_border,MPI_DOUBLE,
                           recvproc[dim],recvproc[dim],world,&request);
      if (nsend) MPI_Send(buf_send,nsend*size_border,MPI_DOUBLE,sendproc[dim],me,world);
      if (nrecv) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;

      // unpack buffer

      m = 0;
      for (i = 0; i < nrecv; i++) m += unpack_border_h(&buf[m]);

      // set all pointers & counters

      smax = MAX(smax,nsend);
      rmax = MAX(rmax,nrecv);
      sendnum[dim] = nsend;
      recvnum[dim] = nrecv;
      size_forward_recv[dim] = nrecv*size_forward;
      size_reverse_send[dim] = nrecv*size_reverse;
      size_reverse_recv[dim] = nsend*size_reverse;
      firstrecv[dim] = npart - nrecv;
    }

  } else {

    for (dim = 0; dim < 3; dim++) {
      nlast = 0;
      twoneed = 2*maxneed[dim];
      for (ineed = 0; ineed < twoneed; ineed++) {

        // find all gost faces within slab boundaries lo/hi
        // store face indices in list for use in future timesteps
        lo = slablo[iswap];
        hi = slabhi[iswap];
        m = nsend = 0;
        if (ineed%2 == 0){
          nfirst = nlast;
          nlast = npart;
        }

        // lees-edwards fix
        if (comm->le && dim == 1 && ineed == 0){
          le_npart = npart;
          lees_edwards(1);
        }

        // sendflag = 0 if I do not send on this swap
        // sendneed test indicates receiver no longer requires data
        // e.g. due to non-PBC or non-uniform sub-domains
        if (ineed/2 >= sendneed[dim][ineed % 2]) sendflag = 0;
        else sendflag = 1;

        if (sendflag) {
          for (i = nfirst; i < nlast; i++) {
	    if (x_part[i][dim] >= lo && x_part[i][dim] < hi) {
	      if (m > maxsend) grow_send(m);
  	      m += pack_border_h(i,&buf_send[m],pbc_flag[iswap],pbc[iswap]);
	      if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
  	      sendlist[iswap][nsend++] = i;
    	    }
          }
        }

        // swap faces with other proc
        // put incoming ghosts at end of my faces arrays
        // if swapping with self, simply do nothing

        if (sendproc[iswap] != me) {
	  MPI_Sendrecv(&nsend,1,MPI_INT,sendproc[iswap],0,
	               &nrecv,1,MPI_INT,recvproc[iswap],0,world,MPI_STATUS_IGNORE);
	  if (nrecv*size_border > maxrecv) 
	    grow_recv(nrecv*size_border);
	  if (nrecv) MPI_Irecv(buf_recv,nrecv*size_border,MPI_DOUBLE,recvproc[iswap],0,world,&request);
	  if (nsend) MPI_Send(buf_send,nsend*size_border,MPI_DOUBLE,sendproc[iswap],0,world);
	  if (nrecv) MPI_Wait(&request,MPI_STATUS_IGNORE);
	  buf = buf_recv; 
        } else {
          nsend = 0;
	  nrecv = 0;
        }
      
        // unpack buffer

        m = 0;
        for (i = 0; i < nrecv; i++) m += unpack_border_h(&buf[m]);
     
        // set all pointers & counters

        smax = MAX(smax,nsend);
        rmax = MAX(rmax,nrecv);      
        sendnum[iswap] = nsend;
        recvnum[iswap] = nrecv;
        size_forward_recv[iswap] = nrecv * size_forward;
        size_reverse_send[iswap] = nrecv * size_reverse;
        size_reverse_recv[iswap] = nsend * size_reverse;
        firstrecv[iswap] = npart - nrecv;

        iswap++;
      }
    }
  }
  
  // insure buffers are large enough for forward and reverse comm
  int max = MAX(size_forward*smax,size_reverse*rmax);
  if (max > maxsend) grow_send(max);
  max = MAX(size_forward*rmax,size_reverse*smax);
  if (max > maxrecv) grow_recv(max);
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::communicate()
{
  int iswap, n, dim;
  MPI_Request request;

  if (user_part){

    for (iswap = 0; iswap < n_neigh; iswap++) {
      if (size_forward_recv[iswap])
        MPI_Irecv(buf_recv,size_forward_recv[iswap],MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world,&request);
      n = pack_comm_h(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
      if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],me,world);
      if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      unpack_comm_h(recvnum[iswap],firstrecv[iswap],buf_recv);
    }

  } else {

    for (iswap = 0; iswap < nswap; iswap++) {

      if (comm->le && iswap == 2*maxneed[0]) lees_edwards(0);  // lees-edwards fix 

      if (sendproc[iswap] != me) {
        if (size_forward_recv[iswap])
          MPI_Irecv(buf_recv,size_forward_recv[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
        n = pack_comm_h(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
        unpack_comm_h(recvnum[iswap],firstrecv[iswap],buf_recv);
      } 
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::reverse_communicate()
{
  int iswap,n, dim;
  MPI_Request request;

  if (user_part){

    for (iswap = n_neigh-1; iswap >= 0; iswap--) {
      if (size_reverse_recv[iswap])
        MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,sendproc[iswap],me,world,&request);
      n = pack_reverse_h(recvnum[iswap],firstrecv[iswap],buf_send);
      if (n) MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world);
      if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      unpack_reverse_h(sendnum[iswap],sendlist[iswap],buf_recv);
    }

  } else {

    for (iswap = nswap-1; iswap >= 0; iswap--) {

      if (sendproc[iswap] != me) {
        if (size_reverse_recv[iswap]) 
          MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
        n = pack_reverse_h(recvnum[iswap],firstrecv[iswap],buf_send);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
        if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);    
        unpack_reverse_h(sendnum[iswap],sendlist[iswap],buf_recv);
      } 
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::local_faces(int n)
{
  int i,j,k;
  double **x = atom->x;
  double **v = atom->v;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (n){
    k = 0; 
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit_comm) {
        for (j = 0; j < 2; j++){
          x_part[k][j] = x[i][j];
          v_part[k][j] = v[i][j];        
	}
        k++;             
      }
  } else{
    for (i = 0; i < npart; i++)
      tags_part[part_list[i]] = -1;
 
    npart = 0;
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit_comm) {
        if (npart == max_npart) grow_part_arrays(); 
        part_list[npart] = tag[i] - min_ind;
        tags_part[part_list[npart]] = npart;
        for (j = 0; j < 2; j++)
          x_part[npart][j] = x[i][j];
        npart++;    
      }
    npart_loc = npart;  
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::grow_send(int n)
{
  maxsend = static_cast<int> (BUFFACTOR * n);
  memory->grow(buf_send,maxsend+BUFEXTRA,"fix_force_bound_2d:buf_send");
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::grow_list(int iswap, int n)
{
  maxsendlist[iswap] = static_cast<int> (BUFFACTOR * n);
  memory->grow(sendlist[iswap],maxsendlist[iswap],"fix_force_bound_2d:sendlist[iswap]");
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::grow_recv(int n)
{
  maxrecv = static_cast<int> (BUFFACTOR * n);
  memory->destroy(buf_recv);
  memory->create(buf_recv,maxrecv,"fix_force_bound_2d:buf_recv");
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::grow_part_arrays()
{
  max_npart += BUFEXTRA;
  memory->grow(part_list,max_npart,"fix_force_bound_2d:part_list");
  memory->grow(x_part,max_npart,2,"fix_force_bound_2d:x_part");
  memory->grow(v_part,max_npart,2,"fix_force_bound_2d:v_part");
  memory->grow(v_delt,max_npart,2,"fix_force_bound_2d:v_delt");
} 

/* ---------------------------------------------------------------------- */

int FixForceBound2D::pack_border_h(int i, double *buf, int pbc_flag, int *pbc)
{
  int m = 0;

  buf[m++] = static_cast<double> (part_list[i]);
  if (pbc_flag == 0) {
    buf[m++] = x_part[i][0];
    buf[m++] = x_part[i][1];
  } else {
    buf[m++] = x_part[i][0] + pbc[0]*prd[0];
    buf[m++] = x_part[i][1] + pbc[1]*prd[1];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

int FixForceBound2D::unpack_border_h(double *buf)
{
  int m = 0;

  if (npart == max_npart) grow_part_arrays();
  part_list[npart] = static_cast<tagint> (buf[m++]);
  tags_part[part_list[npart]] = npart;
  x_part[npart][0] = buf[m++];
  x_part[npart][1] = buf[m++];
  npart++;
  
  return m;
}

/* ---------------------------------------------------------------------- */

int FixForceBound2D::pack_comm_h(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x_part[j][0];
      buf[m++] = x_part[j][1];
      buf[m++] = v_part[j][0];
      buf[m++] = v_part[j][1];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x_part[j][0] + pbc[0]*prd[0];
      buf[m++] = x_part[j][1] + pbc[1]*prd[1];     
      buf[m++] = v_part[j][0];
      buf[m++] = v_part[j][1];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::unpack_comm_h(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x_part[i][0] = buf[m++];
    x_part[i][1] = buf[m++];   
    v_part[i][0] = buf[m++];
    v_part[i][1] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int FixForceBound2D::pack_reverse_h(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = v_delt[i][0];
    buf[m++] = v_delt[i][1];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::unpack_reverse_h(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    v_delt[j][0] += buf[m++];
    v_delt[j][1] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::face_decide()
{
  int i,l,m,n,jj,kk,bin,h_bin,ii,ind_h;
  tagint j,k;
  double dd[3],rr;

  double **x = atom->x;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
    
  for (i=0; i<atom->nlocal; i++){
    num_faces[i] = 0;  
    if (mask[i] & groupbit){ 
      bin = coord2bin(x[i]); 
      if (bin == -1){
        fprintf(stderr, "Negative bin in face_decide: me=%d, step=" BIGINT_FORMAT ", particle - %f %f; box - %f %f %f %f \n",me,update->ntimestep,x[i][0],x[i][1],bboxlo[0],bboxlo[1],bboxhi[0],bboxhi[1]);
      }else{         
        for (kk = 0; kk < num_bin_stensil[bin]; ++kk){
          h_bin = bin_stensil[kk][bin];
          for (ii = 0; ii < num_bin_vertex[h_bin]; ++ii){
            jj = bin_vertex[ii][h_bin];  
            j = part_list[jj];   
            if (molecule[i] != bond_mol[part_bond[j][0]]){
              l = tags_part[j];
              dd[2] = 0.0;
              for (k = 0; k < 2; k++)
                dd[k] = x[i][k] - x_part[l][k];
              domain->minimum_image(dd[0],dd[1],dd[2]);
              rr = dd[0]*dd[0] + dd[1]*dd[1];
	      if (rr < d_cut_sq)
                for (n = 0; n < part_degree[j]; n++){
                  k = part_bond[j][n];
                  ind_h = 1;
                  for (m = 0; m < num_faces[i]; m++)
                    if (ind_faces[i][m] == k){
                      ind_h = 0;
                      break;  
		    }
                  if (ind_h){
                    if (tags_part[bond_tags[k][0]] > -1 &&  tags_part[bond_tags[k][1]] > -1) {
                      if (num_faces[i] == max_faces) grow_face_arrays();
                      ind_faces[i][num_faces[i]] = k;
                      num_faces[i]++;
                    } else{
                       printf("It might make sense to increase comm_cut, because some bonds are omitted!!! \n");
		    }
		  }
                }
            }
          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound2D::grow_face_arrays()
{
  int k, l, m;
  int **ind_tmp = NULL;

  if (max_faces == 0){
    max_faces = FACE_INC;
    memory->create(ind_faces,nnmax,max_faces,"fix_force_bound_2d:ind_faces");
  } else{
    m = max_faces;
    memory->create(ind_tmp,nnmax,m,"fix_force_bound_2d:ind_tmp");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_tmp[k][l] = ind_faces[k][l];
    max_faces += FACE_INC;
    memory->destroy(ind_faces);
    memory->create(ind_faces,nnmax,max_faces,"fix_force_bound_2d:ind_faces");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_faces[k][l] = ind_tmp[k][l];
    memory->destroy(ind_tmp);
  }
}

/* ------------------------------------------------------------------------- */

double FixForceBound2D::solve_quadratic(double *cc)
{
  double tt,dot,t1,t2,uv;

  if (cc[1] == 0.0){
    if (cc[2] != 0.0){
      tt = -cc[3]/cc[2];
      return tt; 
    } else
      return -1.0;
  }

  tt = -1;
  dot = cc[2]*cc[2] - 4.0*cc[1]*cc[3];
  if (dot >= 0){
    t1 = 0.5*(-cc[2]+sqrt(dot))/cc[1];
    t2 = 0.5*(-cc[2]-sqrt(dot))/cc[1];
    if (t2 < t1) {
      uv = t2;
      t2 = t1;
      t1 = uv;
    }
    if (t2>=0.0 && t2<=1.0)
      tt = t2;
    if (t1>=0.0 && t1<=1.0)
      tt = t1;
  }
  
  return tt;
}

/* ----------------------------------------------------------------------
   lees-edwards fix
------------------------------------------------------------------------- */

void FixForceBound2D::lees_edwards(int n)
{
  int i,j;
  double ss;

  if (n && le_max < le_npart - npart_loc){
    le_max = le_npart - npart_loc;
    memory->grow(le_sh,le_max,"fix_force_bound_2d:le_sh");
  }

  if (fabs(domain->boxhi[0] - domain->subhi[0]) < EPSSS){
    ss = 0.5*(domain->subhi[0]+domain->sublo[0]);
    for (i = npart_loc; i < le_npart; i++)
      if (x_part[i][0] > ss){
        x_part[i][1] += comm->shift;
        v_part[i][1] += comm->u_le;
        j = i - npart_loc;
        if (n){
          le_sh[j] = 0;
          while (x_part[i][1] >= domain->boxhi[1]){
            x_part[i][1] -= prd[1];
            le_sh[j]--;
          }
          while (x_part[i][1] < domain->boxlo[1]){
            x_part[i][1] += prd[1];
            le_sh[j]++;
          }
        } else{
          x_part[i][1] += le_sh[j]*prd[1];
        }
      }
  }

  if (fabs(domain->boxlo[0] - domain->sublo[0]) < EPSSS){
    ss = 0.5*(domain->subhi[0]+domain->sublo[0]);
    for (i = npart_loc; i < le_npart; i++)
      if (x_part[i][0] < ss){
        x_part[i][1] -= comm->shift;
        v_part[i][1] -= comm->u_le;
        j = i - npart_loc;
        if (n){
          le_sh[j] = 0;
          while (x_part[i][1] >= domain->boxhi[1]){
            x_part[i][1] -= prd[1];
            le_sh[j]--;
          }
          while (x_part[i][1] < domain->boxlo[1]){
            x_part[i][1] += prd[1];
            le_sh[j]++;
          }
        } else{
          x_part[i][1] += le_sh[j]*prd[1];
        }
      }
  }
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixForceBound2D::write_restart(FILE *fp)
{
  int i,j,l,m,n,kk,nel_tot,offset,ind_sh;
  int *displs, *rcounts;
  double  *list, *tmp, *tmph;

  displs = rcounts = NULL;
  list = tmp = tmph = NULL;

  if (numt_loc)
    offset = numt_loc;
  else
    offset = 1;
  memory->create(tmp,offset,"fix_force_bound_2d:tmp");

  nel_tot = 1;
  memory->create(rcounts,comm->nprocs,"fix_force_bound_2d:rcounts");
  memory->create(displs,comm->nprocs,"fix_force_bound_2d:displs");
  if (comm->nprocs > 1){
    MPI_Gather(&numt_loc, 1, MPI_INT, rcounts, 1, MPI_INT, 0, world);
    if (me == 0){
      nel_tot = 0;
      offset = 0;
      for (i = 0; i < comm->nprocs; i++) {
        nel_tot += rcounts[i];
        displs[i] = offset;
        offset += rcounts[i];
      }
    }
  } else{
    nel_tot = numt_loc;
  }

  l = 0;
  for (i=0; i<num_shapes; i++){
    m = shapes_local[i];
    tmp[l] = static_cast<double>(m);
    l++;
    for (j=0; j<ndiv[m]; j++){
      tmp[l] = fsx[i][0][j];
      tmp[l+1] = fsy[i][0][j];
      tmp[l+2] = fsx[i][1][j];
      tmp[l+3] = fsy[i][1][j];
      l += 4;
    }
  }

  if (l != numt_loc)
    error->warning(FLERR,"Something might be wrong: numt_loc and l are not equal in FixForceBound2D::write_restart!");

  memory->create(tmph,nel_tot,"fix_force_bound_2d:tmph");
  if (comm->nprocs > 1){
    MPI_Gatherv(tmp, l, MPI_DOUBLE, tmph, rcounts, displs, MPI_DOUBLE, 0, world);
  } else{
    for (i=0; i<nel_tot; i++)
      tmph[i] = tmp[i];
  }
  
  if (me == 0){
    memory->create(list,numt_s*2+1,"fix_force_bound_2d:list");
    kk = 0;
    for (i=0; i<num_shapes_tot; i++){
      l = 0;
      ind_sh = 0;
      while (l < nel_tot){
        m = static_cast<int>(tmph[l]);
        l++;
        if (m == i){
          ind_sh = 1;
          break;
        } else
          l += 4*ndiv[m];
      }
      if (ind_sh){
        for (j=0; j<ndiv[i]; j++){
          for (n=0; n<4; n++)
            list[kk+n] = tmph[l+n];
          kk += 4;    
          l += 4;
        }
      } else{
        for (j=0; j<ndiv[i]; j++){
          for (n=0; n<4; n++)
            list[kk+n] = 0.0;
          kk += 4;
	}
      }
    }
    if (kk != numt_s*2)
      error->warning(FLERR,"Something might be wrong: kk and 2*numt_s are not equal in FixForceBound2D::write_restart!");
    list[kk] = static_cast<double> (cur_iter);
    kk++;   

    l = kk * sizeof(double);
    fwrite(&l,sizeof(int),1,fp);
    fwrite(list,sizeof(double),kk,fp);

    memory->destroy(list); 
  }

  memory->destroy(tmp);
  memory->destroy(tmph);
  memory->destroy(rcounts);
  memory->destroy(displs);
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixForceBound2D::restart(char *buf)
{
  int i,j,l,m;
  double *list = (double *) buf;  

  l = 0;
  for (i=0; i<num_shapes_tot; i++)
    for (j=0; j<ndiv[i]; j++){
      m = shapes_global_to_local[i];
      if (m > -1){
        fsx[m][0][j] = list[l];
        fsy[m][0][j] = list[l+1];
        fsx[m][1][j] = list[l+2];
        fsy[m][1][j] = list[l+3];
      }
      l += 4;
    }
  cur_iter = static_cast<int> (list[l]);
  
  read_restart_ind = 1;

  /*char fname[FILENAME_MAX]; 
  FILE *f_write;
  sprintf(fname,"read_bla_%d.dat",me);
  f_write = fopen(fname,"w");
  fprintf(f_write,"%d \n",cur_iter);
  for (i=0; i<num_shapes; i++){
    m = shapes_local[i];
    fprintf(f_write,"%d \n",m);
    for (j=0; j<ndiv[m]; j++){
      fprintf(f_write,"%lf %lf \n",fsx[i][0][j], fsy[i][0][j]);
      fprintf(f_write,"%lf %lf \n",fsx[i][1][j], fsy[i][1][j]);
    }
  }
  fclose(f_write);*/  
}

/* ----------------------------------------------------------------------
   allocation of swap arrays
------------------------------------------------------------------------- */

void FixForceBound2D::allocate_swap(int n)
{
  int i;

  memory->create(sendnum,n,"fix_force_bound_2d:sendnum");
  memory->create(recvnum,n,"fix_force_bound_2d:recvnum");
  memory->create(sendproc,n,"fix_force_bound_2d:sendproc");
  memory->create(recvproc,n,"fix_force_bound_2d:recvproc");
  memory->create(size_forward_recv,n,"fix_force_bound_2d:size");
  memory->create(size_reverse_send,n,"fix_force_bound_2d:size");
  memory->create(size_reverse_recv,n,"fix_force_bound_2d:size");
  memory->create(firstrecv,n,"fix_force_bound_2d:firstrecv");
  memory->create(pbc_flag,n,"fix_force_bound_2d:pbc_flag");
  memory->create(pbc,n,6,"fix_force_bound_2d:pbc");
  
  if (user_part == 0){
    memory->create(slablo,n,"fix_force_bound_2d:slablo");
    memory->create(slabhi,n,"fix_force_bound_2d:slabhi");
  }

  sendlist = (int **) memory->smalloc(n*sizeof(int *),"fix_force_bound_2d:sendlist");
  memory->create(maxsendlist,n,"fix_force_bound_2d:maxsendlist");
  for (i = 0; i < n; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i],BUFMIN,"fix_force_bound_2d:sendlist[i]");
  } 
}

/* ----------------------------------------------------------------------
   free memory for swaps
------------------------------------------------------------------------- */

void FixForceBound2D::free_swap()
{
  int i;

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

  if (user_part == 0){
    memory->destroy(slablo);
    memory->destroy(slabhi);  
  }

  memory->destroy(maxsendlist);
  if (sendlist) 
    for (i = 0; i < maxswap; i++) 
      memory->destroy(sendlist[i]);
  memory->sfree(sendlist);
}

/* ------------------------------------------------------------------------- */


