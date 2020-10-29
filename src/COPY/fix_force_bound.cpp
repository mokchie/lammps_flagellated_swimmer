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
#include "fix_force_bound.h"
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
#define EPSN 1.0e-32
#define BIG 1.0e20
#define DR 0.01
#define MAX_GBITS 64

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixForceBound::FixForceBound(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  int i, j, l, ttyp, igroup, nms, nw_max; 
  double *rdata;
  double dummy,a1[3],a2[3],rr,wd; 
  char grp[50],grp1[50],grp2[50];
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  char fname1[FILENAME_MAX];
  FILE *f_read, *f_read1;

  groupbit_s = groupbit_p = groupbit_p_cell = groupbit_force = groupbit_force_cell = 0;
  f_press = f_force = NULL;
  weight = NULL;
  dr_inv = 1.0/DR;
  if (narg != 4) error->all(FLERR,"Illegal fix force/bound command");
  sprintf(fname,arg[3]);
  user_part = comm->user_part;
  send_bit = NULL;
  m_comm = NULL;
  me = comm->me;

  nms = NUM;
  cur_iter = 1;
  memory->create(rdata,nms,"fix_force_bound:rdata");
  size_border = 4;
  size_forward = 6;
  size_reverse = 3; 
  read_restart_ind = 0;
  ind_press_cell = 0;
  ind_force_cell = 0;
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
      sscanf(buf,"%d %d %lf %d %s %s %s",&ind_press,&n_press,&r_press,&p_apply,&fname1[0],&grp[0],&grp1[0]);
      rdata[l++] = static_cast<double> (n_press);
      rdata[l++] = r_press;
      rdata[l++] = static_cast<double> (p_apply);
      igroup = group->find(grp);
      if (igroup == -1) error->one(FLERR,"Group ID for group_p does not exist");
      groupbit_p = group->bitmask[igroup];
      rdata[l++] = static_cast<double> (groupbit_p);
      igroup = group->find(grp1);
      if (igroup == -1) error->one(FLERR,"Group ID for group_p_cell does not exist");
      groupbit_p_cell = group->bitmask[igroup];    
      rdata[l++] = static_cast<double> (groupbit_p_cell);
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
      
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&ind_force);
    rdata[l++] = static_cast<double> (ind_force);
    if (ind_force){
      sscanf(buf,"%d %d %lf %d %s %s %s",&ind_force,&n_force,&r_force,&f_apply,&fname1[0],&grp[0],&grp1[0]);
      rdata[l++] = static_cast<double> (n_force);
      rdata[l++] = r_force;
      rdata[l++] = static_cast<double> (f_apply);
      igroup = group->find(grp);
      if (igroup == -1) error->one(FLERR,"Group ID for group_force does not exist");
      groupbit_force = group->bitmask[igroup];
      rdata[l++] = static_cast<double> (groupbit_force);
      igroup = group->find(grp1);
      if (igroup == -1) error->one(FLERR,"Group ID for group_force_cell does not exist");
      groupbit_force_cell = group->bitmask[igroup];    
      rdata[l++] = static_cast<double> (groupbit_force_cell);
      f_read1 = fopen(fname1,"r");
      if (f_read1 == (FILE*) NULL)
        error->one(FLERR,"Could not open input force file");
      for (j=0; j<n_force; j++){
        fgets(buf,BUFSIZ,f_read1);
        sscanf(buf,"%lf %lf",&dummy,&rdata[l]);
        l++;
      }
      fclose(f_read1);
    }  
     
    for (i=0; i<num_shapes_tot; i++){
      fgets(buf,BUFSIZ,f_read);
      sscanf(buf,"%d",&ttyp);
      if (l + 16 > nms){
        nms += NUM;
        memory->grow(rdata,nms,"fix_force_bound:rdata");
      }
      switch (ttyp){
        case 1 :             
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8],&rdata[l+9],&rdata[l+10],&rdata[l+11],&rdata[l+12],&rdata[l+13]);
          l += 14;
          break;
        case 2 :
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8],&rdata[l+9],&rdata[l+10],&rdata[l+11],&rdata[l+12],&rdata[l+13],&rdata[l+14], &rdata[l+15]);
          l += 16;
          break;  
        case 3 :
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8],&rdata[l+9],&rdata[l+10],&rdata[l+11],&rdata[l+12],&rdata[l+13]);
          l += 14;
          break;   
        case 4 :
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8],&rdata[l+9],&rdata[l+10]);
          l += 11;
          break; 
      } 
    }
    fclose(f_read);
  } 
  
  MPI_Bcast(&l,1,MPI_INT,0,world);
  if (l > NUM)
    memory->grow(rdata,l,"fix_force_bound:rdata");
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
    memory->create(weight,nw_max,"fix_force_bound:weight");
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
    if (ind_press == 2){
      ind_press = 0;
      ind_press_cell = 1;
    } else if (ind_press == 3){
      ind_press = 1;
      ind_press_cell = 1;
    }              
    n_press = static_cast<int> (rdata[l++]);
    r_press = rdata[l++];
    p_apply = static_cast<int> (rdata[l++]);
    groupbit_p = static_cast<int> (rdata[l++]);
    groupbit_p_cell = static_cast<int> (rdata[l++]);
    memory->create(f_press,n_press+1,"fix_force_bound:f_press");
    for (j=0; j<n_press; j++)
      f_press[j] = rdata[l++];
    f_press[n_press] = 0.0;
  }
  ind_force = static_cast<int> (rdata[l++]);
  if (ind_force){
    if (ind_force == 2){
      ind_force = 0;
      ind_force_cell = 1;
    } else if (ind_force == 3){
      ind_force = 1;
      ind_force_cell = 1;
    }                           
    n_force = static_cast<int> (rdata[l++]);
    r_force = rdata[l++];
    f_apply = static_cast<int> (rdata[l++]);
    groupbit_force = static_cast<int> (rdata[l++]);
    groupbit_force_cell = static_cast<int> (rdata[l++]);
    memory->create(f_force,n_force+1,"fix_force_bound:f_force");
    for (j=0; j<n_force; j++)
      f_force[j] = rdata[l++];
    f_force[n_force] = 0.0;
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
  memory->create(x0,num_shapes_tot,3,"fix_force_bound:x0");
  memory->create(aa,num_shapes_tot,8,"fix_force_bound:aa");
  memory->create(vel,num_shapes_tot,3,"fix_force_bound:vel");
  memory->create(rot,num_shapes_tot,3,3,"fix_force_bound:rot"); 
  memory->create(ndiv,num_shapes_tot,2,"fix_force_bound:ndiv");
  memory->create(ptype,num_shapes_tot,"fix_force_bound:ptype");
  memory->create(refl,num_shapes_tot,"fix_force_bound:refl");
  memory->create(face_order,num_shapes_tot,"fix_force_bound:face_order");
  memory->create(shapes_local,num_shapes_tot,"fix_force_bound:shapes_local");
  memory->create(shapes_global_to_local,num_shapes_tot,"fix_force_bound:shapes_global_to_local");
  bin_shapes = bin_vertex = bin_stensil = NULL;
  num_bin_shapes = num_bin_vertex = num_bin_stensil = NULL;
  ind_shapes = NULL; 
  tot_shapes = num_faces = NULL;
  le_sh = NULL; 
  memory->create(tot_shapes,nnmax,"fix_force_bound:tot_shapes");
  memory->create(num_faces,nnmax,"fix_force_bound:num_faces");       
        
  for (i=0; i<num_shapes_tot; i++){
    ptype[i] = static_cast<int> (rdata[l]);
    switch (ptype[i]){
      case 1 :
        for (j=0; j<3; j++){
          x0[i][j] = rdata[l+j+1];
          a1[j] = rdata[l+j+4] - rdata[l+j+1];
          a2[j] = rdata[l+j+7] - rdata[l+j+1];
          vel[i][j] = rdata[l+j+10];
	    }       
        ndiv[i][0] = 1;
        ndiv[i][1] = 1;    
        refl[i] = static_cast<int> (rdata[l+13]);
        l += 14; 
        break;
      case 2 :           
        for (j=0; j<3; j++){
          x0[i][j] = rdata[l+j+1];
          a1[j] = rdata[l+j+4] - rdata[l+j+1];
          a2[j] = rdata[l+j+7] - rdata[l+j+1];
          vel[i][j] = rdata[l+j+10];
	    }       
        ndiv[i][0] = static_cast<int> (rdata[l+13]);
        ndiv[i][1] = static_cast<int> (rdata[l+14]);    
        refl[i] = static_cast<int> (rdata[l+15]);
        l += 16; 
        break;
      case 3 :          
        for (j=0; j<3; j++){
          x0[i][j] = rdata[l+j+1];
          a1[j] = rdata[l+j+4] - rdata[l+j+1];
          vel[i][j] = rdata[l+j+8];
	    }       
        aa[i][1] = rdata[l+7];
        ndiv[i][0] = static_cast<int> (rdata[l+11]);
        ndiv[i][1] = static_cast<int> (rdata[l+12]);    
        refl[i] = static_cast<int> (rdata[l+13]);
        l += 14; 
        break;
      case 4 :           
        for (j=0; j<3; j++){
          x0[i][j] = rdata[l+j+1];
          vel[i][j] = rdata[l+j+5];
	    }       
        aa[i][0] = rdata[l+4];
        ndiv[i][0] = static_cast<int> (rdata[l+8]);
        ndiv[i][1] = static_cast<int> (rdata[l+9]);    
        refl[i] = static_cast<int> (rdata[l+10]);
        l += 11;  
        break; 
    }
    setup_rot(i,a1,a2);                    
    rot_forward(vel[i][0],vel[i][1],vel[i][2],i);
    face_order[i] = numt;
    numt_s += 2*ndiv[i][0]*ndiv[i][1];
    numt += 4*2*n_per*ndiv[i][0]*ndiv[i][1];
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
    velx = new double***[num_shapes];
    vely = new double***[num_shapes];
    velz = new double***[num_shapes];
    num = new double***[num_shapes];
    fsx = new double***[num_shapes];
    fsy = new double***[num_shapes];
    fsz = new double***[num_shapes]; 
    for (i=0; i<num_shapes; i++){
      j = shapes_local[i];  
      memory->create(velx[i],2*n_per,ndiv[j][0],ndiv[j][1],"fix_force_bound:velx[i]");
      memory->create(vely[i],2*n_per,ndiv[j][0],ndiv[j][1],"fix_force_bound:vely[i]");
      memory->create(velz[i],2*n_per,ndiv[j][0],ndiv[j][1],"fix_force_bound:velz[i]");
      memory->create(num[i],2*n_per,ndiv[j][0],ndiv[j][1],"fix_force_bound:num[i]");
      memory->create(fsx[i],2,ndiv[j][0],ndiv[j][1],"fix_force_bound:fsx[i]");
      memory->create(fsy[i],2,ndiv[j][0],ndiv[j][1],"fix_force_bound:fsy[i]");
      memory->create(fsz[i],2,ndiv[j][0],ndiv[j][1],"fix_force_bound:fsz[i]");
      numt_loc += 6*ndiv[j][0]*ndiv[j][1];
    }
    numt_loc += num_shapes;
  } else{
    velx = vely = velz = NULL;
    num = NULL;
    fsx = fsy = fsz = NULL;
  }

  part_degree = tags_part = norm_ind = NULL;
  tri_mol = tri_list = part_list = NULL;
  part_tri = tri_tags = NULL;
  x_part = v_part = v_delt = f_part = NULL;
  norm = edge1 = edge2 = dif1 = dif2 = dd12 = NULL;
  norm_list = NULL;
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

FixForceBound::~FixForceBound()
{
  int i;

  if (ind_press || ind_press_cell)
    memory->destroy(f_press);
  if (ind_force || ind_force_cell)
    memory->destroy(f_force);
  if (ind_shear) 
    memory->destroy(weight);
  if (ind_shear && num_shapes){
    for (i=0; i<num_shapes; i++){
      memory->destroy(velx[i]);
      memory->destroy(vely[i]);
      memory->destroy(velz[i]);
      memory->destroy(num[i]);
      memory->destroy(fsx[i]);
      memory->destroy(fsy[i]);
      memory->destroy(fsz[i]);
    }
    delete[] velx;
    delete[] vely;
    delete[] velz;
    delete[] num;
    delete[] fsx;
    delete[] fsy;
    delete[] fsz;
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
  memory->destroy(tri_mol);
  memory->destroy(tri_list);
  memory->destroy(part_list);
  memory->destroy(norm_list);
  memory->destroy(part_tri);
  memory->destroy(tri_tags);
  memory->destroy(x_part);
  memory->destroy(v_part); 
  memory->destroy(v_delt); 
  memory->destroy(f_part); 

  if (user_part){
    memory->destroy(m_comm);
    memory->destroy(send_bit);
  }
  
  free_swap();
  memory->destroy(buf_send);
  memory->destroy(buf_recv);

  memory->destroy(norm_ind);   
  memory->destroy(norm);
  memory->destroy(edge1);
  memory->destroy(edge2);
  memory->destroy(dif1);
  memory->destroy(dif2);
  memory->destroy(dd12); 
  if (max_faces) memory->destroy(ind_faces);
  memory->destroy(num_faces);
  memory->destroy(mol_mass);
}

/* ---------------------------------------------------------------------- */

int FixForceBound::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixForceBound::setup(int vflag)
{
  int l,ii;
  int iswap, dim, ineed, nbox,offset;
  tagint i,j,k,max_tri,n_ang;
  tagint *tag = atom->tag; 
  int *mask = atom->mask;
  tagint *molecule = atom->molecule; 
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int *displs, *rcounts;
  tagint *bff, *bff1;  
  double mss[atom->n_mol+1];

  n_accum = 0;  
  // zero out initial arrays
  if (ind_shear && num_shapes)
    for (i = 0; i < num_shapes; i++){
      ii = shapes_local[i];    
      for (k = 0; k < ndiv[ii][0]; k++)
        for (l = 0; l < ndiv[ii][1]; l++){
          for (j = 0; j < 2*n_per; j++){
            velx[i][j][k][l] = 0.0;
            vely[i][j][k][l] = 0.0;
            velz[i][j][k][l] = 0.0;
            num[i][j][k][l] = 0.0;
          }
          if (!read_restart_ind)
            for (j = 0; j < 2; j++){
              fsx[i][j][k][l] = 0.0;
              fsy[i][j][k][l] = 0.0;
              fsz[i][j][k][l] = 0.0;
	    }
	}
    } 

  // assign atom masses 
  memory->create(mol_mass,atom->n_mol+1,"fix_force_bound:mol_mass");
  for (i = 0; i < atom->n_mol+1; i++){
    mss[i] = 0.0;
    mol_mass[i] = 0.0; 
  }
  for (i = 0; i < atom->nlocal; i++)  
    if (atom->mass[atom->type[i]] > mss[molecule[i]])
      mss[molecule[i]] = atom->mass[atom->type[i]];
  MPI_Allreduce(&mss,mol_mass,atom->n_mol+1,MPI_DOUBLE,MPI_MAX,world); 

  // construct arrays and maps for cell structures
  memory->create(rcounts,comm->nprocs,"fix_force_bound:rcounts");
  memory->create(displs,comm->nprocs,"fix_force_bound:displs");
  n_ang = 0;
  for (i = 0; i < nanglelist; i++)
    if (mask[anglelist[i][1]] & groupbit_comm)
      n_ang++;
   
  l = static_cast<int>(n_ang); 
  if (comm->nprocs > 1){
    MPI_Gather(&l, 1, MPI_INT, rcounts, 1, MPI_INT, 0, world);
    //MPI_Gather(&n_ang, 1, MPI_LMP_TAGINT, rcounts, 1, MPI_LMP_TAGINT, 0, world);
    if (me == 0){
      ntri_tot = 0;  
      offset = 0;
      for (i = 0; i < comm->nprocs; i++) {
        ntri_tot += rcounts[i];  
        displs[i] = offset;
        rcounts[i] *= 4;
        offset += rcounts[i];
      }
    }
    MPI_Bcast(&ntri_tot,1,MPI_LMP_TAGINT,0,world); 
  } else{
    ntri_tot = n_ang; 
  }

  memory->create(bff,ntri_tot*4,"fix_force_bound:bff");
  memory->create(bff1,ntri_tot*4,"fix_force_bound:bff1");
  j = 0;

  if (comm->nprocs > 1){
    for (i = 0; i < nanglelist; i++) 
      if (mask[anglelist[i][1]] & groupbit_comm){
        bff1[4*j] = tag[anglelist[i][0]];
        bff1[4*j+1] = tag[anglelist[i][1]];
        bff1[4*j+2] = tag[anglelist[i][2]];
        bff1[4*j+3] = molecule[anglelist[i][0]];
        j++;
      }
    MPI_Gatherv(bff1, 4*n_ang, MPI_LMP_TAGINT, bff, rcounts, displs, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(bff,4*ntri_tot,MPI_LMP_TAGINT,0,world);
  } else{
    for (i = 0; i < nanglelist; i++) 
      if (mask[anglelist[i][1]] & groupbit_comm){
        bff[4*j] = tag[anglelist[i][0]];
        bff[4*j+1] = tag[anglelist[i][1]];
        bff[4*j+2] = tag[anglelist[i][2]];
        bff[4*j+3] = molecule[anglelist[i][0]];
	j++;
      }
  }

  memory->create(tri_tags,ntri_tot,3,"fix_force_bound:tri_tags");
  memory->create(tri_mol,ntri_tot,"fix_force_bound:tri_mol");
  min_ind = 999999999;
  max_ind = -1;
  for (i = 0; i < ntri_tot; i++){
    tri_mol[i] = bff[4*i+3]; 
    for (j = 0; j < 3; j++){
      tri_tags[i][j] = bff[4*i+j];
      if (tri_tags[i][j] < min_ind) min_ind = tri_tags[i][j];
      if (tri_tags[i][j] > max_ind) max_ind = tri_tags[i][j];   
    } 
  }
  npart_tot = max_ind - min_ind + 1;
  memory->destroy(bff);
  memory->destroy(bff1);

  max_ntri = max_npart = 0;
  npart = ntri = 0;
  norm_count = 0;
  memory->create(part_degree,npart_tot,"fix_force_bound:part_degree");
  memory->create(tags_part,npart_tot,"fix_force_bound:tags_part");
  memory->create(norm_ind,ntri_tot,"fix_force_bound:norm_ind");

  for (i = 0; i < npart_tot; i++){
    part_degree[i] = 0;
    tags_part[i] = -1;
  }

  for (i = 0; i < ntri_tot; i++){
    norm_ind[i] = -1;
    for (j = 0; j < 3; j++){
      tri_tags[i][j] -= min_ind;
      part_degree[tri_tags[i][j]]++;   
    }
  }
  max_tri = -1;
  for (i = 0; i < npart_tot; i++){
    if (part_degree[i] > max_tri) max_tri = part_degree[i];
    part_degree[i] = 0;
  }

  memory->create(part_tri,npart_tot,max_tri,"fix_force_bound:part_tri"); 
  for (i = 0; i < ntri_tot; i++)
    for (j = 0; j < 3; j++){
      k = tri_tags[i][j];
      part_tri[k][part_degree[k]] = i;
      part_degree[k]++;   
    }  

  memory->destroy(rcounts);
  memory->destroy(displs);

  /*char f_name[FILENAME_MAX]; 
  FILE* out_stat;
  sprintf(f_name,"output_%d.plt",me);
  out_stat=fopen(f_name,"w");
  fprintf(out_stat,"min_ind=%d, max_ind=%d, npart_tot=%d  \n", min_ind, max_ind, npart_tot);
  fprintf(out_stat,"ntri_tot=%d, max_tri=%d  \n", ntri_tot, max_tri); 
  fprintf(out_stat,"triangles  \n"); 
  for (i = 0; i < ntri_tot; i++)
    fprintf(out_stat,"%d    %ld %ld %ld mol=%d \n", i, tri_tags[i][0], tri_tags[i][1], tri_tags[i][2],tri_mol[i]);
  fprintf(out_stat,"indexes  \n"); 
  for (i = 0; i < npart_tot; i++){
    fprintf(out_stat,"%d   %ld    ", i, part_degree[i]);
    for (j = 0; j < part_degree[i]; j++)  
      fprintf(out_stat,"%ld ", part_tri[i][j]);
    fprintf(out_stat," \n"); 
  }
  fclose(out_stat);*/  

  // allocate arrays for communications
  maxsend = BUFMIN;
  memory->create(buf_send,maxsend+BUFEXTRA,"fix_force_bound:buf_send");
  maxrecv = BUFMIN;
  memory->create(buf_recv,maxrecv,"fix_force_bound:buf_recv");

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
  
    memory->create(m_comm,nbp_loc[0],nbp_loc[1],nbp_loc[2],"fix_force_bound:m_comm");
  
    for (i = 0; i < nbp_loc[0]; i++)
      for (j = 0; j < nbp_loc[1]; j++)
        for (k = 0; k < nbp_loc[2]; k++)
          m_comm[i][j][k] = 0;

    // create bit arrays
  
    nswap = n_neigh;
    maxswap = nswap;
    allocate_swap(maxswap);
     
    if (n_neigh > MAX_GBITS)
      error->one(FLERR,"Not enough bits for the communication matrix in fix_force_bound!");
  
    memory->create(send_bit,n_neigh,"fix_force_bound:send_bit");
    
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
      } else recvneed[i][0] = recvneed[i][1] =
               sendneed[i][0] = sendneed[i][1] = maxneed[i];
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
           memory->grow(bin_shapes,bin_max_shape,nbin,"fix_force_bound:bin_shapes"); 
	 }
       }  
    }
  bin_max_vertex = bin_max_shape;  
  memory->create(bin_vertex,bin_max_vertex,nbin,"fix_force_bound:bin_vertex");
  memory->create(num_bin_vertex,nbin,"fix_force_bound:num_bin_vertex");  

  post_force(vflag);
}
/* ---------------------------------------------------------------------- */

void FixForceBound::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixForceBound::post_integrate()
{
  int i,j,k,kk,ind,cond,i_x[NUM2],cc,bounce,cplane0,cplane1; 
  int ccount,i0,i1,i2,m0,shift[NUM2][3],i_b[NUM2];
  tagint m;
  double dl[3],dl1[3],t_x[NUM2],dd[3],xp[3],xh[3],vh[3],vh1[3],vv[3],norm_h[3],x_x[3]; 
  double xd[3],xc[3],cf[4],a1[3],a2[3],nnp[4],v_loc[3],xp1[3];
  double tt,u_inv,dot,d1,d2,d3,dtt,u0,u1,uu,uun,mss,mprt; 
  double dtv = update->dt; 

  double **x = atom->x;
  double **v = atom->v;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *tag = atom->tag; 

  //double vv_mom[3], vv_ang[3], shf_h[3][3];

  local_faces(1);
  if (comm->nprocs > 1) communicate();

  for (i = 0; i < npart; i++)
    for (j = 0; j < 3; j++)
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
          for (j = 0; j < 3; j++)
            v_loc[j] = v[i][j];    
        if (mask[i] & groupbit){
          for (j = 0; j < 3; j++){
            dl1[j] = v_loc[j]*dtt;
            xp1[j] = x[i][j] - dl1[j];
	  }
          for (k = 0; k < num_faces[i]; k++){
            tt = -1.0;
            m0 = ind_faces[i][k];
            if (m0 != cplane0){
              if (norm_ind[m0] == -1) calc_norm(m0);
              i0 = tags_part[tri_tags[m0][0]];
              m = norm_ind[m0]; 
              for (j = 0; j < 3; j++){
                x_x[j] = x_part[i0][j] - xp1[j]; 
                if (fabs(x_x[j]) > 0.5*prd[j]){
                  if (x_x[j] > 0)
                    shift[ind][j] = 1;
                  else
                    shift[ind][j] = -1;
	        } else
                  shift[ind][j] = 0; 
                a1[j] = edge1[m][j] - dif1[m][j]*dtt;
                a2[j] = edge2[m][j] - dif2[m][j]*dtt; 
	      }
              /*for (j = 0; j < 3; j++){
                if (fabs(edge1[m][j]) > 0.5*prd[j]) 
                  printf("me=%d, a1=%f %f %f \n",me,a1[0],a1[1],a1[2]);
                if (fabs(edge2[m][j]) > 0.5*prd[j]) 
                  printf("me=%d, a2=%f %f %f \n",me,a2[0],a2[1],a2[2]);                
		  } */
              nnp[0] = a1[1]*a2[2] - a1[2]*a2[1];
              nnp[1] = a1[2]*a2[0] - a1[0]*a2[2];
              nnp[2] = a1[0]*a2[1] - a1[1]*a2[0];
              nnp[3] = - nnp[0]*(x_part[i0][0] - v_part[i0][0]*dtt) - nnp[1]*(x_part[i0][1] - v_part[i0][1]*dtt) - nnp[2]*(x_part[i0][2] - v_part[i0][2]*dtt);    
              u0 = nnp[0]*(xp1[0]+prd[0]*shift[ind][0]) + nnp[1]*(xp1[1]+prd[1]*shift[ind][1]) + nnp[2]*(xp1[2]+prd[2]*shift[ind][2]) + nnp[3];
              if ((u0 <= 0.0 && mask[i] & groupbit_inner) || (u0 >= 0.0 && !(mask[i] & groupbit_inner))){
                u1 = norm[m][0]*(x[i][0]+prd[0]*shift[ind][0]) + norm[m][1]*(x[i][1]+prd[1]*shift[ind][1]) + norm[m][2]*(x[i][2]+prd[2]*shift[ind][2]) + norm[m][3];
                if (u0*u1 <= 0.0){ 
                  for (j = 0; j < 3; j++){
                    vv[j] = dl1[j] - dtt*v_part[i0][j];
                    xd[j] = xp1[j]+prd[j]*shift[ind][j] - x_part[i0][j] + dtt*v_part[i0][j];
	          }
                  xc[0] = dtt*(a1[1]*dif2[m][2] - a1[2]*dif2[m][1] + dif1[m][1]*a2[2] - dif1[m][2]*a2[1]);
                  xc[1] = dtt*(a1[2]*dif2[m][0] - a1[0]*dif2[m][2] + dif1[m][2]*a2[0] - dif1[m][0]*a2[2]);
                  xc[2] = dtt*(a1[0]*dif2[m][1] - a1[1]*dif2[m][0] + dif1[m][0]*a2[1] - dif1[m][1]*a2[0]);
                  cf[0] = dtt*dtt*(dd12[m][0]*vv[0] + dd12[m][1]*vv[1] + dd12[m][2]*vv[2]);
                  cf[1] = dtt*dtt*(dd12[m][0]*xd[0] + dd12[m][1]*xd[1] + dd12[m][2]*xd[2]) + xc[0]*vv[0] + xc[1]*vv[1] + xc[2]*vv[2]; 
                  cf[2] = xc[0]*xd[0] + xc[1]*xd[1] + xc[2]*xd[2] + nnp[0]*vv[0] + nnp[1]*vv[1] + nnp[2]*vv[2];
                  cf[3] = nnp[0]*xd[0] + nnp[1]*xd[1] + nnp[2]*xd[2];  
                  tt = find_root(cf);
                  //tt =  solve_quadratic(cf);
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
              for (j = 0; j < 3; j++){
                xh[j] = x[i][j] - x0[kk][j];
                vh[j] = v_loc[j];
	      }  
              rot_forward(xh[0],xh[1],xh[2],kk);
              rot_forward(vh[0],vh[1],vh[2],kk);  
              for (j = 0; j < 3; j++){
                dl[j] = vh[j]*dtt;
                xp[j] = xh[j] - dl[j]; 
	      }
           
              tt = -1.0;
              if (ptype[kk]<3){
                if (xh[2]*xp[2] < 0.0)
                  if (refl[kk] == 3 || (xp[2]>0.0 && refl[kk] == 1) || (xp[2]<0.0 && refl[kk] == 2))
	            tt = xp[2]/(xp[2]-xh[2]);  
	      } else if (ptype[kk] == 3){  
                d1 = sqrt(xh[0]*xh[0] + xh[1]*xh[1]);
                d2 = sqrt(xp[0]*xp[0] + xp[1]*xp[1]);
                if ((d1-aa[kk][1])*(d2-aa[kk][1]) < 0.0)
                  if (refl[kk] == 3 || (d2>aa[kk][1] && refl[kk] == 1) || (d2<aa[kk][1] && refl[kk] == 2)){
                    vv[0] = dl[0]*dl[0] + dl[1]*dl[1];
                    vv[1] = (xp[0]*dl[0] + xp[1]*dl[1])/vv[0];
                    vv[2] = (d2*d2-aa[kk][1]*aa[kk][1])/vv[0];
                    dot =  vv[1]*vv[1] - vv[2];
                    if (dot > 0.0){
                      if (vv[2] > 0) 
	                tt = -vv[1] - sqrt(dot);
	              else
	                tt = -vv[1] + sqrt(dot);

                      //if (tt < EPS) tt = -1.0;  
		    }
	          }
	      } else {
                d1 = sqrt(xh[0]*xh[0] + xh[1]*xh[1] + xh[2]*xh[2]);
                d2 = sqrt(xp[0]*xp[0] + xp[1]*xp[1] + xp[2]*xp[2]);
                if ((d1-aa[kk][0])*(d2-aa[kk][0]) < 0.0)
                  if (refl[kk] == 3 || (d2>aa[kk][0] && refl[kk] == 1) || (d2<aa[kk][0] && refl[kk] == 2)){
                    vv[0] = dl[0]*dl[0] + dl[1]*dl[1] + dl[2]*dl[2];
                    vv[1] = (xp[0]*dl[0] + xp[1]*dl[1] + xp[2]*dl[2])/vv[0];
                    vv[2] = (d2*d2-aa[kk][0]*aa[kk][0])/vv[0];
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
            for (k = 0; k < 3; k++){
              m = shift[kk][k];
              shift[kk][k] = shift[ind-j][k];
              shift[ind-j][k] = m;
	    }  
          }
	  while (ind){
            bounce = 1;
            m = ind - 1;
            tt = t_x[m];
            if (i_b[m]){
	      kk = i_x[m];
              for (k = 0; k < 3; k++){
                xp[k] = x[i][k] - v_loc[k]*dtt - x0[kk][k];
                vh[k] = v_loc[k];
                vh1[k] = v[i][k];
	      }  
              rot_forward(xp[0],xp[1],xp[2],kk);
              rot_forward(vh[0],vh[1],vh[2],kk);
              rot_forward(vh1[0],vh1[1],vh1[2],kk);
              for (k = 0; k < 3; k++)
	        dd[k] = xp[k] + tt*vh[k]*dtt;
         
              switch (ptype[kk]){
                case 1 : 
                  u1 = dd[1]/aa[kk][2]; 
                  u0 = (dd[0]-u1*aa[kk][1])/aa[kk][0];
                  if (u0 < 0.0 || u1 < 0.0 || u0+u1 > 1.0)
                    bounce = 0;
                  break;
                case 2 : 
                  u1 = dd[1]/aa[kk][2]; 
                  u0 = (dd[0]-u1*aa[kk][1])/aa[kk][0];
                  if (u0 < 0.0 || u1 < 0.0 || u0 > 1.0 || u1 > 1.0)
                    bounce = 0;
                  break;
                case 3 :
                  norm_h[0] = dd[0]/aa[kk][1];
                  norm_h[1] = dd[1]/aa[kk][1]; 
                  norm_h[2] = 0.0;
                  break;
                case 4 :
                  for (k = 0; k < 3; k++)
                    norm_h[k] = dd[k]/aa[kk][0];          
                  break;
	      }
	    } else{
	      m0 = i_x[m];
              i0 = tags_part[tri_tags[m0][0]];
              i1 = tags_part[tri_tags[m0][1]];
              i2 = tags_part[tri_tags[m0][2]];
              kk = norm_ind[m0];
              for (k = 0; k < 3; k++){
	        x_x[k] = xp1[k] + prd[k]*shift[m][k] + tt*dl1[k];
	        xc[k] = x_x[k] - x_part[i0][k] + dtt*(1.0-tt)*v_part[i0][k];
                a1[k] = edge1[kk][k] - dif1[kk][k]*dtt*(1.0-tt);
                a2[k] = edge2[kk][k] - dif2[kk][k]*dtt*(1.0-tt);                	              
	      }
	      xd[0] = a1[0]*xc[0] + a1[1]*xc[1] + a1[2]*xc[2];
              xd[1] = a2[0]*xc[0] + a2[1]*xc[1] + a2[2]*xc[2];
              vv[0] = a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2];
              vv[1] = a2[0]*a2[0] + a2[1]*a2[1] + a2[2]*a2[2];
              vv[2] = a1[0]*a2[0] + a1[1]*a2[1] + a1[2]*a2[2]; 
              u_inv = vv[0]*vv[1] - vv[2]*vv[2];
              u0 = (xd[0]*vv[1] - xd[1]*vv[2])/u_inv;
              u1 = (xd[1]*vv[0] - xd[0]*vv[2])/u_inv;  
	      if (u0 <= 0.0 || u1 <= 0.0 || u0+u1 >= 1.0)
                bounce = 0;
	    }
            
	    if (bounce){
              ccount++;
              if (i_b[m]){
                cplane1 = kk; 
                if (mirror){
                  if (ptype[kk]<3){
                    vh[2] = -vh[2];
                    vh1[2] = -vh1[2];
		  } else{
                    d1 = 2.0 * (vh[0]*norm_h[0] + vh[1]*norm_h[1] + vh[2]*norm_h[2]);
                    d2 = 2.0 * (vh1[0]*norm_h[0] + vh1[1]*norm_h[1] + vh1[2]*norm_h[2]);
                    for (k = 0; k < 3; k++){ 
                      vh[k] += -d1*norm_h[k];     
                      vh1[k] += -d2*norm_h[k];
		    }
		  }
	        }
                else {
                  for (k = 0; k < 3; k++){
                    vh[k] = -vh[k] + 2.0*vel[kk][k];     
                    vh1[k] = -vh1[k] + 2.0*vel[kk][k];
		  }
	        }
                
                for (k = 0; k < 3; k++){
                  xp[k] = dd[k];      
                  xh[k] = xp[k] + (1.0-tt)*dtt*vh[k];      // verlet 
	        }
                rot_back(xh[0],xh[1],xh[2],kk);
                rot_back(vh[0],vh[1],vh[2],kk);
                rot_back(vh1[0],vh1[1],vh1[2],kk);
                for (k = 0; k < 3; k++){
                  x[i][k] = xh[k] + x0[kk][k];
                  v_loc[k] = vh[k];
                  v[i][k] = vh1[k];
	        }
	      } else{
                cplane0 = m0; 
                for (k = 0; k < 3; k++){
                  vv[k] = (1.0-u0-u1)*v_part[i0][k] + u0*v_part[i1][k] + u1*v_part[i2][k];  
                  xd[k] = (v_part[i0][k] + v_part[i1][k] + v_part[i2][k])/3.0;
	        }
                mss = mol_mass[tri_mol[m0]];
                mprt = mass[type[i]];   

                /*for (k = 0; k < 3; k++){
                  shf_h[0][k] = x_part[i0][k] - xp1[k];
                  shf_h[1][k] = x_part[i1][k] - xp1[k];
                  shf_h[2][k] = x_part[i2][k] - xp1[k];
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
                  if (fabs(shf_h[2][k]) > 0.5*prd[k]){
                    if (shf_h[2][k] > 0.0)
                      shf_h[2][k] = -prd[k];
                    else
                      shf_h[2][k] = prd[k];
                  } else shf_h[2][k] = 0.0;  
                  d1 = sqrt((x_part[i0][0]-x_part[i1][0])*(x_part[i0][0]-x_part[i1][0]) + (x_part[i0][1]-x_part[i1][1])*(x_part[i0][1]-x_part[i1][1]) + (x_part[i0][2]-x_part[i1][2])*(x_part[i0][2]-x_part[i1][2])); 
                  d2 = sqrt((x_part[i0][0]-x_part[i2][0])*(x_part[i0][0]-x_part[i2][0]) + (x_part[i0][1]-x_part[i2][1])*(x_part[i0][1]-x_part[i2][1]) + (x_part[i0][2]-x_part[i2][2])*(x_part[i0][2]-x_part[i2][2]));
                  d3 = sqrt((x_part[i2][0]-x_part[i1][0])*(x_part[i2][0]-x_part[i1][0]) + (x_part[i2][1]-x_part[i1][1])*(x_part[i2][1]-x_part[i1][1]) + (x_part[i2][2]-x_part[i1][2])*(x_part[i2][2]-x_part[i1][2]));
                  //if (d1 > 3.0 || d2 > 3.0 || d3 > 3.0) printf("Large distances: %15.13lf %15.13lf %15.13lf \n",d1,d2,d3);
                  vv_mom[k] = 0.0;
                  vv_mom[k] += mprt*v[i][k];
                  vv_mom[k] += mss*(v_part[i0][k] + v_part[i1][k] + v_part[i2][k]); 
                  vv_ang[k] = 0.0;
                }
                vv_ang[0] += mprt*(xp1[1]*v[i][2] - xp1[2]*v[i][1]) + mss*((x_part[i0][1]+shf_h[0][1]-dtv*v_part[i0][1])*v_part[i0][2] - (x_part[i0][2]+shf_h[0][2]-dtv*v_part[i0][2])*v_part[i0][1] + (x_part[i1][1]+shf_h[1][1]-dtv*v_part[i1][1])*v_part[i1][2] - (x_part[i1][2]+shf_h[1][2]-dtv*v_part[i1][2])*v_part[i1][1] + (x_part[i2][1]+shf_h[2][1]-dtv*v_part[i2][1])*v_part[i2][2] - (x_part[i2][2]+shf_h[2][2]-dtv*v_part[i2][2])*v_part[i2][1]);
                vv_ang[1] += mprt*(xp1[2]*v[i][0] - xp1[0]*v[i][2]) + mss*((x_part[i0][2]+shf_h[0][2]-dtv*v_part[i0][2])*v_part[i0][0] - (x_part[i0][0]+shf_h[0][0]-dtv*v_part[i0][0])*v_part[i0][2] + (x_part[i1][2]+shf_h[1][2]-dtv*v_part[i1][2])*v_part[i1][0] - (x_part[i1][0]+shf_h[1][0]-dtv*v_part[i1][0])*v_part[i1][2] + (x_part[i2][2]+shf_h[2][2]-dtv*v_part[i2][2])*v_part[i2][0] - (x_part[i2][0]+shf_h[2][0]-dtv*v_part[i2][0])*v_part[i2][2]); 
                vv_ang[2] += mprt*(xp1[0]*v[i][1] - xp1[1]*v[i][0]) + mss*((x_part[i0][0]+shf_h[0][0]-dtv*v_part[i0][0])*v_part[i0][1] - (x_part[i0][1]+shf_h[0][1]-dtv*v_part[i0][1])*v_part[i0][0] + (x_part[i1][0]+shf_h[1][0]-dtv*v_part[i1][0])*v_part[i1][1] - (x_part[i1][1]+shf_h[1][1]-dtv*v_part[i1][1])*v_part[i1][0] + (x_part[i2][0]+shf_h[2][0]-dtv*v_part[i2][0])*v_part[i2][1] - (x_part[i2][1]+shf_h[2][1]-dtv*v_part[i2][1])*v_part[i2][0]);*/ 

                if (ind_bounce == 1){
                  nnp[0] = a1[1]*a2[2] - a1[2]*a2[1];
                  nnp[1] = a1[2]*a2[0] - a1[0]*a2[2];
                  nnp[2] = a1[0]*a2[1] - a1[1]*a2[0];
                  uun = sqrt(nnp[0]*nnp[0] + nnp[1]*nnp[1] + nnp[2]*nnp[2]);
                  uu = 2.0 * ((v_loc[0]-vv[0])*nnp[0] + (v_loc[1]-vv[1])*nnp[1] + (v_loc[2]-vv[2])*nnp[2])/uun;
                  for (k = 0; k < 3; k++){
                    xp1[k] = x_x[k] - prd[k]*shift[m][k];        
                    v_loc[k] -= uu*nnp[k]/uun;
                    x[i][k] = xp1[k] + (1.0-tt)*dtt*v_loc[k];      // verlet 
	          }
	        }  
                else{
                  for (k = 0; k < 3; k++)
                    v_loc[k] = -v_loc[k] + 2.0*vv[k];       
                  for (k = 0; k < 3; k++){
                    xp1[k] = x_x[k] - prd[k]*shift[m][k];      
                    x[i][k] = xp1[k] + (1.0-tt)*dtt*v_loc[k];      // verlet 
	          }  
	        }
 
                for (k = 0; k < 3; k++){
                  vv[k] = 2.0*mprt*(v[i][k]-xd[k])/(mprt+3.0*mss);
                  v[i][k] -= 6.0*mss*(v[i][k]-xd[k])/(mprt+3.0*mss);
                  v_delt[i0][k] += vv[k];
                  v_delt[i1][k] += vv[k];
                  v_delt[i2][k] += vv[k];
                }

                /*for (k = 0; k < 3; k++){
                  vv_mom[k] -= mprt*v[i][k];
                  vv_mom[k] -= mss*(v_part[i0][k] + vv[k] + v_part[i1][k] + vv[k] + v_part[i2][k] + vv[k]); 
                }
                vv_ang[0] -= mprt*(x[i][1]*v[i][2] - x[i][2]*v[i][1]) + mss*((x_part[i0][1]+shf_h[0][1])*(v_part[i0][2]+vv[2]) - (x_part[i0][2]+shf_h[0][2])*(v_part[i0][1]+vv[1]) + (x_part[i1][1]+shf_h[1][1])*(v_part[i1][2]+vv[2]) - (x_part[i1][2]+shf_h[1][2])*(v_part[i1][1]+vv[1]) + (x_part[i2][1]+shf_h[2][1])*(v_part[i2][2]+vv[2]) - (x_part[i2][2]+shf_h[2][2])*(v_part[i2][1]+vv[1]));
                vv_ang[1] -= mprt*(x[i][2]*v[i][0] - x[i][0]*v[i][2]) + mss*((x_part[i0][2]+shf_h[0][2])*(v_part[i0][0]+vv[0]) - (x_part[i0][0]+shf_h[0][0])*(v_part[i0][2]+vv[2]) + (x_part[i1][2]+shf_h[1][2])*(v_part[i1][0]+vv[0]) - (x_part[i1][0]+shf_h[1][0])*(v_part[i1][2]+vv[2]) + (x_part[i2][2]+shf_h[2][2])*(v_part[i2][0]+vv[0]) - (x_part[i2][0]+shf_h[2][0])*(v_part[i2][2]+vv[2])); 
                vv_ang[2] -= mprt*(x[i][0]*v[i][1] - x[i][1]*v[i][0]) + mss*((x_part[i0][0]+shf_h[0][0])*(v_part[i0][1]+vv[1]) - (x_part[i0][1]+shf_h[0][1])*(v_part[i0][0]+vv[0]) + (x_part[i1][0]+shf_h[1][0])*(v_part[i1][1]+vv[1]) - (x_part[i1][1]+shf_h[1][1])*(v_part[i1][0]+vv[0]) + (x_part[i2][0]+shf_h[2][0])*(v_part[i2][1]+vv[1]) - (x_part[i2][1]+shf_h[2][1])*(v_part[i2][0]+vv[0])); 
                if (fabs(vv_mom[0]) > EPS || fabs(vv_mom[1]) > EPS || fabs(vv_mom[2]) > EPS || fabs(vv_ang[0]) > EPS || fabs(vv_ang[1]) > EPS || fabs(vv_ang[2]) > EPS)
                  printf("Momentum change: %15.13lf %15.13lf %15.13lf; Ang. momentum change: %15.13lf %15.13lf %15.13lf \n",vv_mom[0],vv_mom[1],vv_mom[2],vv_ang[0],vv_ang[1],vv_ang[2]);*/

                if (cell_update && mask[i] & groupbit_comm){
                  m = tag[i] - min_ind;
                  m0 = tags_part[m];
                  for (k = 0; k < 3; k++){
                    x_part[m0][k] = x[i][k];
                    v_part[m0][k] = v[i][k];         
	          } 
                  for (k = 0; k < part_degree[m]; k++)
                    if (norm_ind[part_tri[m][k]] > -1)
                      move_norm_arrays(part_tri[m][k]);
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

  if (comm->nprocs > 1) reverse_communicate(0);

  k = 0; 
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit_comm) {
      if (!(mask[i] & groupbit_no_move))
        for (j = 0; j < 3; j++){
          v[i][j] += v_delt[k][j];     
        }        
      k++;             
    }

  /*double vv_tot[3], vv_get[3], vv_ang_get[3];
  double vv_ang[3];
  for (j = 0; j < 3; j++){
    vv_tot[j] = 0.0;
    vv_get[j] = 0.0;
    vv_ang[j] = 0.0;
    vv_ang_get[j] = 0.0;
  }
  for (i = 0; i < nlocal; i++){
    domain->unmap(x[i],atom->image[i],xh); 
    vv_ang[0] += mass[type[i]]*(xh[1]*v[i][2] - xh[2]*v[i][1]);
    vv_ang[1] += mass[type[i]]*(xh[2]*v[i][0] - xh[0]*v[i][2]); 
    vv_ang[2] += mass[type[i]]*(xh[0]*v[i][1] - xh[1]*v[i][0]); 
    for (j = 0; j < 3; j++)
      vv_tot[j] += mass[type[i]]*v[i][j];
  }
  if (comm->nprocs > 1){
     MPI_Reduce(&vv_tot,&vv_get,3,MPI_DOUBLE,MPI_SUM,0,world);
     MPI_Reduce(&vv_ang,&vv_ang_get,3,MPI_DOUBLE,MPI_SUM,0,world);
  } else{
    for (j = 0; j < 3; j++){
      vv_get[j] = vv_tot[j];
      vv_ang_get[j] = vv_ang[j];
    } 
  }
  if (me == 0) printf("Momentum: %15.13lf %15.13lf %15.13lf; Ang. momentum: %15.13lf %15.13lf %15.13lf \n",vv_get[0],vv_get[1],vv_get[2],vv_ang_get[0],vv_ang_get[1],vv_ang_get[2]);*/
}

/* ---------------------------------------------------------------------- */

void FixForceBound::post_force(int vflag)
{
  int i,j,k,l,m,kk,ix,iy,iz,m0,kk1;
  double xh[3],vh[3],ff[3],xc[3],xd[3],ff1[6];
  double u0,u1,weight_h,rr,theta,theta1,hh,rr1,u_inv;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  int step_t = update->ntimestep;
  int n_stress = output->n_stress;

  if (atom->nlocal > nnmax){
    while (nnmax < atom->nlocal)               
      nnmax += BUFMIN;
    if (max_shapes) 
      memory->grow(ind_shapes,nnmax,max_shapes,"fix_force_bound:ind_shapes");
    memory->grow(tot_shapes,nnmax,"fix_force_bound:tot_shapes");
    if (max_faces)
      memory->grow(ind_faces,nnmax,max_faces,"fix_force_bound:ind_faces");
    memory->grow(num_faces,nnmax,"fix_force_bound:num_faces");
  }  
 
  if (neighbor->ago == 0){
    for (i = 0; i < norm_count; i++)
      norm_ind[norm_list[i]] = -1;
    norm_count = 0;  
    shape_decide();
    local_faces(0);
    if (comm->nprocs > 1) borders();
    bin_vertices();
    face_decide();
  }  

  n_accum++;
  if (ind_shear || ind_press || ind_force)
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit_s || mask[i] & groupbit_p || mask[i] & groupbit_force)
        for (m = 0; m < tot_shapes[i]; m++){
          kk = ind_shapes[i][m];
          kk1 = shapes_global_to_local[kk];
          for (j = 0; j < 3; j++){
            xh[j] = x[i][j] - x0[kk][j];
            vh[j] = v[i][j];
	  }
          rot_forward(xh[0],xh[1],xh[2],kk);
          if (cur_iter)
            rot_forward(vh[0],vh[1],vh[2],kk);  
          switch (ptype[kk]){
            case 1 : 
              u1 = xh[1]/aa[kk][2]; 
              u0 = (xh[0]-u1*aa[kk][1])/aa[kk][0];
              if (u0 >= 0.0 && u1 >= 0.0 && u0+u1 <= 1.0){
                if (ind_shear && (mask[i] & groupbit_s)){
                  if (xh[2]>0.0 && xh[2]<r_shear && (s_apply == 1 || s_apply == 0)){
                    if (cur_iter){ 
                      iz = static_cast<int> (xh[2]*n_per/r_shear);
                      velx[kk1][iz][0][0] += vh[0];
                      vely[kk1][iz][0][0] += vh[1];
                      velz[kk1][iz][0][0] += vh[2];
                      num[kk1][iz][0][0] += 1.0;
                    }
                    k = static_cast<int> (xh[2]*dr_inv);
                    weight_h = (weight[k+1]-weight[k])*(xh[2]*dr_inv - k) + weight[k];
                    ff[0] = fsx[kk1][0][0][0]*weight_h;
                    ff[1] = fsy[kk1][0][0][0]*weight_h;
                    ff[2] = 0.0;   
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] -= ff[j]; 
		  } 
                  if (xh[2]>-r_shear && xh[2]<0.0 && (s_apply == 2 || s_apply == 0)){
                    if (cur_iter){
                      iz = static_cast<int> (-xh[2]*n_per/r_shear);
                      velx[kk1][iz+n_per][0][0] += vh[0];
                      vely[kk1][iz+n_per][0][0] += vh[1];
                      velz[kk1][iz+n_per][0][0] += vh[2];
                      num[kk1][iz+n_per][0][0] += 1.0;
                    }
                    k = static_cast<int> (-xh[2]*dr_inv);
                    weight_h = (weight[k+1]-weight[k])*(-xh[2]*dr_inv - k) + weight[k];
                    ff[0] = fsx[kk1][1][0][0]*weight_h;
                    ff[1] = fsy[kk1][1][0][0]*weight_h;
                    ff[2] = 0.0;   
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] -= ff[j]; 
		  }
	        }
                if (ind_press && (mask[i] & groupbit_p)){
                  if (xh[2]>0.0 && xh[2]<r_press && (p_apply == 1 || p_apply == 0)){
                    iz = static_cast<int> (xh[2]*n_press/r_press);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = f_press[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (xh[2]>-r_press && xh[2]<0.0 && (p_apply == 2 || p_apply == 0)){
                    iz = static_cast<int> (-xh[2]*n_press/r_press);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = -f_press[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  }
	        }
                if (ind_force && (mask[i] & groupbit_force)){
                  if (xh[2]>0.0 && xh[2]<r_force && (f_apply == 1 || f_apply == 0)){
                    iz = static_cast<int> (xh[2]*n_force/r_force);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = f_force[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (xh[2]>-r_force && xh[2]<0.0 && (f_apply == 2 || f_apply == 0)){
                    iz = static_cast<int> (-xh[2]*n_force/r_force);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = -f_force[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  }
	        }
	      }
              break;
            case 2 :
              u1 = xh[1]/aa[kk][2]; 
              u0 = (xh[0]-u1*aa[kk][1])/aa[kk][0];
              if (u0 >= 0.0 && u1 >= 0.0 && u0 < 1.0 && u1 < 1.0){
                ix = static_cast<int> (u0*ndiv[kk][0]);
                iy = static_cast<int> (u1*ndiv[kk][1]);
                if (ind_shear && (mask[i] & groupbit_s)){ 
                  if (xh[2]>0.0 && xh[2]<r_shear && (s_apply == 1 || s_apply == 0)){
                    if (cur_iter){
                      iz = static_cast<int> (xh[2]*n_per/r_shear);
                      velx[kk1][iz][ix][iy] += vh[0];
                      vely[kk1][iz][ix][iy] += vh[1];
                      velz[kk1][iz][ix][iy] += vh[2];
                      num[kk1][iz][ix][iy] += 1.0;
                    }
                    k = static_cast<int> (xh[2]*dr_inv);
                    weight_h = (weight[k+1]-weight[k])*(xh[2]*dr_inv - k) + weight[k];
                    ff[0] = fsx[kk1][0][ix][iy]*weight_h;
                    ff[1] = fsy[kk1][0][ix][iy]*weight_h;
                    ff[2] = 0.0;   
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] -= ff[j]; 
		  } 
                  if (xh[2]>-r_shear && xh[2]<0.0 && (s_apply == 2 || s_apply == 0)){
                    if (cur_iter){
                      iz = static_cast<int> (-xh[2]*n_per/r_shear);
                      velx[kk1][iz+n_per][ix][iy] += vh[0];
                      vely[kk1][iz+n_per][ix][iy] += vh[1];
                      velz[kk1][iz+n_per][ix][iy] += vh[2];
                      num[kk1][iz+n_per][ix][iy] += 1.0;
                    }
                    k = static_cast<int> (-xh[2]*dr_inv);
                    weight_h = (weight[k+1]-weight[k])*(-xh[2]*dr_inv - k) + weight[k];
                    ff[0] = fsx[kk1][1][ix][iy]*weight_h;
                    ff[1] = fsy[kk1][1][ix][iy]*weight_h;
                    ff[2] = 0.0;   
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] -= ff[j]; 
		  }
	        }
                if (ind_press && (mask[i] & groupbit_p)){
                  if (xh[2]>0.0 && xh[2]<r_press && (p_apply == 1 || p_apply == 0)){
                    iz = static_cast<int> (xh[2]*n_press/r_press);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = f_press[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (xh[2]>-r_press && xh[2]<0.0 && (p_apply == 2 || p_apply == 0)){
                    iz = static_cast<int> (-xh[2]*n_press/r_press);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = -f_press[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  }
	        }
                if (ind_force && (mask[i] & groupbit_force)){
                  if (xh[2]>0.0 && xh[2]<r_force && (f_apply == 1 || f_apply == 0)){
                    iz = static_cast<int> (xh[2]*n_force/r_force);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = f_force[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (xh[2]>-r_force && xh[2]<0.0 && (f_apply == 2 || f_apply == 0)){
                    iz = static_cast<int> (-xh[2]*n_force/r_force);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = -f_force[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  }
	        }
	      }
              break;
            case 3 :
              if (xh[2]>=0.0 && xh[2] < aa[kk][0]){
                rr = sqrt(xh[0]*xh[0] + xh[1]*xh[1]);
                ix = static_cast<int> (xh[2]*ndiv[kk][0]/aa[kk][0]); 
                theta = acos(xh[0]/rr);
                theta1 = asin(xh[1]/rr);
                if (theta1 < 0.0)
                  theta = 2.0*M_PI - theta;
                iy = static_cast<int> (0.5*theta*ndiv[kk][1]/M_PI);
                if (iy > ndiv[kk][1]-1)
                  iy = ndiv[kk][1]-1;
                hh = rr - aa[kk][1];
                if (ind_shear && (mask[i] & groupbit_s)){
                  if (hh>0.0 && hh<r_shear && (s_apply == 1 || s_apply == 0)){
                    if (cur_iter){
                      iz = static_cast<int> (hh*n_per/r_shear);
                      velx[kk1][iz][ix][iy] += vh[0];
                      vely[kk1][iz][ix][iy] += vh[1];
                      velz[kk1][iz][ix][iy] += vh[2];
                      num[kk1][iz][ix][iy] += 1.0;
                    }
                    k = static_cast<int> (hh*dr_inv);
                    weight_h = (weight[k+1]-weight[k])*(hh*dr_inv - k) + weight[k];
                    ff[0] = fsx[kk1][0][ix][iy]*weight_h;
                    ff[1] = fsy[kk1][0][ix][iy]*weight_h;
                    ff[2] = fsz[kk1][0][ix][iy]*weight_h;   
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] -= ff[j]; 
		  } 
                  if (hh>-r_shear && hh<0.0 && (s_apply == 2 || s_apply == 0)){
                    if (cur_iter){
                      iz = static_cast<int> (-hh*n_per/r_shear);
                      velx[kk1][iz+n_per][ix][iy] += vh[0];
                      vely[kk1][iz+n_per][ix][iy] += vh[1];
                      velz[kk1][iz+n_per][ix][iy] += vh[2];
                      num[kk1][iz+n_per][ix][iy] += 1.0;
                    } 
                    k = static_cast<int> (-hh*dr_inv);
                    weight_h = (weight[k+1]-weight[k])*(-hh*dr_inv - k) + weight[k];
                    ff[0] = fsx[kk1][1][ix][iy]*weight_h;
                    ff[1] = fsy[kk1][1][ix][iy]*weight_h;
                    ff[2] = fsz[kk1][1][ix][iy]*weight_h;   
                    rot_back(ff[0],ff[1],ff[2],kk);
                    //ff[1] = 0.0;
                    //ff[2] = 0.0;
                    for (j = 0; j < 3; j++)
                      f[i][j] -= ff[j]; 
		  }
	        }
                if (ind_press && (mask[i] & groupbit_p)){
                  if (hh>0.0 && hh<r_press && (p_apply == 1 || p_apply == 0)){
                    iz = static_cast<int> (hh*n_press/r_press);
                    ff[0] = f_press[iz]*xh[0]/rr;
                    ff[1] = f_press[iz]*xh[1]/rr;
                    ff[2] = 0.0;
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (hh>-r_press && hh<0.0 && (p_apply == 2 || p_apply == 0)){
                    iz = static_cast<int> (-hh*n_press/r_press);
                    ff[0] = -f_press[iz]*xh[0]/rr;
                    ff[1] = -f_press[iz]*xh[1]/rr;
                    ff[2] = 0.0;
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  }
	        }
                if (ind_force && (mask[i] & groupbit_force)){
                  if (hh>0.0 && hh<r_force && (f_apply == 1 || f_apply == 0)){
                    iz = static_cast<int> (hh*n_force/r_force);
                    ff[0] = f_force[iz]*xh[0]/rr;
                    ff[1] = f_force[iz]*xh[1]/rr;
                    ff[2] = 0.0;
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (hh>-r_force && hh<0.0 && (f_apply == 2 || f_apply == 0)){
                    iz = static_cast<int> (-hh*n_force/r_force);
                    ff[0] = -f_force[iz]*xh[0]/rr;
                    ff[1] = -f_force[iz]*xh[1]/rr;
                    ff[2] = 0.0;
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for (j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  }
	        }
	      }   
              break;
            case 4 :
              rr = sqrt(xh[0]*xh[0] + xh[1]*xh[1] + xh[2]*xh[2]);
              rr1 = sqrt(xh[0]*xh[0] + xh[1]*xh[1]);
              theta = acos(xh[2]/rr);
              ix = static_cast<int> (theta*ndiv[kk][0]/M_PI);
              if (ix > ndiv[kk][0]-1)
                ix = ndiv[kk][0]-1;
              theta = acos(xh[0]/rr1);
              theta1 = asin(xh[1]/rr1);
              if (theta1 < 0.0)
                theta = 2.0*M_PI - theta;
              iy = static_cast<int> (0.5*theta*ndiv[kk][1]/M_PI);
              if (iy > ndiv[kk][1]-1)
                iy = ndiv[kk][1]-1;
              hh = rr - aa[kk][0];
              if (ind_shear && (mask[i] & groupbit_s)){
                if (hh>0.0 && hh<r_shear && (s_apply == 1 || s_apply == 0)){
                  if (cur_iter){
                    iz = static_cast<int> (hh*n_per/r_shear);
                    velx[kk1][iz][ix][iy] += vh[0];
                    vely[kk1][iz][ix][iy] += vh[1];
                    velz[kk1][iz][ix][iy] += vh[2];
                    num[kk1][iz][ix][iy] += 1.0;
                  }
                  k = static_cast<int> (hh*dr_inv);
                  weight_h = (weight[k+1]-weight[k])*(hh*dr_inv - k) + weight[k];
                  ff[0] = fsx[kk1][0][ix][iy]*weight_h;
                  ff[1] = fsy[kk1][0][ix][iy]*weight_h;
                  ff[2] = fsz[kk1][0][ix][iy]*weight_h;   
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for (j = 0; j < 3; j++)
                    f[i][j] -= ff[j]; 
	        } 
                if (hh>-r_shear && hh<0.0 && (s_apply == 2 || s_apply == 0)){
                  if (cur_iter){
                    iz = static_cast<int> (-hh*n_per/r_shear);
                    velx[kk1][iz+n_per][ix][iy] += vh[0];
                    vely[kk1][iz+n_per][ix][iy] += vh[1];
                    velz[kk1][iz+n_per][ix][iy] += vh[2];
                    num[kk1][iz+n_per][ix][iy] += 1.0;
                  }
                  k = static_cast<int> (-hh*dr_inv);
                  weight_h = (weight[k+1]-weight[k])*(-hh*dr_inv - k) + weight[k];
                  ff[0] = fsx[kk1][1][ix][iy]*weight_h;
                  ff[1] = fsy[kk1][1][ix][iy]*weight_h;
                  ff[2] = fsz[kk1][1][ix][iy]*weight_h;   
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for (j = 0; j < 3; j++)
                    f[i][j] -= ff[j]; 
	        }
	      }
              if (ind_press && (mask[i] & groupbit_p)){
                if (hh>0.0 && hh<r_press && (p_apply == 1 || p_apply == 0)){
                  iz = static_cast<int> (hh*n_press/r_press);
                  ff[0] = f_press[iz]*xh[0]/rr;
                  ff[1] = f_press[iz]*xh[1]/rr;
                  ff[2] = f_press[iz]*xh[2]/rr;
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for (j = 0; j < 3; j++)
                    f[i][j] += ff[j]; 
	        } 
                if (hh>-r_press && hh<0.0 && (p_apply == 2 || p_apply == 0)){
                  iz = static_cast<int> (-hh*n_press/r_press);
                  ff[0] = -f_press[iz]*xh[0]/rr;
                  ff[1] = -f_press[iz]*xh[1]/rr;
                  ff[2] = -f_press[iz]*xh[2]/rr;
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for (j = 0; j < 3; j++)
                    f[i][j] += ff[j]; 
	        }
	      } 
              if (ind_force && (mask[i] & groupbit_force)){
                if (hh>0.0 && hh<r_force && (f_apply == 1 || f_apply == 0)){
                  iz = static_cast<int> (hh*n_force/r_force);
                  ff[0] = f_force[iz]*xh[0]/rr;
                  ff[1] = f_force[iz]*xh[1]/rr;
                  ff[2] = f_force[iz]*xh[2]/rr;
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for (j = 0; j < 3; j++)
                    f[i][j] += ff[j]; 
	        } 
                if (hh>-r_force && hh<0.0 && (f_apply == 2 || f_apply == 0)){
                  iz = static_cast<int> (-hh*n_force/r_force);
                  ff[0] = -f_force[iz]*xh[0]/rr;
                  ff[1] = -f_force[iz]*xh[1]/rr;
                  ff[2] = -f_force[iz]*xh[2]/rr;
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for (j = 0; j < 3; j++)
                    f[i][j] += ff[j]; 
	        }
	      }          
              break;
	  }       
	}

  if (ind_press_cell || ind_force_cell){
    for (i = 0; i < npart; i++)
      for (j = 0; j < 3; j++)
        f_part[i][j] = 0.0;      
     
    for (i = 0; i < nlocal; i++){
      if (ind_press_cell && mask[i] & groupbit_p_cell)
        for (m = 0; m < num_faces[i]; m++){
          m0 = ind_faces[i][m];
          if (norm_ind[m0] == -1) calc_norm(m0);
          k = norm_ind[m0];
          rr1 = sqrt(norm[k][0]*norm[k][0] + norm[k][1]*norm[k][1] + norm[k][2]*norm[k][2]);
          for (j = 0; j < 3; j++)
            norm[k][j] /= rr1;
          u0 = - norm[k][0]*x[i][0] - norm[k][1]*x[i][1] - norm[k][2]*x[i][2];
          ix = tags_part[tri_tags[m0][0]];
          u1 = - norm[k][0]*x_part[ix][0] - norm[k][1]*x_part[ix][1] - norm[k][2]*x_part[ix][2]; 
          rr = u1 - u0;
          hh = fabs(rr);
          if (hh < r_press){
            iy = tags_part[tri_tags[m0][1]];
            iz = tags_part[tri_tags[m0][2]];
            kk = static_cast<int> (hh*n_press/r_press); 
            for (j = 0; j < 3; j++){
              xh[j] = x[i][j] - rr*norm[k][j];
              xc[j] = xh[j] - x_part[ix][j]; 
	    }
            xd[0] = edge1[k][0]*xc[0] + edge1[k][1]*xc[1] + edge1[k][2]*xc[2];
            xd[1] = edge2[k][0]*xc[0] + edge2[k][1]*xc[1] + edge2[k][2]*xc[2];
            vh[0] = edge1[k][0]*edge1[k][0] + edge1[k][1]*edge1[k][1] + edge1[k][2]*edge1[k][2];
            vh[1] = edge2[k][0]*edge2[k][0] + edge2[k][1]*edge2[k][1] + edge2[k][2]*edge2[k][2];
            vh[2] = edge1[k][0]*edge2[k][0] + edge1[k][1]*edge2[k][1] + edge1[k][2]*edge2[k][2]; 
            u_inv = vh[0]*vh[1] - vh[2]*vh[2];
            u0 = (xd[0]*vh[1] - xd[1]*vh[2])/u_inv;
            u1 = (xd[1]*vh[0] - xd[0]*vh[2])/u_inv;  
            if (u0 >= 0.0 && u1 >= 0.0 && u0+u1 <= 1.0){
              if (rr < 0.0 && mask[i] & groupbit_inner){
                for (j = 0; j < 3; j++){
                  f[i][j] -= f_press[kk]*norm[k][j];
		  f_part[ix][j] += f_press[kk]*norm[k][j]/3.0;     
                  f_part[iy][j] += f_press[kk]*norm[k][j]/3.0; 
                  f_part[iz][j] += f_press[kk]*norm[k][j]/3.0; 
		}
                if (n_stress){
                  for (j = 0; j < 3; j++){
                    xd[j] = x_part[ix][j] + (edge1[k][j] + edge2[k][j])/3.0;
                    vh[j] = x[i][j] - xd[j]; 
		  }
                  ff1[0] = - f_press[kk]*norm[k][0]*vh[0];
                  ff1[1] = - f_press[kk]*norm[k][1]*vh[1];
                  ff1[2] = - f_press[kk]*norm[k][2]*vh[2];
                  ff1[3] = - f_press[kk]*norm[k][0]*vh[1];
                  ff1[4] = - f_press[kk]*norm[k][0]*vh[2];
                  ff1[5] = - f_press[kk]*norm[k][1]*vh[2]; 
                  for (j = 0; j < n_stress; j++){
                    l = output->stress_id[j];
                    if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
                      output->stat[l]->virial6(i,xd,ff1);
                  }
		}
	      }
              if (rr > 0.0 && !(mask[i] & groupbit_inner)){
                for (j = 0; j < 3; j++){
                  f[i][j] += f_press[kk]*norm[k][j];
		  f_part[ix][j] -= f_press[kk]*norm[k][j]/3.0;     
                  f_part[iy][j] -= f_press[kk]*norm[k][j]/3.0; 
                  f_part[iz][j] -= f_press[kk]*norm[k][j]/3.0; 
		}
                if (n_stress){
                  for (j = 0; j < 3; j++){
                    xd[j] = x_part[ix][j] + (edge1[k][j] + edge2[k][j])/3.0;
                    vh[j] = x[i][j] - xd[j];
                  }
                  ff1[0] = - f_press[kk]*norm[k][0]*vh[0];
                  ff1[1] = - f_press[kk]*norm[k][1]*vh[1];
                  ff1[2] = - f_press[kk]*norm[k][2]*vh[2];
                  ff1[3] = - f_press[kk]*norm[k][0]*vh[1];
                  ff1[4] = - f_press[kk]*norm[k][0]*vh[2];
                  ff1[5] = - f_press[kk]*norm[k][1]*vh[2];
                  for (j = 0; j < n_stress; j++){
                    l = output->stress_id[j];
                    if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
                      output->stat[l]->virial6(i,xd,ff1);
                  }
                }
	      }
	    }
	  }
	}

      if (ind_force_cell && mask[i] & groupbit_force_cell)
        for (m = 0; m < num_faces[i]; m++){
          m0 = ind_faces[i][m];
          if (norm_ind[m0] == -1) calc_norm(m0);
          k = norm_ind[m0];
          rr1 = sqrt(norm[k][0]*norm[k][0] + norm[k][1]*norm[k][1] + norm[k][2]*norm[k][2]);
          for (j = 0; j < 3; j++)
            norm[k][j] /= rr1;
          u0 = - norm[k][0]*x[i][0] - norm[k][1]*x[i][1] - norm[k][2]*x[i][2];
          ix = tags_part[tri_tags[m0][0]];
          u1 = - norm[k][0]*x_part[ix][0] - norm[k][1]*x_part[ix][1] - norm[k][2]*x_part[ix][2]; 
          rr = u1 - u0;
          hh = fabs(rr);
          if (rr > 0.0 && rr < r_force ){
            iy = tags_part[tri_tags[m0][1]];
            iz = tags_part[tri_tags[m0][2]];
            kk = static_cast<int> (hh*n_force/r_force); 
            for (j = 0; j < 3; j++){
              xh[j] = x[i][j] - rr*norm[k][j];
              xc[j] = xh[j] - x_part[ix][j]; 
	    }
            xd[0] = edge1[k][0]*xc[0] + edge1[k][1]*xc[1] + edge1[k][2]*xc[2];
            xd[1] = edge2[k][0]*xc[0] + edge2[k][1]*xc[1] + edge2[k][2]*xc[2];
            vh[0] = edge1[k][0]*edge1[k][0] + edge1[k][1]*edge1[k][1] + edge1[k][2]*edge1[k][2];
            vh[1] = edge2[k][0]*edge2[k][0] + edge2[k][1]*edge2[k][1] + edge2[k][2]*edge2[k][2];
            vh[2] = edge1[k][0]*edge2[k][0] + edge1[k][1]*edge2[k][1] + edge1[k][2]*edge2[k][2]; 
            u_inv = vh[0]*vh[1] - vh[2]*vh[2];
            u0 = (xd[0]*vh[1] - xd[1]*vh[2])/u_inv;
            u1 = (xd[1]*vh[0] - xd[0]*vh[2])/u_inv;  
            if (u0 >= 0.0 && u1 >= 0.0 && u0+u1 <= 1.0){
              for (j = 0; j < 3; j++){
                f[i][j] += f_force[kk]*norm[k][j];
		f_part[ix][j] -= f_force[kk]*norm[k][j]/3.0;     
                f_part[iy][j] -= f_force[kk]*norm[k][j]/3.0; 
                f_part[iz][j] -= f_force[kk]*norm[k][j]/3.0; 
	      }
              
              if (n_stress){
                for (j = 0; j < 3; j++){
                  xd[j] = x_part[ix][j] + (edge1[k][j] + edge2[k][j])/3.0;
                  vh[j] = x[i][j] - xd[j];
                }
                ff1[0] = - f_force[kk]*norm[k][0]*vh[0];
                ff1[1] = - f_force[kk]*norm[k][1]*vh[1];
                ff1[2] = - f_force[kk]*norm[k][2]*vh[2];
                ff1[3] = - f_force[kk]*norm[k][0]*vh[1];
                ff1[4] = - f_force[kk]*norm[k][0]*vh[2];
                ff1[5] = - f_force[kk]*norm[k][1]*vh[2];
                for (j = 0; j < n_stress; j++){
                  l = output->stress_id[j];
                  if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
                    output->stat[l]->virial6(i,xd,ff1);
                }
              } 
	    }
	  }
	}
    }

    if (comm->nprocs > 1) reverse_communicate(1);

    k = 0; 
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit_comm) {
        for (j = 0; j < 3; j++)
          f[i][j] += f_part[k][j];     
        k++;             
      }
  }
  
  for (i = 0; i < norm_count; i++)
    norm_ind[norm_list[i]] = -1;
  norm_count = 0;  

  if (ind_shear && num_shapes_tot)
    if (cur_iter && step_t%iter == 0 && n_accum > 0.6*iter) 
      recalc_force();
}

/* ---------------------------------------------------------------------- */

int FixForceBound::check_shapes(int sh)
{
  int i, ch_ind, nn[3], jj[3];
  double rad, rad_n, rad_t, u0, u1, u2, zz, dl;
  double dd[3], ddx[3];

  for (i = 0; i < 3; ++i){                  // setup bins
    ddx[i] = domain->subhi[i] - domain->sublo[i];
    nn[i] = static_cast<int> (ddx[i]/binsize + 3.0);
  }
  rad = 0.5*sqrt(3.0)*binsize;
  rad_n = rad + r_cut_n; 
  rad_t = rad + r_cut_t;

  ch_ind = 0;
  for (jj[0] = 0; jj[0] < nn[0]; ++jj[0]){             // check if certain bins overlap with a core
    for (jj[1] = 0; jj[1] < nn[1]; ++jj[1]){
      for (jj[2] = 0; jj[2] < nn[2]; ++jj[2]){
        for (i = 0; i < 3; ++i)
          dd[i] = domain->sublo[i] + (jj[i]-0.5)*binsize - x0[sh][i];
        rot_forward(dd[0],dd[1],dd[2],sh);

        switch (ptype[sh]){
          case 1 :
            if (dd[0] > aa[sh][4]-rad_t && dd[0] < aa[sh][5]+rad_t && dd[1] > -rad_t && dd[1] < aa[sh][2]+rad_t && dd[2] > -rad_n && dd[2] < rad_n){
              u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
              zz = rad_t*aa[sh][3]/aa[sh][2];
              dl = rad_t*aa[sh][3]/(aa[sh][6]*aa[sh][1]+aa[sh][7]*aa[sh][2]);                             
              u2 = dd[1]/(aa[sh][2] + dl*aa[sh][2]/aa[sh][3]);
              u1 = (dd[0] - u2*(aa[sh][1] + dl*aa[sh][1]/aa[sh][3]))/(aa[sh][0] + rad_t/aa[sh][6]);
              if (u0 > -zz && u1+u2 < 1.0)
                ch_ind = 1;
	    }
            break;
          case 2 :
            if (dd[0] > aa[sh][4]-rad_t && dd[0] < aa[sh][5]+rad_t && dd[1] > -rad_t && dd[1] < aa[sh][2]+rad_t && dd[2] > -rad_n && dd[2] < rad_n){
              u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
              zz = rad_t*aa[sh][3]/aa[sh][2];
              if (u0 > -zz && u0 < aa[sh][0] + zz)
                ch_ind = 1;
	    }
            break;
          case 3 :
            zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
            if (zz > aa[sh][1]-rad_n && zz < aa[sh][1]+rad_n && dd[2] > -rad_t && dd[2] < aa[sh][0]+rad_t)
              ch_ind = 1;
            break;
          case 4 :
            zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
            if (zz > aa[sh][0]-rad_n && zz < aa[sh][0]+rad_n)
              ch_ind = 1;
            break;
        }
        if (ch_ind) break;
      }
      if (ch_ind) break;
    }
    if (ch_ind) break;
  }

  return ch_ind;
}

/* ---------------------------------------------------------------------- */

void FixForceBound::grow_shape_arrays()
{
  int k, l, m;
  int **ind_tmp;

  if (max_shapes == 0){
    max_shapes = FACE_INC;
    memory->create(ind_shapes,nnmax,max_shapes,"fix_force_bound:ind_shapes");
  } else{
    m = max_shapes;
    memory->create(ind_tmp,nnmax,m,"fix_force_bound:ind_tmp");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_tmp[k][l] = ind_shapes[k][l];
    max_shapes += FACE_INC;
    memory->destroy(ind_shapes);
    memory->create(ind_shapes,nnmax,max_shapes,"fix_force_bound:ind_shapes");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_shapes[k][l] = ind_tmp[k][l];
    memory->destroy(ind_tmp);
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound::setup_rot(int id, double a1[], double a2[])
{
  int i,j;
  double norm_h[3],mr[3][3],a3[2];
  double nr,rx,zz;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++){
      rot[id][i][j] = 0.0;
      mr[i][j] = 0.0;  
    }

  if (ptype[id] < 3){  
    norm_h[0] = a1[1]*a2[2] - a1[2]*a2[1];
    norm_h[1] = a1[2]*a2[0] - a1[0]*a2[2];
    norm_h[2] = a1[0]*a2[1] - a1[1]*a2[0]; 
    nr = sqrt(norm_h[0]*norm_h[0] + norm_h[1]*norm_h[1] + norm_h[2]*norm_h[2]);
    rx = sqrt(norm_h[0]*norm_h[0] + norm_h[1]*norm_h[1]);
    if (rx > EPS){ 
      mr[0][0] = norm_h[0]*norm_h[2]/nr/rx;
      mr[0][1] = norm_h[1]*norm_h[2]/nr/rx;
      mr[0][2] = -rx/nr;
      mr[1][0] = -norm_h[1]/rx;
      mr[1][1] = norm_h[0]/rx;
      mr[1][2] = 0.0;
      mr[2][0] = norm_h[0]/nr;
      mr[2][1] = norm_h[1]/nr;
      mr[2][2] = norm_h[2]/nr;
    }
    else {
      mr[0][0] = 1.0;
      mr[1][1] = 1.0;
      mr[2][2] = 1.0;
      if (norm_h[2] < 0.0)
        mr[2][2] = -1.0;
    }
    a3[0] = mr[0][0]*a1[0] + mr[0][1]*a1[1] + mr[0][2]*a1[2];
    a3[1] = mr[1][0]*a1[0] + mr[1][1]*a1[1] + mr[1][2]*a1[2];
    rx = sqrt(a3[0]*a3[0] + a3[1]*a3[1]);
    rot[id][0][0] = (a3[0]*mr[0][0] + a3[1]*mr[1][0])/rx;
    rot[id][0][1] = (a3[0]*mr[0][1] + a3[1]*mr[1][1])/rx;
    rot[id][0][2] = (a3[0]*mr[0][2] + a3[1]*mr[1][2])/rx;
    rot[id][1][0] = (-a3[1]*mr[0][0] + a3[0]*mr[1][0])/rx;
    rot[id][1][1] = (-a3[1]*mr[0][1] + a3[0]*mr[1][1])/rx;
    rot[id][1][2] = (-a3[1]*mr[0][2] + a3[0]*mr[1][2])/rx;
    rot[id][2][0] = mr[2][0];
    rot[id][2][1] = mr[2][1];
    rot[id][2][2] = mr[2][2];
    aa[id][0] = rx;
    aa[id][1] = rot[id][0][0]*a2[0] + rot[id][0][1]*a2[1] + rot[id][0][2]*a2[2];
    aa[id][2] = rot[id][1][0]*a2[0] + rot[id][1][1]*a2[1] + rot[id][1][2]*a2[2];
    aa[id][3] = sqrt(aa[id][1]*aa[id][1] + aa[id][2]*aa[id][2]);
    aa[id][4] = MIN(0.0,aa[id][1]);
    if (ptype[id] == 1){
      aa[id][5] = MAX(rx,aa[id][1]);
      zz = sqrt(aa[id][2]*aa[id][2] + (rx - aa[id][1])*(rx - aa[id][1]));
      aa[id][6] = aa[id][2]/zz;
      aa[id][7] = (aa[id][0] - aa[id][1])/zz;
    } else{
      aa[id][5] = MAX(rx,rx + aa[id][1]);
    }
  } else if (ptype[id] == 3){
    norm_h[0] = a1[0];
    norm_h[1] = a1[1];
    norm_h[2] = a1[2];
    nr = sqrt(norm_h[0]*norm_h[0] + norm_h[1]*norm_h[1] + norm_h[2]*norm_h[2]);
    rx = sqrt(norm_h[0]*norm_h[0] + norm_h[1]*norm_h[1]);
    if (rx > EPS){ 
      rot[id][0][0] = norm_h[0]*norm_h[2]/nr/rx;
      rot[id][0][1] = norm_h[1]*norm_h[2]/nr/rx;
      rot[id][0][2] = -rx/nr;
      rot[id][1][0] = -norm_h[1]/rx;
      rot[id][1][1] = norm_h[0]/rx;
      rot[id][1][2] = 0.0;
      rot[id][2][0] = norm_h[0]/nr;
      rot[id][2][1] = norm_h[1]/nr;
      rot[id][2][2] = norm_h[2]/nr;
    } else {
      rot[id][0][0] = 1.0;
      rot[id][1][1] = 1.0;
      rot[id][2][2] = 1.0;
      if (norm_h[2] < 0.0)
        rot[id][2][2] = -1.0;
    }
    aa[id][0] = nr;
  } else {
    for (i = 0; i < 3; i++)
      rot[id][i][i] = 1.0; 
  }
} 

/* ---------------------------------------------------------------------- */

void FixForceBound::rot_forward(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][0][1]*y1 + rot[id][0][2]*z1;
  y = rot[id][1][0]*x1 + rot[id][1][1]*y1 + rot[id][1][2]*z1;
  z = rot[id][2][0]*x1 + rot[id][2][1]*y1 + rot[id][2][2]*z1;
}

/* ---------------------------------------------------------------------- */

void FixForceBound::rot_back(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][1][0]*y1 + rot[id][2][0]*z1;
  y = rot[id][0][1]*x1 + rot[id][1][1]*y1 + rot[id][2][1]*z1;
  z = rot[id][0][2]*x1 + rot[id][1][2]*y1 + rot[id][2][2]*z1;
}

/* ---------------------------------------------------------------------- */

void FixForceBound::shape_decide()
{
  int i,sh,k,l,m,kk,bin;
  double dd[3];
  double u0,u1,u2,zz,dl;

  double **x = atom->x;
  int *mask = atom->mask;

  for (i=0; i<atom->nlocal; i++){
    tot_shapes[i] = 0;
    if (mask[i] & groupbit_solid){
      bin = coord2bin(x[i]); 
      if (bin == -1){
        fprintf(stderr, "Negative bin in shape_decide: me=%d, step=" BIGINT_FORMAT ", particle - %f %f %f; box - %f %f %f %f %f %f \n",me,update->ntimestep,x[i][0],x[i][1],x[i][2],bboxlo[0],bboxlo[1],bboxlo[2],bboxhi[0],bboxhi[1],bboxhi[2]);
      }else{
        for (kk = 0; kk < num_bin_shapes[bin]; kk++){
          sh = bin_shapes[kk][bin];
          for (k = 0; k < 3; k++)
            dd[k] = x[i][k] - x0[sh][k];
          rot_forward(dd[0],dd[1],dd[2],sh);
          switch (ptype[sh]){
            case 1 :
              if (dd[0] > aa[sh][4]-r_cut_t && dd[0] < aa[sh][5]+r_cut_t && dd[1] > -r_cut_t && dd[1] < aa[sh][2]+r_cut_t && dd[2] > -r_cut_n && dd[2] < r_cut_n){
                u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
                zz = r_cut_t*aa[sh][3]/aa[sh][2];
                dl = r_cut_t*aa[sh][3]/(aa[sh][6]*aa[sh][1]+aa[sh][7]*aa[sh][2]);
                u2 = dd[1]/(aa[sh][2] + dl*aa[sh][2]/aa[sh][3]);
                u1 = (dd[0] - u2*(aa[sh][1] + dl*aa[sh][1]/aa[sh][3]))/(aa[sh][0] + r_cut_t/aa[sh][6]);
                if (u0 > -zz && u1+u2 < 1.0){
                  if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                  ind_shapes[i][tot_shapes[i]] = sh;
                  tot_shapes[i]++;
                }
	      }
              break;
            case 2 :
              if (dd[0] > aa[sh][4]-r_cut_t && dd[0] < aa[sh][5]+r_cut_t && dd[1] > -r_cut_t && dd[1] < aa[sh][2]+r_cut_t && dd[2] > -r_cut_n && dd[2] < r_cut_n){
                u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
                zz = r_cut_t*aa[sh][3]/aa[sh][2];
                if (u0 > -zz && u0 < aa[sh][0] + zz){
                  if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                  ind_shapes[i][tot_shapes[i]] = sh;
                  tot_shapes[i]++;
                }
	      }
              break;
            case 3 :
              zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
              if (zz > aa[sh][1]-r_cut_n && zz < aa[sh][1]+r_cut_n && dd[2] > -r_cut_t && dd[2] < aa[sh][0]+r_cut_t){
                if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                ind_shapes[i][tot_shapes[i]] = sh;
                tot_shapes[i]++;
              }
              break;
            case 4 :
              zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
              if (zz > aa[sh][0]-r_cut_n && zz < aa[sh][0]+r_cut_n){
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

void FixForceBound::setup_bins()
{
  // bbox = size of the bounding box
  // bbox lo/hi = bounding box of my subdomain extended by r_cut_n

  int i, ii, jj, kk, nn;
  int ix, iy, iz, bin, h_bin;
  double bbox[3];
  double r_extent = r_cut_n;

  for (i = 0; i < 3; i++){
    bboxlo[i] = domain->sublo[i] - r_extent;
    bboxhi[i] = domain->subhi[i] + r_extent;
    bbox[i] = bboxhi[i] - bboxlo[i];
  }

  nbinx = static_cast<int> (bbox[0]*binsizeinv + 1.0);
  nbiny = static_cast<int> (bbox[1]*binsizeinv + 1.0);
  if (domain->dimension == 3) nbinz = static_cast<int> (bbox[2]*binsizeinv + 1.0);
  else {
    nbinz = 1;
    bboxlo[2] = - 0.5*binsize + EPS;
    bboxhi[2] = 0.5*binsize - EPS;
  }
  nbin = nbinx*nbiny*nbinz;
  
  memory->create(bin_shapes,bin_max_shape, nbin, "fix_force_bound:bin_shapes");
  memory->create(num_bin_shapes,nbin,"fix_force_bound:num_bin_shapes");
  for (i = 0; i < nbin; ++i)
    num_bin_shapes[i] = 0;

  nn = static_cast<int> (d_cut*binsizeinv + 1.0);
  ii = 2*nn + 1;
  h_bin = ii*ii*ii;
    
  memory->create(bin_stensil,h_bin, nbin, "fix_force_bound:bin_stensil");
  memory->create(num_bin_stensil,nbin,"fix_force_bound:num_bin_stensil");
  for (bin = 0; bin < nbin; ++bin){
    num_bin_stensil[bin] = 0;
    iz = static_cast<int> (bin/nbinx/nbiny);
    iy = static_cast<int> ((bin-iz*nbinx*nbiny)/nbinx);
    ix = bin-iz*nbinx*nbiny - iy*nbinx;
    for (ii = ix-nn; ii <= ix+nn; ++ii)
      for (jj = iy-nn; jj <= iy+nn; ++jj)
        for (kk = iz-nn; kk <= iz+nn; ++kk){
          h_bin = kk*nbiny*nbinx + jj*nbinx + ii;
          if (bin_distance(ii-ix,jj-iy,kk-iz) < d_cut_sq && h_bin > -1 && h_bin < nbin){
            bin_stensil[num_bin_stensil[bin]][bin] = h_bin;
            num_bin_stensil[bin]++;          
          }       
        }
  }
}

/* ---------------------------------------------------------------------- */

double FixForceBound::bin_distance(int i, int j, int k)
{
  double delx,dely,delz;

  if (i > 0) delx = (i-1)*binsize;
  else if (i == 0) delx = 0.0;
  else delx = (i+1)*binsize;

  if (j > 0) dely = (j-1)*binsize;
  else if (j == 0) dely = 0.0;
  else dely = (j+1)*binsize;

  if (k > 0) delz = (k-1)*binsize;
  else if (k == 0) delz = 0.0;
  else delz = (k+1)*binsize;

  return (delx*delx + dely*dely + delz*delz);
}

/* ---------------------------------------------------------------------- */

int FixForceBound::check_bins(int bin, int sh)
{
  int ix, iy, iz, ch_ind;
  double rad, rad_n, rad_t, u0, u1, u2, zz, dl;
  double dd[3];

  ch_ind = 0;
  rad = 0.5*sqrt(3.0)*binsize;
  rad_n = rad + r_cut_n;
  rad_t = rad + r_cut_t;
  iz = static_cast<int> (bin/nbinx/nbiny);
  iy = static_cast<int> ((bin-iz*nbinx*nbiny)/nbinx);
  ix = bin-iz*nbinx*nbiny - iy*nbinx;
  dd[0] = bboxlo[0] + (ix + 0.5)*binsize - x0[sh][0];
  dd[1] = bboxlo[1] + (iy + 0.5)*binsize - x0[sh][1];
  dd[2] = bboxlo[2] + (iz + 0.5)*binsize - x0[sh][2];

  rot_forward(dd[0],dd[1],dd[2],sh);
  switch (ptype[sh]){
    case 1 :
      if (dd[0] > aa[sh][4]-rad_t && dd[0] < aa[sh][5]+rad_t && dd[1] > -rad_t && dd[1] < aa[sh][2]+rad_t && dd[2] > -rad_n && dd[2] < rad_n){
        u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
        zz = rad_t*aa[sh][3]/aa[sh][2];
        dl = rad_t*aa[sh][3]/(aa[sh][6]*aa[sh][1]+aa[sh][7]*aa[sh][2]);
        u2 = dd[1]/(aa[sh][2] + dl*aa[sh][2]/aa[sh][3]);
        u1 = (dd[0] - u2*(aa[sh][1] + dl*aa[sh][1]/aa[sh][3]))/(aa[sh][0] + rad_t/aa[sh][6]);
        if (u0 > -zz && u1+u2 < 1.0)
          ch_ind = 1;
      }
      break;
    case 2 :
      if (dd[0] > aa[sh][4]-rad_t && dd[0] < aa[sh][5]+rad_t && dd[1] > -rad_t && dd[1] < aa[sh][2]+rad_t && dd[2] > -rad_n && dd[2] < rad_n){
        u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
        zz = rad_t*aa[sh][3]/aa[sh][2];
        if (u0 > -zz && u0 < aa[sh][0] + zz)
          ch_ind = 1;
      }
      break;
    case 3 :
      zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
      if (zz > aa[sh][1]-rad_n && zz < aa[sh][1]+rad_n && dd[2] > -rad_t && dd[2] < aa[sh][0]+rad_t)
        ch_ind = 1;
      break;
    case 4 :
      zz = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
      if (zz > aa[sh][0]-rad_n && zz < aa[sh][0]+rad_n)
        ch_ind = 1;
      break;
  }

  return ch_ind;
}     

/* ---------------------------------------------------------------------- */

void FixForceBound::bin_vertices()
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
        memory->grow(bin_vertex,bin_max_vertex,nbin,"fix_force_bound:bin_vertex"); 
      }     
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixForceBound::coord2bin(double *xx)
{
  int ix,iy,iz;

  ix = static_cast<int> ((xx[0]-bboxlo[0])*binsizeinv);
  if (ix > nbinx-1 || ix < 0) return -1;
  iy = static_cast<int> ((xx[1]-bboxlo[1])*binsizeinv);
  if (iy > nbiny-1 || iy < 0) return -1;
  iz = static_cast<int> ((xx[2]-bboxlo[2])*binsizeinv);
  if (iz > nbinz-1 || iz < 0) return -1;
  
  return (iz*nbiny*nbinx + iy*nbinx + ix);
}

/* ---------------------------------------------------------------------- */

void FixForceBound::recalc_force()
{
  int i,j,k,l,m,nn,st,ii;
  double vv,theta,theta1,nrm[3];
  double *tmp,*ntmp;

  memory->create(tmp,numt,"fix_force_bound:tmp");
  memory->create(ntmp,numt,"fix_force_bound:ntmp");

  n_accum = 0;
  for (i = 0; i < numt; i++){
    tmp[i] = 0.0;
    ntmp[i] = 0.0;
  }

  for (i = 0; i < num_shapes; i++){
    ii = shapes_local[i];   
    st = face_order[ii]; 
    l = 0; 
    for (j=0; j<ndiv[ii][0]; j++)
      for (k=0; k<ndiv[ii][1]; k++)
        for (m=0; m<2*n_per; m++){
          nn = st + l;
          tmp[nn] = num[i][m][j][k];
          tmp[nn+1] = velx[i][m][j][k];
          tmp[nn+2] = vely[i][m][j][k];
          tmp[nn+3] = velz[i][m][j][k];
          num[i][m][j][k] = 0.0;
          velx[i][m][j][k] = 0.0;
          vely[i][m][j][k] = 0.0;
          velz[i][m][j][k] = 0.0;
          l += 4;
	}
  }

  MPI_Allreduce(tmp,ntmp,numt,MPI_DOUBLE,MPI_SUM,world);
  
  for (i = 0; i < num_shapes; i++){
    ii = shapes_local[i];    
    st = face_order[ii]; 
    l = 0; 
    for (j=0; j<ndiv[ii][0]; j++)
      for (k=0; k<ndiv[ii][1]; k++)
        for (m=0; m<2*n_per; m++){
          nn = st + l;
          if (m == 0 && ntmp[nn] > 0.0 && ntmp[nn+4] > 0.0){
            vv=0.5*(3.0*ntmp[nn+1]/ntmp[nn]-ntmp[nn+5]/ntmp[nn+4]);
            fsx[i][0][j][k] += coeff*(vv-vel[ii][0]);
            vv=0.5*(3.0*ntmp[nn+2]/ntmp[nn]-ntmp[nn+6]/ntmp[nn+4]);
            fsy[i][0][j][k] += coeff*(vv-vel[ii][1]);
            vv=0.5*(3.0*ntmp[nn+3]/ntmp[nn]-ntmp[nn+7]/ntmp[nn+4]);
            fsz[i][0][j][k] += coeff*(vv-vel[ii][2]); 
	  }
          if (m == n_per && ntmp[nn] > 0.0 && ntmp[nn+4] > 0.0){
            vv=0.5*(3.0*ntmp[nn+1]/ntmp[nn]-ntmp[nn+5]/ntmp[nn+4]);
            fsx[i][1][j][k] += coeff*(vv-vel[ii][0]);
            vv=0.5*(3.0*ntmp[nn+2]/ntmp[nn]-ntmp[nn+6]/ntmp[nn+4]);
            fsy[i][1][j][k] += coeff*(vv-vel[ii][1]);
            vv=0.5*(3.0*ntmp[nn+3]/ntmp[nn]-ntmp[nn+7]/ntmp[nn+4]);
            fsz[i][1][j][k] += coeff*(vv-vel[ii][2]); 
	  }
	  l += 4;
	}
  }

  // projection of the adaptive shear force onto the tangential direction for cylinder and sphere shapes
  for (i = 0; i < num_shapes; i++){
    ii = shapes_local[i];
    if (ptype[ii] > 2){
      if (ptype[ii] == 3){
        for (k=0; k<ndiv[ii][1]; k++){
          theta = 2*M_PI*(k+0.5)/ndiv[ii][1];
          nrm[0] = cos(theta);
          nrm[1] = sin(theta);
          for (j=0; j<ndiv[ii][0]; j++)
            for (m=0; m<2; m++){
              vv = nrm[0]*fsx[i][m][j][k] + nrm[1]*fsy[i][m][j][k];
              fsx[i][m][j][k] -= vv*nrm[0];
              fsy[i][m][j][k] -= vv*nrm[1];
            }
        }
      } else {
        for (j=0; j<ndiv[ii][0]; j++){
          for (k=0; k<ndiv[ii][1]; k++){
            theta1 = M_PI*(j+0.5)/ndiv[ii][0];
            theta = 2*M_PI*(k+0.5)/ndiv[ii][1];
            nrm[0] = sin(theta1)*cos(theta);
            nrm[1] = sin(theta1)*sin(theta);
            nrm[2] = cos(theta1);

            for (m=0; m<2; m++){
              vv = nrm[0]*fsx[i][m][j][k] + nrm[1]*fsy[i][m][j][k] + nrm[2]*fsz[i][m][j][k];
              fsx[i][m][j][k] -= vv*nrm[0];
              fsy[i][m][j][k] -= vv*nrm[1];
              fsz[i][m][j][k] -= vv*nrm[2];
            }
          }
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
    for (j=0; j<ndiv[m][0]; j++)
      for (k=0; k<ndiv[m][1]; k++){
        fprintf(f_write,"%lf %lf %lf \n",fsx[i][0][j][k], fsy[i][0][j][k], fsz[i][0][j][k]);
        fprintf(f_write,"%lf %lf %lf \n",fsx[i][1][j][k], fsy[i][1][j][k], fsz[i][1][j][k]);
      }
  }
  fclose(f_write);*/
}

/* ---------------------------------------------------------------------- */

void FixForceBound::borders()
{
  int i,m,iswap,dim,ineed,nsend,nrecv,nfirst,nlast,sendflag,twoneed,smax,rmax;
  int nn[3],ind;
  double lo,hi;
  double *buf;
  MPI_Request request;

  iswap = 0;
  smax = rmax = 0; 
  
  if (user_part){

    for (dim = 0; dim < n_neigh; dim++) {

      // find atoms within send bins 
      // check only local atoms 
      // store sent atom indices in list for use in future timesteps

      m = nsend = 0;

      for (i = 0; i < npart; i++){
        ind = comm->coords_to_bin(x_part[i],nn,nbp_loc,box_min);
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

void FixForceBound::communicate()
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

void FixForceBound::reverse_communicate(int v_f)
{
  int iswap,n, dim;
  MPI_Request request;

  if (user_part){

    for (iswap = n_neigh-1; iswap >= 0; iswap--) {
      if (size_reverse_recv[iswap])
        MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,sendproc[iswap],me,world,&request);
      n = pack_reverse_h(recvnum[iswap],firstrecv[iswap],buf_send,v_f);
      if (n) MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],recvproc[iswap],world);
      if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      unpack_reverse_h(sendnum[iswap],sendlist[iswap],buf_recv,v_f);
    }

  } else {

    for (iswap = nswap-1; iswap >= 0; iswap--) {

      if (sendproc[iswap] != me) {
        if (size_reverse_recv[iswap]) 
          MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
        n = pack_reverse_h(recvnum[iswap],firstrecv[iswap],buf_send,v_f);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
        if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);    
        unpack_reverse_h(sendnum[iswap],sendlist[iswap],buf_recv,v_f);
      } 
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound::local_faces(int n)
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
        for (j = 0; j < 3; j++){
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
        for (j = 0; j < 3; j++)
          x_part[npart][j] = x[i][j];
        npart++;    
      }
    npart_loc = npart;  
  }
}


/* ---------------------------------------------------------------------- */

void FixForceBound::grow_send(int n)
{
  maxsend = static_cast<int> (BUFFACTOR * n);
  memory->grow(buf_send,maxsend+BUFEXTRA,"fix_force_bound:buf_send");
}

/* ---------------------------------------------------------------------- */

void FixForceBound::grow_list(int iswap, int n)
{
  maxsendlist[iswap] = static_cast<int> (BUFFACTOR * n);
  memory->grow(sendlist[iswap],maxsendlist[iswap],"fix_force_bound:sendlist[iswap]");
}

/* ---------------------------------------------------------------------- */

void FixForceBound::grow_recv(int n)
{
  maxrecv = static_cast<int> (BUFFACTOR * n);
  memory->destroy(buf_recv);
  memory->create(buf_recv,maxrecv,"fix_force_bound:buf_recv");
}

/* ---------------------------------------------------------------------- */

void FixForceBound::grow_tri_arrays()
{
  max_ntri += BUFEXTRA;
  memory->grow(tri_list,max_ntri,"fix_force_bound:tri_list");
  memory->grow(norm_list,max_ntri,"fix_force_bound:norm_list");
  memory->grow(norm,max_ntri,4,"fix_force_bound:norm");
  memory->grow(edge1,max_ntri,3,"fix_force_bound:edge1");
  memory->grow(edge2,max_ntri,3,"fix_force_bound:edge2");
  memory->grow(dif1,max_ntri,3,"fix_force_bound:dif1");
  memory->grow(dif2,max_ntri,3,"fix_force_bound:dif2");
  memory->grow(dd12,max_ntri,3,"fix_force_bound:dd12"); 
}

/* ---------------------------------------------------------------------- */

void FixForceBound::grow_part_arrays()
{
  max_npart += BUFEXTRA;
  memory->grow(part_list,max_npart,"fix_force_bound:part_list");
  memory->grow(x_part,max_npart,3,"fix_force_bound:x_part");
  memory->grow(v_part,max_npart,3,"fix_force_bound:v_part");
  memory->grow(v_delt,max_npart,3,"fix_force_bound:v_delt");
  memory->grow(f_part,max_npart,3,"fix_force_bound:f_part");
} 

/* ---------------------------------------------------------------------- */

int FixForceBound::pack_border_h(int i, double *buf, int pbc_flag, int *pbc)
{
  int m = 0;

  buf[m++] = static_cast<double> (part_list[i]);
  if (pbc_flag == 0) {
    buf[m++] = x_part[i][0];
    buf[m++] = x_part[i][1];
    buf[m++] = x_part[i][2];
  } else {
    buf[m++] = x_part[i][0] + pbc[0]*prd[0];
    buf[m++] = x_part[i][1] + pbc[1]*prd[1];
    buf[m++] = x_part[i][2] + pbc[2]*prd[2];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

int FixForceBound::unpack_border_h(double *buf)
{
  int m = 0;

  if (npart == max_npart) grow_part_arrays();
  part_list[npart] = static_cast<tagint> (buf[m++]);
  tags_part[part_list[npart]] = npart;
  x_part[npart][0] = buf[m++];
  x_part[npart][1] = buf[m++];
  x_part[npart][2] = buf[m++];
  npart++;
  
  return m;
}

/* ---------------------------------------------------------------------- */

int FixForceBound::pack_comm_h(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x_part[j][0];
      buf[m++] = x_part[j][1];
      buf[m++] = x_part[j][2];
      buf[m++] = v_part[j][0];
      buf[m++] = v_part[j][1];
      buf[m++] = v_part[j][2];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x_part[j][0] + pbc[0]*prd[0];
      buf[m++] = x_part[j][1] + pbc[1]*prd[1];
      buf[m++] = x_part[j][2] + pbc[2]*prd[2];       
      buf[m++] = v_part[j][0];
      buf[m++] = v_part[j][1];
      buf[m++] = v_part[j][2];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixForceBound::unpack_comm_h(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x_part[i][0] = buf[m++];
    x_part[i][1] = buf[m++];
    x_part[i][2] = buf[m++];    
    v_part[i][0] = buf[m++];
    v_part[i][1] = buf[m++];
    v_part[i][2] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int FixForceBound::pack_reverse_h(int n, int first, double *buf, int v_f)
{
  int i,m,last;

  m = 0;
  last = first + n;
  if (v_f)
    for (i = first; i < last; i++) {
      buf[m++] = f_part[i][0];
      buf[m++] = f_part[i][1];
      buf[m++] = f_part[i][2];
    }
  else 
    for (i = first; i < last; i++) {
      buf[m++] = v_delt[i][0];
      buf[m++] = v_delt[i][1];
      buf[m++] = v_delt[i][2];
    }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixForceBound::unpack_reverse_h(int n, int *list, double *buf, int v_f)
{
  int i,j,m;

  m = 0;
  if (v_f)
    for (i = 0; i < n; i++) {
      j = list[i]; 
      f_part[j][0] += buf[m++];
      f_part[j][1] += buf[m++];
      f_part[j][2] += buf[m++];
    }
  else  
    for (i = 0; i < n; i++) {
      j = list[i];
      v_delt[j][0] += buf[m++];
      v_delt[j][1] += buf[m++];
      v_delt[j][2] += buf[m++];
    }
}

/* ---------------------------------------------------------------------- */

/*void FixForceBound::face_decide()
{
  int i,l,m,ind,jj;
  tagint j,k,n;
  double dd[3],rr;

  double **x = atom->x;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;  

  ntri = 0;
  for (jj = 0; jj < npart; jj++){
    j = part_list[jj]; 
    for (n = 0; n < part_degree[j]; n++){
      k = part_tri[j][n];
      if (norm_ind[k] == -1 && tags_part[tri_tags[k][0]] > -1 &&  tags_part[tri_tags[k][1]] > -1 && tags_part[tri_tags[k][2]] > -1) {
        if (ntri == max_ntri) grow_tri_arrays();
        norm_ind[k] = ntri;
        tri_list[ntri] = k;
        ntri++;
      }
    }
  }
  for (jj = 0; jj < ntri; jj++)
    norm_ind[tri_list[jj]] = -1; 
   
  for (i=0; i<atom->nlocal; i++)
    if (mask[i] & groupbit){ 
      num_faces[i] = 0;
      for (j = 0; j < ntri; j++){
        n = tri_list[j]; 
        if (molecule[i] != tri_mol[n]){
          ind = 0;
          for (l = 0; l < 3; l++){ 
            jj = tags_part[tri_tags[n][l]]; 
            for (k = 0; k < 3; k++)
              dd[k] = x[i][k] - x_part[jj][k];
            domain->minimum_image(dd[0],dd[1],dd[2]);
            rr = dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2];
            if (rr < d_cut_sq){
              ind = 1;
              break;      
	    }
	  }  
	  if (ind){
            if (num_faces[i] == max_faces) grow_face_arrays();
            ind_faces[i][num_faces[i]] = n;
            num_faces[i]++;
	  }
	}
      }
    } 
}*/

/* ---------------------------------------------------------------------- */

void FixForceBound::face_decide()
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
        fprintf(stderr, "Negative bin in face_decide: me=%d, step=" BIGINT_FORMAT ", particle - %f %f %f; box - %f %f %f %f %f %f \n",me,update->ntimestep,x[i][0],x[i][1],x[i][2],bboxlo[0],bboxlo[1],bboxlo[2],bboxhi[0],bboxhi[1],bboxhi[2]);
      }else{         
        for (kk = 0; kk < num_bin_stensil[bin]; ++kk){
          h_bin = bin_stensil[kk][bin];
          for (ii = 0; ii < num_bin_vertex[h_bin]; ++ii){
            jj = bin_vertex[ii][h_bin];  
            j = part_list[jj];   
            if (molecule[i] != tri_mol[part_tri[j][0]]){
              l = tags_part[j];
              for (k = 0; k < 3; k++)
                dd[k] = x[i][k] - x_part[l][k];
              domain->minimum_image(dd[0],dd[1],dd[2]);
              rr = dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2];
	      if (rr < d_cut_sq)
                for (n = 0; n < part_degree[j]; n++){
                  k = part_tri[j][n];
                  ind_h = 1;
                  for (m = 0; m < num_faces[i]; m++)
                    if (ind_faces[i][m] == k){
                      ind_h = 0;
                      break;
                    }
                  if (ind_h){
                    if (norm_ind[k] == -1 && tags_part[tri_tags[k][0]] > -1 &&  tags_part[tri_tags[k][1]] > -1 && tags_part[tri_tags[k][2]] > -1) {
                      norm_ind[k] = 1;
                      if (num_faces[i] == max_faces) grow_face_arrays();
                      ind_faces[i][num_faces[i]] = k;
                      num_faces[i]++;
                    } else {
                      if (norm_ind[k] == -1) 
                        printf("It might make sense to increase comm_cut, because some triangles are omitted!!! \n");
	  	    }
		  }
                }
            }
          }
        }
      }
      for (jj = 0; jj < num_faces[i]; jj++)
        norm_ind[ind_faces[i][jj]] = -1;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound::grow_face_arrays()
{
  int k, l, m;
  int **ind_tmp;

  if (max_faces == 0){
    max_faces = FACE_INC;
    memory->create(ind_faces,nnmax,max_faces,"fix_force_bound:ind_faces");
  } else{
    m = max_faces;
    memory->create(ind_tmp,nnmax,m,"fix_force_bound:ind_tmp");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_tmp[k][l] = ind_faces[k][l];
    max_faces += FACE_INC;
    memory->destroy(ind_faces);
    memory->create(ind_faces,nnmax,max_faces,"fix_force_bound:ind_faces");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_faces[k][l] = ind_tmp[k][l];
    memory->destroy(ind_tmp);
  }
}

/* ---------------------------------------------------------------------- */

void FixForceBound::calc_norm(int m)
{
  int k, i0, i1, i2;

  if (norm_count == max_ntri) grow_tri_arrays();
  norm_ind[m] = norm_count;
  norm_list[norm_count] = m;
  i0 = tags_part[tri_tags[m][0]];
  i1 = tags_part[tri_tags[m][1]];
  i2 = tags_part[tri_tags[m][2]]; 
  for (k = 0; k < 3; k++){
    edge1[norm_count][k] = x_part[i1][k] - x_part[i0][k];
    edge2[norm_count][k] = x_part[i2][k] - x_part[i0][k];
    dif1[norm_count][k] = v_part[i1][k] - v_part[i0][k];
    dif2[norm_count][k] = v_part[i2][k] - v_part[i0][k]; 
  }
  domain->minimum_image(edge1[norm_count][0],edge1[norm_count][1],edge1[norm_count][2]);
  domain->minimum_image(edge2[norm_count][0],edge2[norm_count][1],edge2[norm_count][2]);

  norm[norm_count][0] = edge1[norm_count][1]*edge2[norm_count][2] - edge1[norm_count][2]*edge2[norm_count][1];
  norm[norm_count][1] = edge1[norm_count][2]*edge2[norm_count][0] - edge1[norm_count][0]*edge2[norm_count][2];
  norm[norm_count][2] = edge1[norm_count][0]*edge2[norm_count][1] - edge1[norm_count][1]*edge2[norm_count][0];
  norm[norm_count][3] = - norm[norm_count][0]*x_part[i0][0] - norm[norm_count][1]*x_part[i0][1] - norm[norm_count][2]*x_part[i0][2];

  dd12[norm_count][0] = dif1[norm_count][1]*dif2[norm_count][2] - dif1[norm_count][2]*dif2[norm_count][1];
  dd12[norm_count][1] = dif1[norm_count][2]*dif2[norm_count][0] - dif1[norm_count][0]*dif2[norm_count][2];
  dd12[norm_count][2] = dif1[norm_count][0]*dif2[norm_count][1] - dif1[norm_count][1]*dif2[norm_count][0];  

  norm_count++;
}

/* ------------------------------------------------------------------------- */

double FixForceBound::find_root(double *cc)
{
  double dot,tt,t1,t2;
  double cp[3];

  if (cc[0] == 0.0){
    tt = solve_quadratic(cc);
    return tt;
  }
  cp[0] = 3.0*cc[0];
  cp[1] = 2.0*cc[1];
  cp[2] = cc[2];  

  dot = cp[1]*cp[1] - 4.0*cp[0]*cp[2];
  if (dot < 0){
    tt = newton(0.0,cc,cp);
  }
  else {
    t1 = 0.5*(-cp[1]+sqrt(dot))/cp[0];
    t2 = 0.5*(-cp[1]-sqrt(dot))/cp[0];
    if (t2 < t1) {
      tt = t2;
      t2 = t1;
      t1 = tt;
    }
    if (t2<0.0 || t1>1.0 || (t1<0.0 && t2>1.0)){
      tt = newton(0.0,cc,cp); 
    }
    else{
      if (t1>=0.0 && t2<=1.0){
        tt = newton(0.0,cc,cp);
        if (tt<0.0 && tt>1.0)
          tt = newton(1.0,cc,cp);
        if (tt<0.0 && tt>1.0)
          tt = newton(0.5*(t1+t2),cc,cp);  
      }else{
        tt = newton(0.0,cc,cp);
        if (tt<0.0 && tt>1.0)
          tt = newton(1.0,cc,cp); 
      }  
    }
  }

  return tt;
}

/* ------------------------------------------------------------------------- */

double FixForceBound::newton(double tt, double *cc, double *cp)
{
  int nn;
  double tt1,val,valp;
  int it_max = 50;

  nn = 0;
  val = cc[0]*tt*tt*tt + cc[1]*tt*tt + cc[2]*tt + cc[3];
  while (fabs(val)>EPSN && nn<it_max){
    valp = cp[0]*tt*tt + cp[1]*tt + cp[2];
    tt1 = tt - val/valp;
    nn++;
    tt = tt1;
    val = cc[0]*tt*tt*tt + cc[1]*tt*tt + cc[2]*tt + cc[3];
  }

  return tt;
}

/* ------------------------------------------------------------------------- */

double FixForceBound::solve_quadratic(double *cc)
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

/* ------------------------------------------------------------------------- */

void FixForceBound::move_norm_arrays(tagint m)
{
  int k,i;

  i = norm_ind[m];
  for (k = 0; k < 3; k++){
    norm[i][k] = norm[norm_count-1][k];
    edge1[i][k] = edge1[norm_count-1][k];  
    edge2[i][k] = edge2[norm_count-1][k];
    dif1[i][k] = dif1[norm_count-1][k];
    dif2[i][k] = dif2[norm_count-1][k];
    dd12[i][k] = dd12[norm_count-1][k];    
  }
  norm[i][3] = norm[norm_count-1][3];
  norm_list[i] = norm_list[norm_count-1]; 
  norm_ind[m] = -1;
  norm_ind[norm_list[i]] = i;  
  norm_count--;
}

/* ----------------------------------------------------------------------
   lees-edwards fix
------------------------------------------------------------------------- */

void FixForceBound::lees_edwards(int n)
{
  int i,j;
  double ss;

  if (n && le_max < le_npart - npart_loc){
    le_max = le_npart - npart_loc;
    memory->grow(le_sh,le_max,"fix_force_bound:le_sh");
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

void FixForceBound::write_restart(FILE *fp)
{
  int i,j,k,l,m,n,kk,nel_tot,offset,ind_sh;
  int *displs, *rcounts;
  double  *list, *tmp, *tmph;

  if (numt_loc)
    offset = numt_loc;
  else
    offset = 1;
  memory->create(tmp,offset,"fix_force_bound:tmp");

  nel_tot = 1;
  memory->create(rcounts,comm->nprocs,"fix_force_bound:rcounts");
  memory->create(displs,comm->nprocs,"fix_force_bound:displs");
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
    for (j=0; j<ndiv[m][0]; j++)
      for (k=0; k<ndiv[m][1]; k++){
        tmp[l] = fsx[i][0][j][k];
        tmp[l+1] = fsy[i][0][j][k];
        tmp[l+2] = fsz[i][0][j][k];
        tmp[l+3] = fsx[i][1][j][k];
        tmp[l+4] = fsy[i][1][j][k];
        tmp[l+5] = fsz[i][1][j][k];
        l += 6;
      }
  }

  if (l != numt_loc)
    error->warning(FLERR,"Something might be wrong: numt_loc and l are not equal in FixForceBound::write_restart!");

  memory->create(tmph,nel_tot,"fix_force_bound:tmph");
  if (comm->nprocs > 1){
    MPI_Gatherv(tmp, l, MPI_DOUBLE, tmph, rcounts, displs, MPI_DOUBLE, 0, world);
  } else{
    for (i=0; i<nel_tot; i++)
      tmph[i] = tmp[i];
  }
  
  if (me == 0){
    memory->create(list,numt_s*3+1,"fix_force_bound:list");
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
          l += 6*ndiv[m][0]*ndiv[m][1];
      }
      if (ind_sh){
        for (j=0; j<ndiv[i][0]; j++)
          for (k=0; k<ndiv[i][1]; k++){
            for (n=0; n<6; n++)
              list[kk+n] = tmph[l+n];
            kk += 6;    
            l += 6;
          }
      } else{
        for (j=0; j<ndiv[i][0]; j++)
          for (k=0; k<ndiv[i][1]; k++){
            for (n=0; n<6; n++)
              list[kk+n] = 0.0;
            kk += 6;
          }
      }
    }
    if (kk != numt_s*3)
      error->warning(FLERR,"Something might be wrong: kk and 3*numt_s are not equal in FixForceBound::write_restart!");
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

void FixForceBound::restart(char *buf)
{
  int i,j,k,l,m;
  double *list = (double *) buf;  

  l = 0;
  for (i=0; i<num_shapes_tot; i++)
    for (j=0; j<ndiv[i][0]; j++)
      for (k=0; k<ndiv[i][1]; k++){
        m = shapes_global_to_local[i];
        if (m > -1){
          fsx[m][0][j][k] = list[l];
          fsy[m][0][j][k] = list[l+1];
          fsz[m][0][j][k] = list[l+2];
          fsx[m][1][j][k] = list[l+3];
          fsy[m][1][j][k] = list[l+4];
          fsz[m][1][j][k] = list[l+5];
        }
        l += 6;
      }
  cur_iter = static_cast<int> (list[l]);
  
  read_restart_ind = 1;

  /* char fname[FILENAME_MAX]; 
  FILE *f_write;
  sprintf(fname,"read_bla_%d.dat",me);
  f_write = fopen(fname,"w");
  fprintf(f_write,"%d \n",cur_iter);
  for (i=0; i<num_shapes; i++){
    m = shapes_local[i];
    fprintf(f_write,"%d \n",m);
    for (j=0; j<ndiv[m][0]; j++)
      for (k=0; k<ndiv[m][1]; k++){
        fprintf(f_write,"%lf %lf %lf \n",fsx[i][0][j][k], fsy[i][0][j][k], fsz[i][0][j][k]);
        fprintf(f_write,"%lf %lf %lf \n",fsx[i][1][j][k], fsy[i][1][j][k], fsz[i][1][j][k]);
      }
  }
  fclose(f_write);  */
}

/* ----------------------------------------------------------------------
   allocation of swap arrays
------------------------------------------------------------------------- */

void FixForceBound::allocate_swap(int n)
{
  int i;

  memory->create(sendnum,n,"fix_force_bound:sendnum");
  memory->create(recvnum,n,"fix_force_bound:recvnum");
  memory->create(sendproc,n,"fix_force_bound:sendproc");
  memory->create(recvproc,n,"fix_force_bound:recvproc");
  memory->create(size_forward_recv,n,"fix_force_bound:size");
  memory->create(size_reverse_send,n,"fix_force_bound:size");
  memory->create(size_reverse_recv,n,"fix_force_bound:size");
  memory->create(firstrecv,n,"fix_force_bound:firstrecv");
  memory->create(pbc_flag,n,"fix_force_bound:pbc_flag");
  memory->create(pbc,n,6,"fix_force_bound:pbc");
  
  if (user_part == 0){
    memory->create(slablo,n,"fix_force_bound:slablo");
    memory->create(slabhi,n,"fix_force_bound:slabhi");
  }

  sendlist = (int **) memory->smalloc(n*sizeof(int *),"fix_force_bound:sendlist");
  memory->create(maxsendlist,n,"fix_force_bound:maxsendlist");
  for (i = 0; i < n; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i],BUFMIN,"fix_force_bound:sendlist[i]");
  } 
}

/* ----------------------------------------------------------------------
   free memory for swaps
------------------------------------------------------------------------- */

void FixForceBound::free_swap()
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


