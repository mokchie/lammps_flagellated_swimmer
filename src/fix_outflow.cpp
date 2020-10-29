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

#include "lmptype.h"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fix_outflow.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "group.h"
#include "error.h"
#include "memory.h"
#include "update.h"
#include "neighbor.h"
#include "domain.h"

#define  NUM 10000
#define  NUM1 10
#define  FACE_INC 5
#define  BUFMIN 1000
#define  EPS 1e-6
#define  DR 0.01
#define  BIG 1.0e20

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixOutflow::FixOutflow(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  int i, j, ttyp, igroup,l,nms,nm_tot,ndv1,ndv2;
  double *rdata = NULL;
  double *tmp = NULL;
  double dummy,a1[3],a2[3],wd,rr;
  char grp[50];
  char buf[BUFSIZ];
  char fname[FILENAME_MAX],fname1[FILENAME_MAX];
  FILE *f_read,*f_read1;

  ptype = flux_ind = shapes_local = tot_shapes = num_bin_shapes = NULL;
  displs = rcounts = face_order = glob_order = NULL;
  ndiv = ind_shapes = bin_shapes = NULL;
  del_mol_list = NULL;
  f_press = flux = weight = area = NULL;
  x0 = aa = NULL;
  rot = beta = NULL;
  veln = num = save_beta = NULL;
  dr_inv = 1.0/DR;
  read_restart_ind = 0;
  restart_global = 1;

  nms = NUM;
  groupbit_sol = 0;
  if (narg != 4) error->all(FLERR,"Illegal fix_outflow command");
  sprintf(fname,arg[3]);
  me = comm->me;
  cur_iter = 1;

  if (me == 0){
    l = 0;
    f_read = fopen(fname,"r");
    if (f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open input boundary file from fix_outflow!");
    memory->create(rdata,nms,"fix_outflow:rdata");
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&nm_tot);
    if (nm_tot <= 0)
      error->one(FLERR,"The total number of shapes in input boundary file from fix_outflow is non-positive!"); 
    rdata[l++] = static_cast<double> (nm_tot); 

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %lf %d %d %d %s",&r_cut_n,&r_cut_t,&binsize,&iter,&mmax_iter,&del_every,grp);
    rdata[l++] = r_cut_n;
    rdata[l++] = r_cut_t;
    rdata[l++] = binsize;
    rdata[l++] = static_cast<double> (iter);
    rdata[l++] = static_cast<double> (mmax_iter);
    rdata[l++] = static_cast<double> (del_every);
    igroup = group->find(grp);
    if (igroup == -1)
      error->one(FLERR,"Group ID for group_solvent does not exist in fix_outflow!");
    groupbit_sol = group->bitmask[igroup];
    rdata[l++] = static_cast<double> (groupbit_sol);

    fscanf(f_read,"%d",&num_flux);
    rdata[l++] = static_cast<double> (num_flux);
    if (num_flux <= 0)
      error->one(FLERR,"The number of fluxes in input boundary file from fix_outflow is non-positive!"); 
    for (i=0; i<num_flux; i++){
      fscanf(f_read,"%lf",&dummy);
      rdata[l++] = dummy;  
    }
    fgets(buf,BUFSIZ,f_read);

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %lf %lf %d",&r_shear,&r_shift,&coeff_s,&power,&qk_max);
    rdata[l++] = r_shear;
    rdata[l++] = r_shift;
    rdata[l++] = coeff_s;
    rdata[l++] = power;
    rdata[l++] = static_cast<double> (qk_max);

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d %lf %lf %lf %s",&n_press,&r_press,&coeff_p,&rho_target,fname1);
    rdata[l++] = static_cast<double> (n_press);
    rdata[l++] = r_press;
    rdata[l++] = coeff_p;
    rdata[l++] = rho_target;
    f_read1 = fopen(fname1,"r");
    if (f_read1 == (FILE*) NULL)
      error->one(FLERR,"Could not open input pressure file in fix_outflow!");
    for (i=0; i<n_press; i++){
      fgets(buf,BUFSIZ,f_read1);
      sscanf(buf,"%lf %lf",&dummy,&rdata[l]);
      l++;
    }
    fclose(f_read1);

    if (r_cut_n < r_shear+r_shift || r_cut_n < r_press)
      error->warning(FLERR,"The parameter r_cut_n is too small in fix_outflow!");

    for (i=0; i<nm_tot; i++){
      fgets(buf,BUFSIZ,f_read);
      sscanf(buf,"%d", &ttyp);
      if (l + 15 > nms){
        nms += NUM;
        memory->grow(rdata,nms,"fix_outflow:rdata");
      }
      switch (ttyp){
        case 1 :
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],
		&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8],&rdata[l+9],&rdata[l+10]);
	  l +=11;
  	  break;
        case 2 :
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],
		&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8],&rdata[l+9],&rdata[l+10],&rdata[l+11],&rdata[l+12]);
          l += 13;
          break;
      }
    }
    fclose(f_read);
  }

  MPI_Bcast(&l,1,MPI_INT,0,world);
  if (me)
    memory->create(rdata,l,"fix_outflow:rdata");
  MPI_Bcast(rdata,l,MPI_DOUBLE,0,world);

  l = 0;
  nm_tot = static_cast<int> (rdata[l++]);
  r_cut_n = rdata[l++];
  r_cut_t = rdata[l++];
  binsize = rdata[l++];
  iter = static_cast<int> (rdata[l++]);
  mmax_iter = static_cast<int> (rdata[l++]);
  del_every = static_cast<int> (rdata[l++]);
  groupbit_sol = static_cast<int> (rdata[l++]);
  if (binsize <= 0.0)
    binsize = 0.5*r_cut_n;
  binsizeinv = 1.0/binsize;

  num_flux = static_cast<int> (rdata[l++]);
  memory->create(flux,num_flux,"fix_outflow:flux");
  for (i=0; i<num_flux; i++)
    flux[i] = rdata[l++];

  r_shear = rdata[l++];
  r_shift = rdata[l++];
  coeff_s = rdata[l++];
  power = rdata[l++];
  qk_max = static_cast<int> (rdata[l++]);
  nw_max = static_cast<int> (r_shear*dr_inv) + 2;
  memory->create(weight,nw_max,"fix_outflow:weight");
  for (i = 0; i < nw_max; i++){
    rr = i*DR;
    if (rr > r_shear)
      wd = 0.0;
    else  
      wd = 1.0 - rr/r_shear;
    weight[i] = pow(wd,power);
  }  

  n_press = static_cast<int> (rdata[l++]);
  r_press = rdata[l++];
  coeff_p = rdata[l++]; 
  rho_target = rdata[l++];
  memory->create(f_press,n_press+1,"fix_outflow:f_press");
  for (i=0; i<n_press; i++)
    f_press[i] = rdata[l++];
  f_press[n_press] = 0.0;

  num_shapes = 0;
  max_shapes = 0;
  numt = 0;
  num_shapes_tot = 0;
  del_mol_max = NUM1;
  nnmax = atom->nlocal;
  if (nnmax == 0) nnmax = 1;
  bin_max_shape = NUM1; 
  max_area = NUM1;
  area_tot = 0.0;

  grow_basic_arrays(0);
  memory->create(del_mol_list,del_mol_max,"fix_outflow:del_mol_list");  
  memory->create(tot_shapes,nnmax,"fix_outflow:tot_shapes");
  memory->create(area,max_area,"fix_outflow:area");
  memory->create(glob_order,max_area,"fix_outflow:glob_order");

  nms = l;
  memory->create(tmp,nm_tot,"fix_outflow:tmp");
  for (i=0; i<nm_tot; i++){
    ptype[num_shapes] = static_cast<int> (rdata[l]);
    switch (ptype[num_shapes]){
      case 1 :
        for (j=0; j<3; j++){
          x0[num_shapes][j] = rdata[l+j+1];
          a1[j] = rdata[l+j+4] - rdata[l+j+1];
          a2[j] = rdata[l+j+7] - rdata[l+j+1];
        }
        flux_ind[num_shapes] = static_cast<int> (rdata[l+10]);
        ndiv[num_shapes][0] = 1;
        ndiv[num_shapes][1] = 1;
	l += 11;
	break;
      case 2 :
        for (j=0; j<3; j++){
          x0[num_shapes][j] = rdata[l+j+1];
          a1[j] = rdata[l+j+4] - rdata[l+j+1];
          a2[j] = rdata[l+j+7] - rdata[l+j+1];
        }
        ndiv[num_shapes][0] = static_cast<int> (rdata[l+10]);
        ndiv[num_shapes][1] = static_cast<int> (rdata[l+11]);
        flux_ind[num_shapes] = static_cast<int> (rdata[l+12]);
	l += 13;
        break;
    }
    setup_rot(num_shapes,a1,a2);
    tmp[i] = area[num_shapes];

    if (check_shapes(num_shapes)){
      shapes_local[num_shapes] = i;
      num_shapes++;
      if (num_shapes == max_loc_shapes)
        grow_basic_arrays(1);
      if (num_shapes == max_area){
        max_area += NUM1;
        memory->grow(area,max_area,"fix_outflow:area");
        memory->grow(glob_order,max_area,"fix_outflow:glob_order");
      }  
    }
  }

  set_comm_groups();   // setup communication groups

  l = nms;
  nms = 0;
  for (i=0; i<nm_tot; i++){
    ttyp = static_cast<int> (rdata[l]);
    switch (ttyp){
      case 1 :
        j = static_cast<int> (rdata[l+10]); 
        ndv1 = 1;
        ndv2 = 1;
        l += 11;
	break;
      case 2 :
        j = static_cast<int> (rdata[l+12]);
        ndv1 = static_cast<int> (rdata[l+10]);
        ndv2 = static_cast<int> (rdata[l+11]);         
        l += 13; 
        break;
    }
    if (j == noutf){
      if (shapes_local[nms] == i){
        shapes_local[nms] = num_shapes_tot;
        face_order[nms] = numt;
        nms++; 
      }
      glob_order[num_shapes_tot] = ndv1*ndv2;
      numt += 4*ndv1*ndv2;
      area[num_shapes_tot] = tmp[i];
      area_tot += area[num_shapes_tot];
      num_shapes_tot++;
      if (num_shapes_tot == max_area){
        max_area += NUM1;
        memory->grow(area,max_area,"fix_outflow:area");
        memory->grow(glob_order,max_area,"fix_outflow:glob_order");
      } 
    }
  }

  if (nms != num_shapes)
    error->one(FLERR,"Some inconsistency error at the preprocessing in fix_outflow!");

  memory->destroy(rdata);
  memory->destroy(tmp);

  if (num_shapes){
    veln = new double***[num_shapes];
    num = new double***[num_shapes];
    save_beta = new double***[num_shapes];
    beta = new double**[num_shapes]; 
    for (i=0; i<num_shapes; i++){
      memory->create(veln[i],2,ndiv[i][0],ndiv[i][1],"fix_outflow:veln[i]");
      memory->create(num[i],2,ndiv[i][0],ndiv[i][1],"fix_outflow:num[i]");
      memory->create(save_beta[i],qk_max,ndiv[i][0],ndiv[i][1],"fix_outflow:save_beta[i]");
      memory->create(beta[i],ndiv[i][0],ndiv[i][1],"fix_outflow:beta[i]"); 
    } 
  }

  memory->create(displs,comm->nprocs,"fix_outflow:displs");
  memory->create(rcounts,comm->nprocs,"fix_outflow:rcounts");
}

/* ---------------------------------------------------------------------- */

FixOutflow::~FixOutflow()
{
  int i;

  if (num_shapes){
    for (i=0; i<num_shapes; i++){
      memory->destroy(veln[i]);
      memory->destroy(num[i]);
      memory->destroy(save_beta[i]);
      memory->destroy(beta[i]);
    }
    delete[] veln;
    delete[] num;
    delete[] save_beta;
    delete[] beta;
  }     

  memory->destroy(ptype); 
  memory->destroy(ndiv);
  memory->destroy(rot);
  memory->destroy(x0);
  memory->destroy(aa); 
  memory->destroy(shapes_local);
  memory->destroy(flux);
  memory->destroy(flux_ind);
  memory->destroy(area);
  memory->destroy(del_mol_list);
  memory->destroy(rcounts);
  memory->destroy(displs);
  memory->destroy(weight);
  memory->destroy(face_order);
  memory->destroy(f_press);
  memory->destroy(glob_order);
  memory->destroy(bin_shapes);
  memory->destroy(num_bin_shapes);
  memory->destroy(tot_shapes);
  if (max_shapes) memory->destroy(ind_shapes);
}

/* ---------------------------------------------------------------------- */

int FixOutflow::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */


void FixOutflow::setup(int vflag)
{
  int i,j,k,l;

  if (neighbor->every > 1 || neighbor->delay > 1) 
    error->all(FLERR,"Reneighboring should be done every time step when using fix_outflow!");

  n_accum = 0;
  del_mol_num = 0;
  gamma = 1.0;

  // zero out initial arrays
  if (num_shapes) 
    for (i = 0; i < num_shapes; ++i){    
      for (k = 0; k < ndiv[i][0]; k++)
        for (l = 0; l < ndiv[i][1]; l++){
          for (j = 0; j < 2; j++){
            veln[i][j][k][l] = 0.0;
            num[i][j][k][l] = 0.0;
          }
          if (!read_restart_ind){
            for (j = 0; j < qk_max; j++) 
              save_beta[i][j][k][l] = 0.0;
            beta[i][k][l] = 0.0;
          }
	}
    }

  // setup bins in the domain
  setup_bins();
  for (i = 0; i < nbin; ++i)
    for (j = 0; j < num_shapes; ++j){
       if (check_bins(i,j)){
	 bin_shapes[num_bin_shapes[i]][i] = j;
         num_bin_shapes[i]++;
         if (num_bin_shapes[i] == bin_max_shape){
           bin_max_shape += NUM1;
           memory->grow(bin_shapes,bin_max_shape,nbin,"fix_outflow:bin_shapes");
         }
       }
    }

  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixOutflow::pre_exchange()
{

  int i,j,k,kk,ind,nd;
  double hh,xh[3];
  bigint ntemp;
  bool inside;
  double **x = atom->x;
  int nlocal = atom->nlocal;
  tagint *molecule = atom->molecule;
  tagint *del_buf = NULL;
  int step_t = update->ntimestep;

  if (bc_comm_rank > -1){
    i = 0;
    while (i < nlocal){  //delelting particles
      inside = true;
      for (k = 0; k < tot_shapes[i]; ++k){
	kk = ind_shapes[i][k];
	for (j = 0; j < 3; ++j)
	  xh[j] = x[i][j] - x0[kk][j];
	rot_forward(xh[0],xh[1],xh[2],kk);
	hh = xh[2];
	if (hh < 0.0){
	  inside = false;
	  break;
	}
      }

      if (inside == false){
	if (atom->molecular){
	  if (molecule[i] > 0){
	    if (del_mol_num == del_mol_max){
	      del_mol_max += NUM1;
	      memory->grow(del_mol_list,del_mol_max,"fix_outflow:del_mol_list");
	    }
	    ind = 1;
	    for (j = 0; j < del_mol_num; ++j)
	      if (molecule[i] == del_mol_list[j]){
		ind = 0;
		break;
	      }
	    if (ind){
	      del_mol_list[del_mol_num] = molecule[i];
	      del_mol_num++;
	    }
	    ++i;	
	  } else{
	    k = nlocal-1;
	    if (k > 0){
	      for (j = 0; j < tot_shapes[k]; ++j)
		ind_shapes[i][j] = ind_shapes[k][j];
	      tot_shapes[i] = tot_shapes[k]; 
	      atom->avec->copy(k,i,1); 
	    }
	    atom->nlocal--;
	    nlocal = atom->nlocal;
	  }
	} else{
	  k = nlocal-1;
	  if (k > 0){ 
	    for (j = 0; j < tot_shapes[k]; ++j)
	      ind_shapes[i][j] = ind_shapes[k][j];
	    tot_shapes[i] = tot_shapes[k]; 
	    atom->avec->copy(k,i,1);
	  }
	  atom->nlocal--;
	  nlocal = atom->nlocal;
	}
      } else
	++i;
    }
  }

  if (atom->molecular)
    if (step_t%del_every == 0){

      MPI_Gather(&del_mol_num,1,MPI_INT,rcounts,1,MPI_INT,0,world);

      if (me == 0){
	nd = 0;
	for (i=0; i<comm->nprocs; ++i){
	  displs[i] = nd;
	  nd += rcounts[i];
	}
      }
      MPI_Bcast(&nd,1,MPI_INT,0,world);

      if (nd){
        atom->fix_force_bound_ind = 1; 
	memory->create(del_buf,nd,"fix_outflow:del_buf");

	MPI_Gatherv(del_mol_list,del_mol_num,MPI_LMP_TAGINT,del_buf,rcounts,displs,MPI_LMP_TAGINT,0,world);

	MPI_Bcast(del_buf,nd,MPI_LMP_TAGINT,0,world);

	i = 0;
	while (i < nlocal){
	  if (molecule[i] > 0){
	    ind = 0;
	    for (j=0; j<nd; ++j)
	      if (molecule[i] == del_buf[j]){
		ind = 1;
		break;
	      }
	    if (ind){
	      k = nlocal-1;
	      if (k > 0){
		for (j = 0; j < tot_shapes[k]; ++j)
		  ind_shapes[i][j] = ind_shapes[k][j];
		tot_shapes[i] = tot_shapes[k]; 
		atom->avec->copy(k,i,1);
	      }
              atom->nlocal--;
              nlocal = atom->nlocal;
	    } else
	      ++i;
	  } else
	    ++i;
	}
	memory->destroy(del_buf);
	del_mol_num = 0;
      }
    }

  ntemp = atom->nlocal;
  MPI_Allreduce(&ntemp,&atom->natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (atom->natoms < 0 || atom->natoms >= MAXBIGINT)
    error->all(FLERR,"Too many total atoms!");

  if (atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set(); 
  }
}

/*------------------------------------------------------------------------*/

void FixOutflow::post_force(int vflag)
{
  int i,j,k,m,kk,ix,iy,iz;
  double xh[3],vh[3],ff[3];
  double u0,u1,weight_h,zz;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int step_t = update->ntimestep;

  if (atom->nlocal > nnmax){
    while (nnmax < atom->nlocal)               
      nnmax += BUFMIN;
    if (max_shapes)
      memory->grow(ind_shapes,nnmax,max_shapes,"fix_outflow:ind_shapes");
    memory->grow(tot_shapes,nnmax,"fix_outflow:tot_shapes");
  }  

  if (neighbor->ago == 0)
    if (bc_comm_rank > -1)
      shape_decide();

  n_accum++;
  if (bc_comm_rank > -1)
    for (i = 0; i < nlocal; i++)
      for (m = 0; m < tot_shapes[i]; m++){
	kk = ind_shapes[i][m]; 
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

	      if (mask[i] & groupbit_sol){
		if (xh[2] > 0.0 && xh[2] < r_shift + r_shear){
                  if (xh[2] > r_shift){
                    zz = xh[2]-r_shift;
                    if (cur_iter){
		      iz = static_cast<int> (2.0*zz/r_shear);
		      veln[kk][iz][0][0] += vh[2];
		      num[kk][iz][0][0] += 1.0;
                    }
                  }

                  if (xh[2] < r_shear){
		    k = static_cast<int> (xh[2]*dr_inv);
		    weight_h = (weight[k+1]-weight[k])*(xh[2]*dr_inv - k) + weight[k];
		    ff[0] = 0.0;
		    ff[1] = 0.0;
		    ff[2] = beta[kk][0][0]*weight_h;
		    rot_back(ff[0],ff[1],ff[2],kk);
		    for (j = 0; j < 3; j++)
		      f[i][j] += ff[j];
                  }
	        }
                
		if (xh[2]>0.0 && xh[2]<r_press){ 
		  iz = static_cast<int> (n_press*xh[2]/r_press);
                  k = static_cast<int> (xh[2]*dr_inv);
                  weight_h = (weight[k+1]-weight[k])*(xh[2]*dr_inv - k) + weight[k];
		  ff[0] = 0.0;
		  ff[1] = 0.0;
		  ff[2] = gamma*f_press[iz];
		  //ff[2] = f_press[iz] + gamma*weight_h;
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
	      if (mask[i] & groupbit_sol){
		if (xh[2] > 0.0 && xh[2] < r_shift + r_shear){
                  if (xh[2] > r_shift){
                    zz = xh[2]-r_shift;
                    if (cur_iter){
		      iz = static_cast<int> (2*zz/r_shear);
		      veln[kk][iz][ix][iy] += vh[2];
		      num[kk][iz][ix][iy] += 1.0;
                    }
                  }

                  if (xh[2] < r_shear){
		    k = static_cast<int> (xh[2]*dr_inv);
		    weight_h = (weight[k+1]-weight[k])*(xh[2]*dr_inv - k) + weight[k];
		    ff[0] = 0.0;
		    ff[1] = 0.0;
		    ff[2] = beta[kk][ix][iy]*weight_h;
		    rot_back(ff[0],ff[1],ff[2],kk);
		    for (j = 0; j < 3; j++)
		      f[i][j] += ff[j];
                  }
		}

		if (xh[2]>0.0 && xh[2]<r_press){ 
		  iz = static_cast<int> (n_press*xh[2]/r_press);
                  k = static_cast<int> (xh[2]*dr_inv);
                  weight_h = (weight[k+1]-weight[k])*(xh[2]*dr_inv - k) + weight[k];       
		  ff[0] = 0.0;
		  ff[1] = 0.0;
		  ff[2] = gamma*f_press[iz];
                  //ff[2] = f_press[iz] + gamma*weight_h;
		  rot_back(ff[0],ff[1],ff[2],kk);
		  for (j = 0; j < 3; j++)
		    f[i][j] += ff[j];
		}
              }  
	    }
	    break;
	}
      }

  if (cur_iter && step_t%iter == 0  && n_accum > 0.6*iter)
    recalc_force();
}

/* ---------------------------------------------------------------------- */

void FixOutflow::recalc_force()
{
  int i,j,k,l,m,nn,st,ii;
  double q_tot, rho_h;
  double *tmp,*ntmp;
 
  tmp = ntmp = NULL;
 
  if (numt){
    memory->create(tmp,numt,"fix_outflow:tmp");
    memory->create(ntmp,numt,"fix_outflow:ntmp");
    for (i = 0; i < numt; i++){
      tmp[i] = 0.0;
      ntmp[i] = 0.0;
    }
  }

  for (i = 0; i < num_shapes; i++){
    st = face_order[i];
    l = 0;
    for (j=0; j<ndiv[i][0]; j++)
      for (k=0; k<ndiv[i][1]; k++)
        for (m=0; m<2; m++){
          nn = st + l;
          tmp[nn] = num[i][m][j][k];
          tmp[nn+1] = veln[i][m][j][k];
          num[i][m][j][k] = 0.0;
          veln[i][m][j][k] = 0.0;
          l += 2;
        }
  }

  if (bc_comm_rank > -1){
    MPI_Allreduce(tmp,ntmp,numt,MPI_DOUBLE,MPI_SUM,bc_comm);

    if (flux[noutf] > 0.0){
      q_tot = 0;
      l = 0; 
      for (i = 0; i < num_shapes_tot; i++)
        for (j=0; j<glob_order[i]; j++){
          if (ntmp[l] + ntmp[l+2] > 0)
	    q_tot -= (ntmp[l+1] + ntmp[l+3])/(ntmp[l] + ntmp[l+2])*area[i];
          l += 4;
        }
      gamma += coeff_p*(q_tot - flux[noutf])/area_tot;
    } else {
      rho_h = 0;
      l = 0;
      for (i = 0; i < num_shapes_tot; i++)
        for (j=0; j<glob_order[i]; j++){
          rho_h += ntmp[l] + ntmp[l+2];
          l += 4;
        }
      rho_h /= r_shear*area_tot*n_accum; 
      gamma += coeff_p*(rho_target - rho_h);
    }

    for (i = 0; i < num_shapes; i++){
      st = face_order[i];
      l = 0;
      for (j = 0; j < ndiv[i][0]; j++)
	for (k = 0; k < ndiv[i][1]; k++){
	  nn = st + l;
          beta[i][j][k] = 0.0;
          for (ii=qk_max-1; ii>0; ii--)
            save_beta[i][ii][j][k] = save_beta[i][ii-1][j][k];
          if (ntmp[nn+2] > 0 && ntmp[nn] > 0)
            save_beta[i][0][j][k] = coeff_s*(ntmp[nn+3]/ntmp[nn+2] - ntmp[nn+1]/ntmp[nn]);
          else
            save_beta[i][0][j][k] = 0.0;           
          for (ii = 0; ii < qk_max; ii++)
            beta[i][j][k] += save_beta[i][ii][j][k];
	  l += 4;
	}
    }
  }

  if (numt){
    memory->destroy(tmp);
    memory->destroy(ntmp);
  }

  cur_iter++;
  if (cur_iter > mmax_iter) cur_iter = 0;
  n_accum = 0;
}

/*------------------------------------------------------------------------*/

void FixOutflow::set_comm_groups()
{
  int i,j,k,color;
  int *tmp = NULL;  

  noutf = -1;
  memory->create(tmp,num_flux,"fix_outflow:tmp");
  for (i = 0; i < num_flux; ++i)
    tmp[i] = 0;

  for (i = 0; i < num_shapes; ++i){
    k = flux_ind[i];
    tmp[k] = 1; 
  }

  j = 0;
  for (i = 0; i < num_flux; ++i)  
    j += tmp[i];
  if (j > 1) 
    error->one(FLERR,"There are cores, which may participate in multiple outflows! This will not function properly in fix outflow!");

  color = 0;
  for (i = 0; i < num_flux; ++i) 
    if (tmp[i]){
      color = i+1;
      noutf = i;
      break;
    }
 
  MPI_Comm_split(world,color,me,&bc_comm);              // create new communicators
  if (color > 0){
    MPI_Comm_rank(bc_comm,&bc_comm_rank);
    MPI_Comm_size(bc_comm,&bc_comm_size);
  } else{
    bc_comm = MPI_COMM_NULL;
    bc_comm_rank = -1;
    bc_comm_size = 0;
  }  

  memory->destroy(tmp);
}

/*------------------------------------------------------------------------*/

void FixOutflow::reset_target(double rho_new)
{
  int i,j,m,sh,ind;
  double dd[3],u0,u1,u2,zz,dl;

  double **x = atom->x;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double *rho = atom->rho;
  int nall = nlocal + atom->nghost;

  if (bc_comm_rank > -1)
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit_sol)
        for (m = 0; m < tot_shapes[i]; m++){
          sh = ind_shapes[i][m];
          for (j = 0; j < 3; j++)
            dd[j] = x[i][j] - x0[sh][j];
          rot_forward(dd[0],dd[1],dd[2],sh);
          ind = 0;
          if (dd[0] > aa[sh][4]-r_cut_t && dd[0] < aa[sh][5]+r_cut_t && dd[1] > -r_cut_t && dd[1] < aa[sh][2]+r_cut_t && dd[2] >= 0.0 && dd[2] < r_press + r_cut_t){
            u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
            zz = r_cut_t*aa[sh][3]/aa[sh][2];
      
            switch (ptype[sh]){
              case 1 :
                dl = r_cut_t*aa[sh][3]/(aa[sh][6]*aa[sh][1]+aa[sh][7]*aa[sh][2]);
                u2 = dd[1]/(aa[sh][2] + dl*aa[sh][2]/aa[sh][3]);
                u1 = (dd[0] - u2*(aa[sh][1] + dl*aa[sh][1]/aa[sh][3]))/(aa[sh][0] + r_cut_t/aa[sh][6]);
                if (u0 > -zz && u1+u2 < 1.0){
                  rho[i] = rho_new;
                  ind = 1;                  
                }
                break;
              case 2 :
                if (u0 > -zz && u0 < aa[sh][0] + zz){
                  rho[i] = rho_new;
                  ind = 1;
                }
                break;
            }
          }
          if (ind) break;
        }

  if (bc_comm_rank > -1)
    for (i = nlocal; i < nall; i++)
      if (mask[i] & groupbit_sol)
        for (sh = 0; sh < num_shapes; sh++){
          for (j = 0; j < 3; j++)
            dd[j] = x[i][j] - x0[sh][j];
          rot_forward(dd[0],dd[1],dd[2],sh);
          ind = 0;
          if (dd[0] > aa[sh][4]-r_cut_t && dd[0] < aa[sh][5]+r_cut_t && dd[1] > -r_cut_t && dd[1] < aa[sh][2]+r_cut_t && dd[2] >= 0.0 && dd[2] < r_press + r_cut_t){
            u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
            zz = r_cut_t*aa[sh][3]/aa[sh][2];

            switch (ptype[sh]){
              case 1 :
                dl = r_cut_t*aa[sh][3]/(aa[sh][6]*aa[sh][1]+aa[sh][7]*aa[sh][2]);
                u2 = dd[1]/(aa[sh][2] + dl*aa[sh][2]/aa[sh][3]);
                u1 = (dd[0] - u2*(aa[sh][1] + dl*aa[sh][1]/aa[sh][3]))/(aa[sh][0] + r_cut_t/aa[sh][6]);
                if (u0 > -zz && u1+u2 < 1.0){
                  rho[i] = rho_new;
                  ind = 1;
                }
                break;
              case 2 :
                if (u0 > -zz && u0 < aa[sh][0] + zz){
                  rho[i] = rho_new;
                  ind = 1;
                }
                break;
            }
          }
          if (ind) break;
        }
}

/*------------------------------------------------------------------------*/

int FixOutflow::check_shapes(int sh)
{
  int i,ch_ind,nn[3],jj[3];
  double rad, rad_n, rad_t, u0, u1, u2, zz, dl;
  double dd[3],ddx[3];
 
  for (i = 0; i < 3; ++i){                  // setup bins
    ddx[i] = domain->subhi[i] - domain->sublo[i];
    nn[i] = static_cast<int> (ddx[i]/binsize + 3.0);
  }
  if (domain->dimension == 2)
    nn[2] = 1;

  rad = 0.5*sqrt(3.0)*binsize;
  rad_n = rad + r_cut_n; 
  rad_t = rad + r_cut_t;

  ch_ind = 0;
  for (jj[0] = 0; jj[0] < nn[0]; ++jj[0]){             // check if certain bins overlap with a core
    for (jj[1] = 0; jj[1] < nn[1]; ++jj[1]){
      for (jj[2] = 0; jj[2] < nn[2]; ++jj[2]){
        for (i = 0; i < 3; ++i)
          dd[i] = domain->sublo[i] + (jj[i]-0.5)*binsize - x0[sh][i];
        if (domain->dimension == 2)
          dd[2] = domain->sublo[2] - x0[sh][2];
        rot_forward(dd[0],dd[1],dd[2],sh);

        if (dd[0] > aa[sh][4]-rad_t && dd[0] < aa[sh][5]+rad_t && dd[1] > -rad_t && dd[1] < aa[sh][2]+rad_t && dd[2] > -rad && dd[2] < rad_n){
          u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
          zz = rad_t*aa[sh][3]/aa[sh][2];

          switch (ptype[sh]){
            case 1 :
              dl = rad_t*aa[sh][3]/(aa[sh][6]*aa[sh][1]+aa[sh][7]*aa[sh][2]);                             
              u2 = dd[1]/(aa[sh][2] + dl*aa[sh][2]/aa[sh][3]);
              u1 = (dd[0] - u2*(aa[sh][1] + dl*aa[sh][1]/aa[sh][3]))/(aa[sh][0] + rad_t/aa[sh][6]);
              if (u0 > -zz && u1+u2 < 1.0)
                ch_ind = 1;
              break;
            case 2 :
              if (u0 > -zz && u0 < aa[sh][0] + zz)
                ch_ind = 1;
              break;
          }
        }
        if (ch_ind) break;
      }
      if (ch_ind) break;
    }
    if (ch_ind) break;
  }  

  return ch_ind;
}

/*------------------------------------------------------------------------*/

void FixOutflow::setup_rot(int id, double a1[], double a2[])
{
  int i,j;
  double norm[3],mr[3][3],a3[2];
  double nr,rx,zz;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++){
      rot[id][i][j] = 0.0;
      mr[i][j] = 0.0;
    }

  norm[0] = a1[1]*a2[2] - a1[2]*a2[1];
  norm[1] = a1[2]*a2[0] - a1[0]*a2[2];
  norm[2] = a1[0]*a2[1] - a1[1]*a2[0];
  nr = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
  rx = sqrt(norm[0]*norm[0] + norm[1]*norm[1]);
  if (rx > EPS){
    mr[0][0] = norm[0]*norm[2]/nr/rx;
    mr[0][1] = norm[1]*norm[2]/nr/rx;
    mr[0][2] = -rx/nr;
    mr[1][0] = -norm[1]/rx;
    mr[1][1] = norm[0]/rx;
    mr[1][2] = 0.0;
    mr[2][0] = norm[0]/nr;
    mr[2][1] = norm[1]/nr;
    mr[2][2] = norm[2]/nr;
  }
  else {
    mr[0][0] = 1.0;
    mr[1][1] = 1.0;
    mr[2][2] = 1.0;
    if (norm[2] < 0.0){
      mr[1][1] = -1.0;
      mr[2][2] = -1.0;
    }
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
    area[id] = 0.5*nr;
  } else{
    aa[id][5] = MAX(rx,rx + aa[id][1]);
    area[id] = nr / ndiv[id][0] / ndiv[id][1];
  }
}

/* ---------------------------------------------------------------------- */

void FixOutflow::rot_forward(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][0][1]*y1 + rot[id][0][2]*z1;
  y = rot[id][1][0]*x1 + rot[id][1][1]*y1 + rot[id][1][2]*z1;
  z = rot[id][2][0]*x1 + rot[id][2][1]*y1 + rot[id][2][2]*z1;
}

/* ---------------------------------------------------------------------- */

void FixOutflow::rot_back(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][1][0]*y1 + rot[id][2][0]*z1;
  y = rot[id][0][1]*x1 + rot[id][1][1]*y1 + rot[id][2][1]*z1;
  z = rot[id][0][2]*x1 + rot[id][1][2]*y1 + rot[id][2][2]*z1;
}

/*------------------------------------------------------------------------*/

void FixOutflow::shape_decide()
{
  int i,sh,k,l,m,kk,bin;
  double dd[3];
  double u0,u1,u2,zz,dl;

  double **x = atom->x;
  int *mask = atom->mask;

  for (i=0; i<atom->nlocal; i++){
    tot_shapes[i] = 0;
    if (mask[i] & groupbit){
      bin = coord2bin(x[i]);
      if (bin == -1){
        fprintf(stderr, "Negative bin in shape_decide: me=%d, step=" BIGINT_FORMAT ", particle - %f %f %f; box - %f %f %f %f %f %f \n",me,update->ntimestep,x[i][0],x[i][1],x[i][2],bboxlo[0],bboxlo[1],bboxlo[2],bboxhi[0],bboxhi[1],bboxhi[2]);
      } else{
        for (kk = 0; kk < num_bin_shapes[bin]; kk++){
          sh = bin_shapes[kk][bin];
          for (k = 0; k < 3; k++)
            dd[k] = x[i][k] - x0[sh][k];
          rot_forward(dd[0],dd[1],dd[2],sh);

          if (dd[0] > aa[sh][4]-r_cut_t && dd[0] < aa[sh][5]+r_cut_t && dd[1] > -r_cut_t && dd[1] < aa[sh][2]+r_cut_t && dd[2] >= 0.0 && dd[2] < r_cut_n){
            u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
            zz = r_cut_t*aa[sh][3]/aa[sh][2];

            switch (ptype[sh]){
              case 1 :
                dl = r_cut_t*aa[sh][3]/(aa[sh][6]*aa[sh][1]+aa[sh][7]*aa[sh][2]);
                u2 = dd[1]/(aa[sh][2] + dl*aa[sh][2]/aa[sh][3]);
                u1 = (dd[0] - u2*(aa[sh][1] + dl*aa[sh][1]/aa[sh][3]))/(aa[sh][0] + r_cut_t/aa[sh][6]);
                if (u0 > -zz && u1+u2 < 1.0){
                  if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                  ind_shapes[i][tot_shapes[i]] = sh;
                  tot_shapes[i]++;
                }
                break;
              case 2 :
                if (u0 > -zz && u0 < aa[sh][0] + zz){
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
}

/* ---------------------------------------------------------------------- */

void FixOutflow::grow_shape_arrays()
{
  int k, l, m;
  int **ind_tmp = NULL;

  if (max_shapes == 0){
    max_shapes = FACE_INC;
    memory->create(ind_shapes,nnmax,max_shapes,"fix_inflow:ind_shapes");
  } else{
    m = max_shapes;
    memory->create(ind_tmp,nnmax,m,"fix_inflow:ind_tmp");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_tmp[k][l] = ind_shapes[k][l];
    max_shapes += FACE_INC;
    memory->destroy(ind_shapes);
    memory->create(ind_shapes,nnmax,max_shapes,"fix_inflow:ind_shapes");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_shapes[k][l] = ind_tmp[k][l];
    memory->destroy(ind_tmp);
  }
}

/* ---------------------------------------------------------------------- */

void FixOutflow::grow_basic_arrays(int id)
{

  if (id == 0){ 
    max_loc_shapes = NUM1;
    memory->create(x0,max_loc_shapes,3,"fix_outflow:x0");
    memory->create(aa,max_loc_shapes,8,"fix_outflow:aa");
    memory->create(rot,max_loc_shapes,3,3,"fix_outflow:rot");
    memory->create(ndiv,max_loc_shapes,2,"fix_outflow:ndiv");
    memory->create(ptype,max_loc_shapes,"fix_outflow:ptype");
    memory->create(flux_ind,max_loc_shapes,"fix_outflow:flux_ind");
    memory->create(face_order,max_loc_shapes,"fix_outflow:face_order");
    memory->create(shapes_local,max_loc_shapes,"fix_outflow:shapes_local");
  } else{
    max_loc_shapes += NUM1;
    memory->grow(x0,max_loc_shapes,3,"fix_outflow:x0");
    memory->grow(aa,max_loc_shapes,8,"fix_outflow:aa");
    memory->grow(rot,max_loc_shapes,3,3,"fix_outflow:rot");
    memory->grow(ndiv,max_loc_shapes,2,"fix_outflow:ndiv");
    memory->grow(ptype,max_loc_shapes,"fix_outflow:ptype");
    memory->grow(flux_ind,max_loc_shapes,"fix_outflow:flux_ind");
    memory->grow(face_order,max_loc_shapes,"fix_outflow:face_order");
    memory->grow(shapes_local,max_loc_shapes,"fix_outflow:shapes_local");
  }
}

/* ---------------------------------------------------------------------- */

void FixOutflow::setup_bins()
{
  // bbox = size of the bounding box
  // bbox lo/hi = bounding box of my subdomain extended by r_cut_n

  int i;
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

  memory->create(bin_shapes,bin_max_shape,nbin, "fix_inflow:bin_shapes");
  memory->create(num_bin_shapes,nbin,"fix_inflow:num_bin_shapes");
  for (i = 0; i < nbin; ++i)
    num_bin_shapes[i] = 0;
}

/* ---------------------------------------------------------------------- */

int FixOutflow::check_bins(int bin, int sh)
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
  if (dd[0] > aa[sh][4]-rad_t && dd[0] < aa[sh][5]+rad_t && dd[1] > -rad_t && dd[1] < aa[sh][2]+rad_t && dd[2] > -rad && dd[2] < rad_n){
    u0 = dd[0] - dd[1]*aa[sh][1]/aa[sh][2];
    zz = rad_t*aa[sh][3]/aa[sh][2];

    switch (ptype[sh]){
      case 1 :
        dl = rad_t*aa[sh][3]/(aa[sh][6]*aa[sh][1]+aa[sh][7]*aa[sh][2]);
        u2 = dd[1]/(aa[sh][2] + dl*aa[sh][2]/aa[sh][3]);
        u1 = (dd[0] - u2*(aa[sh][1] + dl*aa[sh][1]/aa[sh][3]))/(aa[sh][0] + rad_t/aa[sh][6]);
        if (u0 > -zz && u1+u2 < 1.0)
          ch_ind = 1;
        break;
      case 2 :
        if (u0 > -zz && u0 < aa[sh][0] + zz)
          ch_ind = 1;
        break;
    }
  }

  return ch_ind;
}

/* ---------------------------------------------------------------------- */

int FixOutflow::coord2bin(double *xx)
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

void FixOutflow::write_restart(FILE *fp)
{
  int i,j,k,l,m,st,nn;
  int *nm,*nm1,*prt,*prt1;
  double zz;
  double *gm,*gm1;
  double *tmp,*tmp1,*tmp0;

  nm = nm1 = prt = prt1 = NULL;
  gm = gm1 = tmp = tmp1 = tmp0 = NULL;

  memory->create(nm,num_flux,"fix_outflow:nm");
  memory->create(nm1,num_flux,"fix_outflow:nm1");
  memory->create(gm,num_flux,"fix_outflow:gm");
  memory->create(gm1,num_flux,"fix_outflow:gm1");
  memory->create(prt,num_flux,"fix_outflow:prt");  
  memory->create(prt1,num_flux,"fix_outflow:prt1");  

  for (i = 0; i < num_flux; i++){
    nm[i] = 0;
    nm1[i] = 0;
    gm[i] = 0.0;
    gm1[i] = 0.0;
    prt[i] = 0;
    prt1[i] = 0;
  }
 
  if (bc_comm_rank == 0){
    gm1[noutf] = gamma;
    prt1[noutf] = me;
  }
  MPI_Reduce(gm1,gm,num_flux,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(prt1,prt,num_flux,MPI_INT,MPI_SUM,0,world);

  if (bc_comm_rank > -1)
    for (i = 0; i < num_shapes_tot; i++)
      nm1[noutf] += qk_max*glob_order[i];
  MPI_Reduce(nm1,nm,num_flux,MPI_INT,MPI_MAX,0,world);

  if (bc_comm_rank > -1){
    memory->create(tmp,nm1[noutf],"fix_outflow:tmp");
    memory->create(tmp1,nm1[noutf],"fix_outflow:tmp1");
    for (i = 0; i < nm1[noutf]; i++){
      tmp[i] = -BIG;
      tmp1[i] = -BIG;
    }  

    for (i = 0; i < num_shapes; i++){
      st = static_cast<int>(0.25*face_order[i]*qk_max);
      l = 0;
      for (j=0; j<ndiv[i][0]; j++)
        for (k=0; k<ndiv[i][1]; k++){
          nn = st + l;
          for (m=0; m<qk_max; m++)
            tmp1[nn+m] = save_beta[i][m][j][k];
          l += qk_max;
        }
    } 
    MPI_Reduce(tmp1,tmp,nm1[noutf],MPI_DOUBLE,MPI_MAX,0,bc_comm);
  }

  if (me == 0){
    st = -1;
    nn = 0;
    for (i = 0; i < num_flux; i++){
      if (nm[i] > st)
        st = nm[i];
      nn += nm[i] + 2; 
    }
    memory->create(tmp0,st+3,"fix_outflow:tmp0");
    l = (nn+1)*sizeof(double);
    fwrite(&l,sizeof(int),1,fp);
    zz = static_cast<double> (cur_iter);
    fwrite(&zz,sizeof(double),1,fp);    
  }

  for (i = 0; i < num_flux; i++){
    if (me == 0){
      if (bc_comm_rank == 0 && i == noutf){
        for (j = 0; j < nm[i]; j++)
          tmp0[j+2] = tmp[j]; 
      } else
        MPI_Recv(&tmp0[2],nm[i],MPI_DOUBLE,prt[i],prt[i],world,MPI_STATUS_IGNORE);  

      tmp0[0] = static_cast<double>(nm[i]+2);
      tmp0[1] = gm[i];  
      fwrite(tmp0,sizeof(double),nm[i]+2,fp);           
    }
    
    if (bc_comm_rank == 0)
      if (me != 0 && i == noutf)
        MPI_Send(tmp,nm1[noutf],MPI_DOUBLE,0,me,world);  
  }

  memory->destroy(nm);
  memory->destroy(nm1);
  memory->destroy(gm);
  memory->destroy(gm1);
  memory->destroy(prt);
  memory->destroy(prt1);
  if (bc_comm_rank > -1){  
    memory->destroy(tmp);
    memory->destroy(tmp1);
  }
  if (me == 0)
    memory->destroy(tmp0);

  /*char fname[FILENAME_MAX];
  FILE *f_write;
  sprintf(fname,"write_bla_%d.dat",me);
  f_write = fopen(fname,"w");
  fprintf(f_write,"%lf %d \n",gamma,cur_iter);
  for (i=0; i<num_shapes; i++)
    for (j=0; j<ndiv[i][0]; j++)
      for (k=0; k<ndiv[i][1]; k++){
        fprintf(f_write,"%lf \n",beta[i][j][k]);
        for (l=0; l<qk_max; l++)
          fprintf(f_write,"%lf ",save_beta[i][l][j][k]);
        fprintf(f_write,"\n");
      }
  fclose(f_write);*/
}

/* ---------------------------------------------------------------------- */

void FixOutflow::restart(char *buf)
{
  int i,j,k,l,m,st,nn,sloc,ii;
  double *list = (double *) buf;

  cur_iter = static_cast<int> (list[0]);
  list++;
  if (bc_comm_rank > -1){
    st = 0; 
    for (i=0; i<noutf; i++)      
      st += static_cast<int>(list[st]); 

    gamma = list[st+1];
    st += 2;

    for (i = 0; i < num_shapes; i++){
      sloc = static_cast<int>(0.25*face_order[i]*qk_max);
      l = 0;
      for (j=0; j<ndiv[i][0]; j++)
        for (k=0; k<ndiv[i][1]; k++){
          nn = st + sloc + l;
          for (m=0; m<qk_max; m++)
            save_beta[i][m][j][k] = list[nn+m];
          l += qk_max;
          beta[i][j][k] = 0.0;
          for (ii = 0; ii < qk_max; ii++)
            beta[i][j][k] += save_beta[i][ii][j][k]; 
        }
    }    
  } else
    gamma = 1.0;  

  read_restart_ind = 1;

  /*char fname[FILENAME_MAX];
  FILE *f_write;
  sprintf(fname,"read_bla_%d.dat",me);
  f_write = fopen(fname,"w");
  fprintf(f_write,"%lf %d \n",gamma,cur_iter);
  for (i=0; i<num_shapes; i++)
    for (j=0; j<ndiv[i][0]; j++)
      for (k=0; k<ndiv[i][1]; k++){
        fprintf(f_write,"%lf \n",beta[i][j][k]);
        for (l=0; l<qk_max; l++)
          fprintf(f_write,"%lf ",save_beta[i][l][j][k]);
        fprintf(f_write,"\n");
      }
  fclose(f_write);*/
}

/* ---------------------------------------------------------------------- */
