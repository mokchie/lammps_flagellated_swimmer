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
#include "fix_solid_bound.h"
#include "atom.h"
#include "comm.h"
#include "group.h"
#include "error.h"
#include "memory.h"
#include "update.h"
#include "neighbor.h"
#include "force.h"
#include "domain.h"
#include "universe.h"

#define  NUM 10000
#define  NUM1 10
#define  FACE_INC 5
#define  BUFMIN 1000
#define  EPS 1e-6
#define  DR 0.01

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSolidBound::FixSolidBound(LAMMPS *lmp, int narg, char **arg) : Fix(lmp,narg, arg)
{
  int i, j, l, ttyp, igroup, nms, nw_max; 
  double *rdata;
  double dummy,a1[3],a2[3],rr,wd; 
  char grp[50];
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  char fname1[FILENAME_MAX];
  FILE *f_read, *f_read1;

  groupbit_s = groupbit_p = 0;
  f_press = NULL;
  weight = NULL;
  dr_inv = 1.0/DR;
  if (narg != 4) error->all(FLERR,"Illegal fix solid/bound command");
  sprintf(fname,arg[3]);

 //read in plane.dat file except shapes
  nms = NUM;
  cur_iter = 1;
  memory->create(rdata,nms,"fix_solid_bound:rdata");
  max_count = 3;
  ind_read_shear = 0;
  ind_write_shear = 0;

  if (comm->me == 0){
    l = 0;
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open input solid-boundary file");
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&num_shapes_tot);
    rdata[l++] = static_cast<double> (num_shapes_tot);
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %d %lf",&r_cut_n,&r_cut_t,&mirror,&binsize);
    rdata[l++] = r_cut_n;
    rdata[l++] = r_cut_t;
    rdata[l++] = static_cast<double> (mirror);
    rdata[l++] = binsize;

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&ind_shear);
    rdata[l++] = static_cast<double> (ind_shear);
    if (ind_shear){
      sscanf(buf,"%d %lf %lf %lf %d %d %d %d %d %d %s",&ind_shear,&r_shear,&coeff,&power,&n_per,&iter,&mmax_iter,&ind_read_shear,&ind_write_shear,&s_apply,&grp[0]);
      rdata[l++] = r_shear;
      rdata[l++] = coeff;
      rdata[l++] = power;
      rdata[l++] = static_cast<double> (n_per);
      rdata[l++] = static_cast<double> (iter);
      rdata[l++] = static_cast<double> (mmax_iter);
      rdata[l++] = static_cast<double> (ind_read_shear);
      rdata[l++] = static_cast<double> (ind_write_shear);
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
      if(f_read1 == (FILE*) NULL)
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
      if (l + 16 > nms){
        nms += NUM;
        memory->grow(rdata,nms,"fix_solid_bound:rdata");
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
    memory->grow(rdata,l,"fix_solid_bound:rdata");
  MPI_Bcast(rdata,l,MPI_DOUBLE,0,world);

  l = 0;  
  num_shapes_tot = static_cast<int> (rdata[l++]);
  r_cut_n = rdata[l++];
  r_cut_t = rdata[l++]; 
  mirror = static_cast<int> (rdata[l++]);  
  binsize = rdata[l++]; 

  ind_shear = static_cast<int> (rdata[l++]);
  if (ind_shear){
    r_shear = rdata[l++];
    coeff = rdata[l++];
    power = rdata[l++];
    n_per = static_cast<int> (rdata[l++]);
    iter = static_cast<int> (rdata[l++]);
    mmax_iter = static_cast<int> (rdata[l++]);
    ind_read_shear = static_cast<int> (rdata[l++]);
    ind_write_shear = static_cast<int> (rdata[l++]); 
    s_apply = static_cast<int> (rdata[l++]);
    groupbit_s = static_cast<int> (rdata[l++]);
    
    nw_max = static_cast<int> (r_shear*dr_inv) + 1;
    memory->create(weight,nw_max,"fix_solid_bound:weight");
    for (i = 0; i < nw_max; i++){
      rr = i*DR;
      if (rr > r_shear)
        wd = 0.0;
      else  
        wd = 1.0 - rr/r_shear;
      weight[i] = pow(wd,power);
    }
  }      
  ind_press = static_cast<int> (rdata[l++]);
  if (ind_press){       
    n_press = static_cast<int> (rdata[l++]);
    r_press = rdata[l++];
    p_apply = static_cast<int> (rdata[l++]);
    groupbit_p = static_cast<int> (rdata[l++]);
    memory->create(f_press,n_press,"fix_solid_bound:f_press");
    for (j=0; j<n_press; j++)
      f_press[j] = rdata[l++];
  }

  numt = 0;
  numt_s = 0;
  num_shapes = 0;
  max_shapes = 0;
  nnmax = atom->nlocal;
  if (nnmax == 0) nnmax = 1;
  bin_max_shape = NUM1;
  memory->create(x0,num_shapes_tot,3,"fix_solid_bound:x0");
  memory->create(aa,num_shapes_tot,4,"fix_solid_bound:aa");
  memory->create(vel,num_shapes_tot,3,"fix_solid_bound:vel");
  memory->create(rot,num_shapes_tot,3,3,"fix_solid_bound:rot");
  memory->create(ndiv,num_shapes_tot,2,"fix_solid_bound:ndiv");
  memory->create(ptype,num_shapes_tot,"fix_solid_bound:ptype");
  memory->create(refl,num_shapes_tot,"fix_solid_bound:refl");
  memory->create(face_order,num_shapes_tot,"fix_solid_bound:face_order");
  memory->create(shapes_local,num_shapes_tot,"fix_solid_bound:shapes_local");
  memory->create(shapes_global_to_local,num_shapes_tot,"fix_solid_bound:shapes_global_to_local");
  bin_shapes = NULL; num_bin_shapes = NULL;
  ind_shapes = NULL; tot_shapes = NULL;
  memory->create(tot_shapes,nnmax,"fix_solid_bound:tot_shapes");

 //types of boundarys: (1) triangle, (2) parallelogram, (3) cylinder , (4) sphere
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
      memory->create(velx[i],2*n_per,ndiv[j][0],ndiv[j][1],"fix_solid_bound:velx[i]");
      memory->create(vely[i],2*n_per,ndiv[j][0],ndiv[j][1],"fix_solid_bound:vely[i]");
      memory->create(velz[i],2*n_per,ndiv[j][0],ndiv[j][1],"fix_solid_bound:velz[i]");
      memory->create(num[i],2*n_per,ndiv[j][0],ndiv[j][1],"fix_solid_bound:num[i]");
      memory->create(fsx[i],2,ndiv[j][0],ndiv[j][1],"fix_solid_bound:fsx[i]");
      memory->create(fsy[i],2,ndiv[j][0],ndiv[j][1],"fix_solid_bound:fsy[i]");
      memory->create(fsz[i],2,ndiv[j][0],ndiv[j][1],"fix_solid_bound:fsz[i]");
      numt_loc += 6*ndiv[j][0]*ndiv[j][1];
    }
    numt_loc += num_shapes;
    if (ind_read_shear)
      read_shear_forces();
  } else{
    velx = vely = velz = NULL;
    num = NULL;
    fsx = fsy = fsz = NULL;
  } 
}

/* ---------------------------------------------------------------------- */

FixSolidBound::~FixSolidBound()
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
  memory->destroy(bin_shapes);
  memory->destroy(num_bin_shapes);
}

/* ---------------------------------------------------------------------- */

int FixSolidBound::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSolidBound::setup(int vflag)
{
  int i,j,k,l,ii;
    
  n_accum = 0;
  // zero out initial arrays
  if (ind_shear && num_shapes)
    for(i = 0; i < num_shapes; i++){
      ii = shapes_local[i];
      for(k = 0; k < ndiv[ii][0]; k++)
        for(l = 0; l < ndiv[ii][1]; l++){
          for(j = 0; j < 2*n_per; j++){
            velx[i][j][k][l] = 0.0;
            vely[i][j][k][l] = 0.0;
            velz[i][j][k][l] = 0.0;
            num[i][j][k][l] = 0.0;
          }
          if (!ind_read_shear)
            for(j = 0; j < 2; j++){
              fsx[i][j][k][l] = 0.0;
              fsy[i][j][k][l] = 0.0;
              fsz[i][j][k][l] = 0.0;
            }
        }
    }  

  // setup bins in the domain
  setup_bins();
  for(i = 0; i < nbin; ++i)
    for(j = 0; j < num_shapes; ++j){
       k = shapes_local[j];
       if (check_bins(i,k)){
         bin_shapes[num_bin_shapes[i]][i] = k;
         num_bin_shapes[i]++;
         if (num_bin_shapes[i] == bin_max_shape){
           bin_max_shape += NUM1;
           memory->grow(bin_shapes,bin_max_shape,nbin,"fix_solid_bound:bin_shapes");
         }
       }
    }

  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixSolidBound::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixSolidBound::post_integrate()
{
  int i,j,k,kk,ind,cond,i_x[NUM],cc,bounce,m,cplane,ccount;
  double dl[3], t_x[NUM], dd[3],xp[3],xh[3],vh[3],vv[3],norm[3];
  double tt,dot,d1,d2,dtt,u0,u1; 
  double dtv = update->dt; 

  double **x = atom->x;
  double **v = atom->v;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
   
  for(i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      cond = 1;
      ccount = 0;
      dtt = dtv;
      cplane = -1; 
      while (cond && ccount < max_count){
        ind = 0;
        for(m = 0; m < tot_shapes[i]; m++){
          kk = ind_shapes[i][m];
          if (kk != cplane){
            for(j = 0; j < 3; j++){
              xh[j] = x[i][j] - x0[kk][j];
              vh[j] = v[i][j];
	    }  
            rot_forward(xh[0],xh[1],xh[2],kk);
            rot_forward(vh[0],vh[1],vh[2],kk);  
            for(j = 0; j < 3; j++){
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
	        ind++;
	    }
	  }
        }
     
        if (ind > 0){
          for(j = 1; j < ind; j++){          
            tt = 2.0; 
            for(k = 0; k < ind-j+1; k++)
              if (t_x[k]<tt) {
                tt = t_x[k];
                kk = k;      
	      }
            cc = i_x[kk];
            t_x[kk] = t_x[ind-j];
            i_x[kk] = i_x[ind-j];
            t_x[ind-j] = tt;
            i_x[ind-j] = cc;   
          }
	  while(ind){
            bounce = 1;
            tt = t_x[ind - 1];
	    kk = i_x[ind - 1];
            for(k = 0; k < 3; k++){
              xp[k] = x[i][k] - v[i][k]*dtt - x0[kk][k];
              vh[k] = v[i][k];
	    }  
            rot_forward(xp[0],xp[1],xp[2],kk);
            rot_forward(vh[0],vh[1],vh[2],kk);
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
                norm[0] = dd[0]/aa[kk][1];
                norm[1] = dd[1]/aa[kk][1]; 
                norm[2] = 0.0;
                break;
              case 4 :
                for(k = 0; k < 3; k++)
                  norm[k] = dd[k]/aa[kk][0];          
                break;
	    }
            
	    if (bounce){
              ccount++;
              cplane = kk; 
              if (mirror){
                if (ptype[kk]<3){
                  vh[2] = -vh[2];
		} else{
                  d1 = 2.0 * (vh[0]*norm[0] + vh[1]*norm[1] + vh[2]*norm[2]);
                  for (k = 0; k < 3; k++) 
                    vh[k] += -d1*norm[k];     
		}
	      }
              else {
                for (k = 0; k < 3; k++)
                  vh[k] = -vh[k] + 2.0*vel[kk][k];     
	      }
                
              for (k = 0; k < 3; k++){
                xp[k] = dd[k];      
                xh[k] = xp[k] + (1.0-tt)*dtt*vh[k];
	      }
              rot_back(xh[0],xh[1],xh[2],kk);
              rot_back(vh[0],vh[1],vh[2],kk);
              for (k = 0; k < 3; k++){
                x[i][k] = xh[k] + x0[kk][k];
                v[i][k] = vh[k];
	      }  
              dtt = (1.0-tt)*dtt;  
              ind = 0; 
	    } //if(bound)
	    else{
              ind--;
              if (ind == 0) 
                cond = 0;
	    }
	  } //while(ind)
	} //if(ind>0)
        else{
          cond = 0;
        }
      } //while(cond)
    } //if(mask && groubit) 
    //for(i<nlocal)
}

/* ---------------------------------------------------------------------- */

void FixSolidBound::post_force(int vflag)
{
  int i,m,k,kk,j,ix,iy,iz,kk1;
  double xh[3],vh[3],ff[3],u0,u1,weight_h,rr,theta,theta1,hh,rr1;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  int step_t = update->ntimestep;

  if (atom->nlocal > nnmax){
    while (nnmax < atom->nlocal)               
      nnmax += BUFMIN;
    if (max_shapes)
      memory->grow(ind_shapes,nnmax,max_shapes,"fix_solid_bound:ind_shapes");
    memory->grow(tot_shapes,nnmax,"fix_solid_bound:tot_shapes");
  }  

  if (neighbor->ago == 0)
    shape_decide();

  n_accum++;
  if (ind_shear || ind_press)
    for(i = 0; i < nlocal; i++)
      if (mask[i] & groupbit_s || mask[i] & groupbit_p)
        for(m = 0; m < tot_shapes[i]; m++){
          kk = ind_shapes[i][m];
          kk1 = shapes_global_to_local[kk];
          for(j = 0; j < 3; j++){
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
                    for(j = 0; j < 3; j++)
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
                    for(j = 0; j < 3; j++)
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
                    for(j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (xh[2]>-r_press && xh[2]<0.0 && (p_apply == 2 || p_apply == 0)){
                    iz = static_cast<int> (-xh[2]*n_press/r_press);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = -f_press[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for(j = 0; j < 3; j++)
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
                    for(j = 0; j < 3; j++)
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
                    for(j = 0; j < 3; j++)
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
                    for(j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (xh[2]>-r_press && xh[2]<0.0 && (p_apply == 2 || p_apply == 0)){
                    iz = static_cast<int> (-xh[2]*n_press/r_press);
                    ff[0] = 0.0;
                    ff[1] = 0.0;
                    ff[2] = -f_press[iz];
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for(j = 0; j < 3; j++)
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
                    for(j = 0; j < 3; j++)
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
                    for(j = 0; j < 3; j++)
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
                    for(j = 0; j < 3; j++)
                      f[i][j] += ff[j]; 
		  } 
                  if (hh>-r_press && hh<0.0 && (p_apply == 2 || p_apply == 0)){
                    iz = static_cast<int> (-hh*n_press/r_press);
                    ff[0] = -f_press[iz]*xh[0]/rr;
                    ff[1] = -f_press[iz]*xh[1]/rr;
                    ff[2] = 0.0;
                    rot_back(ff[0],ff[1],ff[2],kk);
                    for(j = 0; j < 3; j++)
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
                  for(j = 0; j < 3; j++)
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
                  for(j = 0; j < 3; j++)
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
                  for(j = 0; j < 3; j++)
                    f[i][j] += ff[j]; 
	        } 
                if (hh>-r_press && hh<0.0 && (p_apply == 2 || p_apply == 0)){
                  iz = static_cast<int> (-hh*n_press/r_press);
                  ff[0] = -f_press[iz]*xh[0]/rr;
                  ff[1] = -f_press[iz]*xh[1]/rr;
                  ff[2] = -f_press[iz]*xh[2]/rr;
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for(j = 0; j < 3; j++)
                    f[i][j] += ff[j]; 
		}
	      }       
              break;
	  }       
	}

  if (ind_shear && num_shapes_tot)
    if (cur_iter && step_t%iter == 0  && n_accum > 0.6*iter)
      recalc_force();
}

/* ---------------------------------------------------------------------- */

int FixSolidBound::check_shapes(int sh)
{
  int i, ch_ind;
  double rad, u0, u1;
  double dd[3], xd[2];

  rad = 0;
  ch_ind = 0;
  for (i = 0; i < 3; ++i){
    rad += (domain->subhi[i]-domain->sublo[i])*(domain->subhi[i]-domain->sublo[i]);
    dd[i] = 0.5*(domain->subhi[i]+domain->sublo[i]) - x0[sh][i];
  }
  rad  = 0.5*sqrt(rad) + r_cut_n;

  rot_forward(dd[0],dd[1],dd[2],sh);
  switch (ptype[sh]){
    case 1 :
      u1 = dd[1]/aa[sh][2];
      u0 = (dd[0]-u1*aa[sh][1])/aa[sh][0];
      if (u0 > -rad/aa[sh][0] && u1 > -rad/aa[sh][3] && u0+u1 < 1.0 + rad/aa[sh][0] + rad/aa[sh][3] && dd[2] < rad && dd[2] > -rad)
        ch_ind = 1;
      break;
    case 2 :
      u1 = dd[1]/aa[sh][2];
      u0 = (dd[0]-u1*aa[sh][1])/aa[sh][0];
      if (u0 > -rad/aa[sh][0] && u1 > -rad/aa[sh][3] && u0 < 1.0 + rad/aa[sh][0] && u1 < 1.0 + rad/aa[sh][3] && dd[2] < rad && dd[2] > -rad)
        ch_ind = 1;
      break;
    case 3 :
      xd[0] = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
      if (xd[0] > aa[sh][1]-rad && xd[0] < aa[sh][1]+rad && dd[2] > -rad && dd[2] < aa[sh][0]+rad)
        ch_ind = 1;
      break;
    case 4 :
      xd[0] = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
      if (xd[0] > aa[sh][0]-rad && xd[0] < aa[sh][0]+rad)
        ch_ind = 1;
      break;
  }

  return ch_ind;
}     

/* ---------------------------------------------------------------------- */

void FixSolidBound::grow_shape_arrays()
{
  int k, l, m;
  int **ind_tmp;

  if (max_shapes == 0){
    max_shapes = FACE_INC;
    memory->create(ind_shapes,nnmax,max_shapes,"fix_solid_bound:ind_shapes");
  } else{
    m = max_shapes;
    memory->create(ind_tmp,nnmax,m,"fix_solid_bound:ind_tmp");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_tmp[k][l] = ind_shapes[k][l];
    max_shapes += FACE_INC;
    memory->destroy(ind_shapes);
    memory->create(ind_shapes,nnmax,max_shapes,"fix_solid_bound:ind_shapes");
    for (k=0; k<nnmax; k++)
      for (l=0; l<m; l++)
        ind_shapes[k][l] = ind_tmp[k][l];
    memory->destroy(ind_tmp);
  }
}

/* ---------------------------------------------------------------------- */

void FixSolidBound::setup_rot(int id, double a1[], double a2[])
{
  int i,j;
  double norm[3],mr[3][3],a3[2];
  double nr,rx;

  for(i = 0; i < 3; i++)
    for(j = 0; j < 3; j++){
      rot[id][i][j] = 0.0;
      mr[i][j] = 0.0;  
    }

  if (ptype[id] < 3){  
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
      if (norm[2] < 0.0)
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
  } else if (ptype[id] == 3){
    norm[0] = a1[0];
    norm[1] = a1[1];
    norm[2] = a1[2];
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
    aa[id][0] = nr;
  } else {
    for(i = 0; i < 3; i++)
      rot[id][i][i] = 1.0; 
  }
} 

/* ---------------------------------------------------------------------- */

void FixSolidBound::rot_forward(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][0][1]*y1 + rot[id][0][2]*z1;
  y = rot[id][1][0]*x1 + rot[id][1][1]*y1 + rot[id][1][2]*z1;
  z = rot[id][2][0]*x1 + rot[id][2][1]*y1 + rot[id][2][2]*z1;
}

/* ---------------------------------------------------------------------- */

void FixSolidBound::rot_back(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][1][0]*y1 + rot[id][2][0]*z1;
  y = rot[id][0][1]*x1 + rot[id][1][1]*y1 + rot[id][2][1]*z1;
  z = rot[id][0][2]*x1 + rot[id][1][2]*y1 + rot[id][2][2]*z1;
}

/* ---------------------------------------------------------------------- */

void FixSolidBound::shape_decide()
{
  int i,j,k,l,m,kk,bin;
  double dd[3],xd[2];
  double u0,u1;

  double **x = atom->x;
  int *mask = atom->mask;

  for (i=0; i<atom->nlocal; i++){
    tot_shapes[i] = 0;
    if (mask[i] & groupbit){
      bin = coord2bin(x[i]);
      if (bin == -1){
        fprintf(stderr, "Negative bin in shape_decide: me=%d, step=" BIGINT_FORMAT ", particle - %f %f %f; box - %f %f %f %f %f %f \n",comm->me,update->ntimestep,x[i][0],x[i][1],x[i][2],bboxlo[0],bboxlo[1],bboxlo[2],bboxhi[0],bboxhi[1],bboxhi[2]);
      }else{
        for (kk = 0; kk < num_bin_shapes[bin]; kk++){
          j = bin_shapes[kk][bin];
          for (k = 0; k < 3; k++)
            dd[k] = x[i][k] - x0[j][k];
          rot_forward(dd[0],dd[1],dd[2],j);
          switch (ptype[j]){
            case 1 :
              u1 = dd[1]/aa[j][2];
              u0 = (dd[0]-u1*aa[j][1])/aa[j][0];
              if (u0 > -r_cut_t/aa[j][0] && u1 > -r_cut_t/aa[j][3] && u0+u1 < 1.0 + r_cut_t/aa[j][0] + r_cut_t/aa[j][3] && dd[2] < r_cut_n && dd[2] > -r_cut_n){
                if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                ind_shapes[i][tot_shapes[i]] = j;
                tot_shapes[i]++;
              }
              break;
            case 2 :
              u1 = dd[1]/aa[j][2];
              u0 = (dd[0]-u1*aa[j][1])/aa[j][0];
              if (u0 > -r_cut_t/aa[j][0] && u1 > -r_cut_t/aa[j][3] && u0 < 1.0 + r_cut_t/aa[j][0] && u1 < 1.0 + r_cut_t/aa[j][3] && dd[2] < r_cut_n && dd[2] > -r_cut_n){
                if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                ind_shapes[i][tot_shapes[i]] = j;
                tot_shapes[i]++;
              }
              break;
            case 3 :
              xd[0] = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
              if (xd[0] > aa[j][1]-r_cut_n && xd[0] < aa[j][1]+r_cut_n && dd[2] > -r_cut_t && dd[2] < aa[j][0]+r_cut_t){
                if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                ind_shapes[i][tot_shapes[i]] = j;
                tot_shapes[i]++;
              }
              break;
            case 4 :
              xd[0] = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
              if (xd[0] > aa[j][0]-r_cut_n && xd[0] < aa[j][0]+r_cut_n){
                if (tot_shapes[i] == max_shapes) grow_shape_arrays();
                ind_shapes[i][tot_shapes[i]] = j;
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

void FixSolidBound::setup_bins()
{
  // bbox = size of the bounding box
  // bbox lo/hi = bounding box of my subdomain extended by r_cut_n

  int i;
  double bbox[3];
  double r_extent = r_cut_n;

  bboxlo[0] = domain->sublo[0] - r_extent;
  bboxlo[1] = domain->sublo[1] - r_extent;
  bboxlo[2] = domain->sublo[2] - r_extent;
  bboxhi[0] = domain->subhi[0] + r_extent;
  bboxhi[1] = domain->subhi[1] + r_extent;
  bboxhi[2] = domain->subhi[2] + r_extent;

  bbox[0] = bboxhi[0] - bboxlo[0];
  bbox[1] = bboxhi[1] - bboxlo[1];
  bbox[2] = bboxhi[2] - bboxlo[2];

  //printf("solid_bound: me=%d; r_ext:%f; domain:%f %f %f %f %f %f; bsubbox:%f %f %f %f %f %f \n",comm->me,r_extent,domain->sublo[0],domain->sublo[1],domain->sublo[2],domain->subhi[0],domain->subhi[1],domain->subhi[2],bboxlo[0],bboxlo[1],bboxlo[2],bboxhi[0],bboxhi[1],bboxhi[2]);

  // optimal bin size is roughly 1/2 r_cut_n

  if (binsize <= 0.0)
    binsize = 0.5*r_cut_n;
  binsizeinv = 1.0/binsize;

  nbinx = static_cast<int> (bbox[0]*binsizeinv + 1.0);
  nbiny = static_cast<int> (bbox[1]*binsizeinv + 1.0);
  if (domain->dimension == 3) nbinz = static_cast<int> (bbox[2]*binsizeinv + 1.0);
  else { 
    nbinz = 1;
    bboxlo[2] = - 0.5*binsize + EPS;
    bboxhi[2] = 0.5*binsize - EPS;
  }
  nbin = nbinx*nbiny*nbinz;

  memory->create(bin_shapes,bin_max_shape,nbin, "fix_solid_bound:bin_shapes");
  memory->create(num_bin_shapes,nbin,"fix_solid_bound:num_bin_shapes");
  for(i = 0; i < nbin; ++i)
    num_bin_shapes[i] = 0;
}

/* ---------------------------------------------------------------------- */

int FixSolidBound::check_bins(int bin, int sh)
{
  int ix, iy, iz, ch_ind;
  double rad, u0, u1;
  double dd[3], xd[2];

  ch_ind = 0;
  rad = r_cut_n + 0.5*sqrt(3.0)*binsize;
  iz = static_cast<int> (bin/nbinx/nbiny);
  iy = static_cast<int> ((bin-iz*nbinx*nbiny)/nbinx);
  ix = bin-iz*nbinx*nbiny - iy*nbinx;
  dd[0] = bboxlo[0] + (ix + 0.5)*binsize - x0[sh][0];
  dd[1] = bboxlo[1] + (iy + 0.5)*binsize - x0[sh][1];
  dd[2] = bboxlo[2] + (iz + 0.5)*binsize - x0[sh][2];

  rot_forward(dd[0],dd[1],dd[2],sh);
  switch (ptype[sh]){
    case 1 :
      u1 = dd[1]/aa[sh][2];
      u0 = (dd[0]-u1*aa[sh][1])/aa[sh][0];
      if (u0 > -rad/aa[sh][0] && u1 > -rad/aa[sh][3] && u0+u1 < 1.0 + rad/aa[sh][0] + rad/aa[sh][3] && dd[2] < rad && dd[2] > -rad)
        ch_ind = 1;
      break;
    case 2 :
      u1 = dd[1]/aa[sh][2];
      u0 = (dd[0]-u1*aa[sh][1])/aa[sh][0];
      if (u0 > -rad/aa[sh][0] && u1 > -rad/aa[sh][3] && u0 < 1.0 + rad/aa[sh][0] && u1 < 1.0 + rad/aa[sh][3] && dd[2] < rad && dd[2] > -rad)
        ch_ind = 1;
      break;
    case 3 :
      xd[0] = sqrt(dd[0]*dd[0] + dd[1]*dd[1]);
      if (xd[0] > aa[sh][1]-rad && xd[0] < aa[sh][1]+rad && dd[2] > -rad && dd[2] < aa[sh][0]+rad)
        ch_ind = 1;
      break;
    case 4 :
      xd[0] = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
      if (xd[0] > aa[sh][0]-rad && xd[0] < aa[sh][0]+rad)
        ch_ind = 1;
      break;
  }

  return ch_ind;
}

/* ---------------------------------------------------------------------- */

int FixSolidBound::coord2bin(double *xx)
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


void FixSolidBound::recalc_force()
{
  int i,j,k,l,m,nn,st,ii;
  double tmp[numt],ntmp[numt],vv;

  n_accum = 0;
  for(i = 0; i < numt; i++){
    tmp[i] = 0.0;
    ntmp[i] = 0.0;
  }

  for(i = 0; i < num_shapes; i++){
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

  MPI_Allreduce(&tmp,&ntmp,numt,MPI_DOUBLE,MPI_SUM,world);

  for(i = 0; i < num_shapes; i++){
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

  cur_iter++;
  if (cur_iter > mmax_iter) cur_iter = 0;

  if (ind_write_shear)
    write_shear_forces();

  /*char fname[FILENAME_MAX];
  FILE *f_write;
  sprintf(fname,"write_bla_%d.dat",comm->me);
  f_write = fopen(fname,"w");
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

void FixSolidBound::read_shear_forces()
{
  int i,j,k,l,m;
  char fname[FILENAME_MAX];
  FILE *f_read;
  char buf[BUFSIZ];
  double tmp[numt_s*3];

  if (comm->me == 0){
    sprintf(fname,"shear_forces%d.dat",universe->iworld);
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open shear_forces.dat file");
    l = 0;
    for (i=0; i<num_shapes_tot; i++)
      for (j=0; j<ndiv[i][0]; j++)
        for (k=0; k<ndiv[i][1]; k++)
          for (m=0; m<2; m++){
            fgets(buf,BUFSIZ,f_read);
            sscanf(buf,"%lf %lf %lf",&tmp[l],&tmp[l+1],&tmp[l+2]);
            l += 3;
          }
    fclose(f_read);
  }
  MPI_Bcast(&tmp[0],numt_s*3,MPI_DOUBLE,0,world);

  l = 0;
  for (i=0; i<num_shapes_tot; i++)
    for (j=0; j<ndiv[i][0]; j++)
      for (k=0; k<ndiv[i][1]; k++){
        m = shapes_global_to_local[i];
        if (m > -1){
          fsx[m][0][j][k] = tmp[l];
          fsy[m][0][j][k] = tmp[l+1];
          fsz[m][0][j][k] = tmp[l+2];
          fsx[m][1][j][k] = tmp[l+3];
          fsy[m][1][j][k] = tmp[l+4];
          fsz[m][1][j][k] = tmp[l+5];
        }
        l += 6;
      }

  /* FILE *f_write;
  sprintf(fname,"read_bla_%d.dat",comm->me);
  f_write = fopen(fname,"w");
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

/* ------------------------------------------------------------------------- */

void FixSolidBound::write_shear_forces()
{
  int i,j,k,l,m,nel_tot,offset,ind_sh;
  char fname[FILENAME_MAX];
  FILE *f_write;
  int *displs, *rcounts;

  if (numt_loc)
    offset = numt_loc;
  else
    offset = 1;
  double tmp[offset];

  nel_tot = 1;
  memory->create(rcounts,comm->nprocs,"fix_solid_bound:rcounts");
  memory->create(displs,comm->nprocs,"fix_solid_bound:displs");
  if (comm->nprocs > 1){
    MPI_Gather(&numt_loc, 1, MPI_INT, rcounts, 1, MPI_INT, 0, world);
    if (comm->me == 0){
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
    error->warning(FLERR,"Something might be wrong: numt_loc and l are not equal in write_shear_forces()!");

  double tmph[nel_tot];
  if (comm->nprocs > 1){
    MPI_Gatherv(&tmp[0], l, MPI_DOUBLE, &tmph[0], rcounts, displs, MPI_DOUBLE, 0, world);
  } else{
    for (i=0; i<nel_tot; i++)
      tmph[i] = tmp[i];
  }

  if (comm->me == 0){
    sprintf(fname,"dump_shear_forces%d.dat",universe->iworld);
    f_write = fopen(fname,"w");
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
            fprintf(f_write,"%lf %lf %lf \n",tmph[l], tmph[l+1], tmph[l+2]);
            fprintf(f_write,"%lf %lf %lf \n",tmph[l+3], tmph[l+4], tmph[l+5]);
            l += 6;
          }
      } else{
        for (j=0; j<ndiv[i][0]; j++)
          for (k=0; k<ndiv[i][1]; k++){
            fprintf(f_write,"0.0 0.0 0.0 \n");
            fprintf(f_write,"0.0 0.0 0.0 \n");
          }
      }
    }
    fclose(f_write);
  }

  memory->destroy(rcounts);
  memory->destroy(displs);
}

/* ------------------------------------------------------------------------- */

