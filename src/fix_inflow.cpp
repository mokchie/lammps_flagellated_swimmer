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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fix_inflow.h"
#include "random_mars.h"
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

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixInflow::FixInflow(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  int i, j, k, ttyp, igroup,l,nms,ndv1,ndv2;
  double *rdata = NULL;
  double dummy,a1[3],a2[3],dens;
  char grp[50];
  char buf[BUFSIZ];
  char fname[FILENAME_MAX],fname1[FILENAME_MAX];
  FILE *f_read,*f_read1;

  ptype = shapes_local = tot_shapes = num_bin_shapes = at_type = at_groupbit = NULL;
  ndiv = ind_shapes = bin_shapes = info_flow = NULL;
  rdata = f_press = at_dens = vls = NULL;
  area = rot = NULL;
  vel = NULL;
  x0 = aa = ncount = NULL;

  me = comm->me;
  nms = NUM;
  groupbit_p = 0;
  max_count = 3;
  ranmars0 = new RanMars(lmp, 25+me);// 4 random number
  if (narg != 4) error->all(FLERR,"Illegal fix_inflow command");
  sprintf(fname,arg[3]);

  if (me == 0){
    l = 0;
    f_read = fopen(fname,"r");
    if (f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open input boundary file from fix_inflow.");
    memory->create(rdata,nms,"fix_inflow:rdata");
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d %d",&num_shapes_tot,&num_at_types);
    rdata[l++] = static_cast<double> (num_shapes_tot);
    rdata[l++] = static_cast<double> (num_at_types); 

    for (i=0; i<num_at_types; i++){
      fgets(buf,BUFSIZ,f_read);  
      sscanf(buf,"%d %lf %s",&ttyp,&dens,grp);
      rdata[l++] = static_cast<double> (ttyp);
      rdata[l++] = dens;
      igroup = group->find(grp);
      if (igroup == -1) error->one(FLERR,"Group ID does not exist in fix_inflow.");
      rdata[l++] = static_cast<double> (group->bitmask[igroup]);
    }   

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %lf %lf %d",&r_cut_n,&r_cut_t,&kbt,&binsize,&mirror);
    rdata[l++] = r_cut_n;
    rdata[l++] = r_cut_t;
    rdata[l++] = kbt;
    rdata[l++] = binsize;
    rdata[l++] = static_cast<double> (mirror);

    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%d",&ind_press);
    rdata[l++] = static_cast<double> (ind_press);
    if (ind_press){
      sscanf(buf,"%d %d %lf %s %s",&ind_press,&n_press,&r_press,&fname1[0],&grp[0]);
      rdata[l++] = static_cast<double> (n_press);
      rdata[l++] = r_press;
      igroup = group->find(grp);
      if (igroup == -1) error->one(FLERR,"Group ID for group_p does not exist in fix_inflow.");
      groupbit_p = group->bitmask[igroup];
      rdata[l++] = static_cast<double> (groupbit_p);
      f_read1 = fopen(fname1,"r");
      if (f_read1 == (FILE*) NULL)
        error->one(FLERR,"Could not open input pressure file in fix_inflow.");
      for (j=0; j<n_press; j++){
        fgets(buf,BUFSIZ,f_read1);
        sscanf(buf,"%lf %lf",&dummy,&rdata[l]);
        l++;
      }
      fclose(f_read1);

      if (r_cut_n < r_press)
        error->warning(FLERR,"The parameter r_cut_n is too small in fix_inflow!");       
    }

    for (i=0; i<num_shapes_tot; i++){
      fgets(buf,BUFSIZ,f_read);
      sscanf(buf,"%d", &ttyp);
      if (l + 12 > nms){
        nms += NUM;
        memory->grow(rdata,nms,"fix_inflow:rdata");
      }
      switch (ttyp){
        case 1 :
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],
		&rdata[l+2],&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8],&rdata[l+9]);
          ndv1 = 1;
          ndv2 = 1;
          l += 10;  
  	  break;
        case 2 :
          sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&rdata[l],&rdata[l+1],&rdata[l+2],
		&rdata[l+3],&rdata[l+4],&rdata[l+5],&rdata[l+6],&rdata[l+7],&rdata[l+8],&rdata[l+9],&rdata[l+10],&rdata[l+11]);
          ndv1 = static_cast<int> (rdata[l+10]);
          ndv2 = static_cast<int> (rdata[l+11]);
          l += 12;
          break;
      }
      if (l + ndv1*ndv2*3 > nms){
        while (l + ndv1*ndv2*3 < nms)
          nms += NUM;
        memory->grow(rdata,nms,"fix_inflow:rdata");
      }
      for (j = 0; j < ndv1; ++j)
        for (k = 0; k < ndv2; ++k){
          fgets(buf,BUFSIZ,f_read);
          sscanf(buf, "%lf %lf %lf", &rdata[l], &rdata[l+1], &rdata[l+2]);
          l += 3;
      }
    }
    fclose(f_read);
  }

  MPI_Bcast(&l,1,MPI_INT,0,world);
  if (me)
    memory->create(rdata,l,"fix_inflow:rdata");
  MPI_Bcast(rdata,l,MPI_DOUBLE,0,world);

  l = 0;
  num_shapes_tot = static_cast<int> (rdata[l++]);
  num_at_types = static_cast<int> (rdata[l++]);

  memory->create(at_type,num_at_types,"fix_inflow:at_type");
  memory->create(at_groupbit,num_at_types,"fix_inflow:at_groupbit");
  memory->create(at_dens,num_at_types,"fix_inflow:at_dens");
  memory->create(vls,num_at_types,"fix_inflow:vls");
  for (i=0; i<num_at_types; i++){
    at_type[i] = static_cast<int> (rdata[l++]);
    at_dens[i] = rdata[l++];
    at_groupbit[i] = static_cast<int> (rdata[l++]);
  }
  
  r_cut_n = rdata[l++];
  r_cut_t = rdata[l++];
  kbt = rdata[l++];
  binsize = rdata[l++];
  mirror = static_cast<int> (rdata[l++]); 
  if (binsize <= 0.0)
    binsize = 0.5*r_cut_n;
  binsizeinv = 1.0/binsize;

  for (i=0; i<num_at_types; i++) 
    vls[i] = sqrt(kbt/atom->mass[at_type[i]]);

  ind_press = static_cast<int> (rdata[l++]);
  if (ind_press){
    n_press = static_cast<int> (rdata[l++]);
    r_press = rdata[l++];
    groupbit_p = static_cast<int> (rdata[l++]);
    memory->create(f_press,n_press+1,"fix_inflow:f_press");
    for (j=0; j<n_press; j++)
      f_press[j] = rdata[l++];
    f_press[n_press] = 0.0;
  }

  num_shapes = 0;
  max_shapes = 0;
  num_flow_tot = 0;
  nnmax = atom->nlocal;
  if (nnmax == 0) nnmax = 1;
  bin_max_shape = NUM1; 
  memory->create(x0,num_shapes_tot,3,"fix_inflow:x0");
  memory->create(aa,num_shapes_tot,8,"fix_inflow:aa");
  memory->create(rot,num_shapes_tot,3,3,"fix_inflow:rot");
  memory->create(ndiv,num_shapes_tot,2,"fix_inflow:ndiv");
  memory->create(ptype,num_shapes_tot,"fix_inflow:ptype");
  vel = new double***[num_shapes_tot];
  area = new double**[num_shapes_tot];
  memory->create(shapes_local,num_shapes_tot,"fix_inflow:shapes_local");
  memory->create(tot_shapes,nnmax,"fix_inflow:tot_shapes");

  for (i=0; i<num_shapes_tot; i++){
    ptype[i] = static_cast<int> (rdata[l]);
    switch (ptype[i]){
      case 1 :
        for (j=0; j<3; j++){
          x0[i][j] = rdata[l+j+1];
          a1[j] = rdata[l+j+4] - rdata[l+j+1];
          a2[j] = rdata[l+j+7] - rdata[l+j+1];
        }
        ndiv[i][0] = 1;
        ndiv[i][1] = 1;
        memory->create(area[i],ndiv[i][0], ndiv[i][1], "fix_inflow:area[i]");
        setup_rot(i,a1,a2);
	l += 10;
        break;
      case 2 :
        for (j=0; j<3; j++){
          x0[i][j] = rdata[l+j+1];
          a1[j] = rdata[l+j+4] - rdata[l+j+1];
          a2[j] = rdata[l+j+7] - rdata[l+j+1];
        }
        ndiv[i][0] = static_cast<int> (rdata[l+10]);
        ndiv[i][1] = static_cast<int> (rdata[l+11]);
        memory->create(area[i],ndiv[i][0], ndiv[i][1], "fix_inflow:area[i]");
        setup_rot(i,a1,a2);
	l += 12;
        break;
    }
    num_flow_tot += ndiv[i][0]*ndiv[i][1];

    if (check_shapes(i)){
      shapes_local[num_shapes] = i;
      num_shapes++;
    }

    memory->create(vel[i],ndiv[i][0],ndiv[i][1],3,"fix_inflow:vel[i]");
    for (j = 0; j < ndiv[i][0]; ++j)
      for (k = 0; k < ndiv[i][1]; ++k){
        rot_forward(rdata[l],rdata[l+1],rdata[l+2],i);
        vel[i][j][k][0] = rdata[l];
        vel[i][j][k][1] = rdata[l+1];
        vel[i][j][k][2] = rdata[l+2];
	l += 3;
	if (vel[i][j][k][2] <= 0.0) error->one(FLERR," A shape has a non-positive normal velocity!"); 
      }
  }

  memory->destroy(rdata);
}

/* ---------------------------------------------------------------------- */

FixInflow::~FixInflow()
{
  int i;

  for (i = 0; i < num_shapes_tot; ++i){
    memory->destroy(vel[i]);
    memory->destroy(area[i]);
  }
  delete[] vel;
  delete[] area;

  if (num_flow){
    memory->destroy(ncount);
    memory->destroy(info_flow);
  }
  memory->destroy(ptype); 
  memory->destroy(ndiv);
  memory->destroy(rot);
  memory->destroy(x0);
  memory->destroy(aa); 
  memory->destroy(shapes_local);
  memory->destroy(at_type);
  memory->destroy(at_groupbit);
  memory->destroy(at_dens);
  memory->destroy(vls); 

  if (ind_press) memory->destroy(f_press);

  memory->destroy(bin_shapes);
  memory->destroy(num_bin_shapes);
  memory->destroy(tot_shapes);
  if (max_shapes) memory->destroy(ind_shapes);

  delete ranmars0;
}

/* ---------------------------------------------------------------------- */

int FixInflow::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= PRE_EXCHANGE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */


void FixInflow::setup(int vflag)
{
  int i,j,k,l,nn[3],nmt,ind,fl_tmp[num_flow_tot][3];
  double coord[3];
  double subhi[3], sublo[3]; 

  dtv = update->dt;

  if (neighbor->every > 1 || neighbor->delay > 1) 
    error->all(FLERR,"Reneighboring should be done every time step when using fix_inflow!");

  for (i = 0; i < 3; ++i){
    subhi[i] = domain->subhi[i];
    sublo[i] = domain->sublo[i]; 
  }

  num_flow = 0;
  for (i = 0; i < num_shapes_tot; ++i)
    for (j = 0; j < ndiv[i][0]; ++j)
      for (k = 0; k < ndiv[i][1]; ++k){    
        if (ptype[i] == 1){
          coord[0] = (aa[i][0] + aa[i][1]) / 3.0;
          coord[1] = aa[i][2] / 3.0;
	} else{
          coord[0] = aa[i][0]*(j+0.5)/ndiv[i][0] + aa[i][1]*(k+0.5)/ndiv[i][1];
          coord[1] = aa[i][2]*(k+0.5)/ndiv[i][1];
	}
        coord[2] = 0.0;
        rot_back(coord[0],coord[1],coord[2],i);
        for (l = 0; l < 3; ++l)
          coord[l] += x0[i][l];

        if (comm->user_part){
          ind = comm->coords_to_bin(coord,nn,comm->nbp_loc,comm->box_min);
          if (ind)
            if (comm->m_comm[nn[0]][nn[1]][nn[2]] & comm->local_bit){  
      	      fl_tmp[num_flow][0] = i;
              fl_tmp[num_flow][1] = j;
              fl_tmp[num_flow][2] = k;
              num_flow++;
	    }
        } else {
          if (coord[0] >= sublo[0] && coord[0] < subhi[0] && coord[1] >= sublo[1]
              && coord[1] < subhi[1] && coord[2] >= sublo[2] && coord[2] < subhi[2]) {
            fl_tmp[num_flow][0] = i;
            fl_tmp[num_flow][1] = j;
            fl_tmp[num_flow][2] = k;
            num_flow++;
	  }
        }
      }

  nmt = 0; 
  MPI_Reduce(&num_flow, &nmt, 1, MPI_INT, MPI_SUM, 0, world);
  if (me == 0 && nmt != num_flow_tot)
    error->one(FLERR,"Not all injection faces were properly assigned!");  

  if (num_flow > 0){
    memory->create(ncount,num_flow,num_at_types,"fix_inflow:ncount");
    memory->create(info_flow,num_flow,3,"fix_inflow:info_flow");
  }
  for (i = 0; i < num_flow; ++i){ 
    for (k = 0; k < num_at_types; ++k)    
      ncount[i][k] = 0.0;
    for (j = 0; j < 3; ++j)
      info_flow[i][j] = fl_tmp[i][j];
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
           memory->grow(bin_shapes,bin_max_shape,nbin,"fix_inflow:bin_shapes");
         }
       }
    }

  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixInflow::post_integrate()
{
  int i,j,k,m,kk,ind,i_x[NUM],cc,ix,iy;
  double dl[3],t_x[NUM],dd[3],xp[3],xh[3],vh[3];
  double tt,dtt,u0,u1;
  int cplane,cond,bounce,ccount;
  double **x = atom->x;
  double **v = atom->v;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      cond = 1;
      ccount = 0;
      dtt = dtv;
      cplane = -1;
      while (cond && ccount < max_count){
        ind = 0;
        for (m = 0; m < tot_shapes[i]; ++m){
          kk = ind_shapes[i][m];
          if (kk != cplane){
            for (j = 0; j < 3; j++){
              xh[j] = x[i][j] - x0[kk][j];
              vh[j] = v[i][j];
            }
            rot_forward(xh[0],xh[1],xh[2],kk);
            rot_forward(vh[0],vh[1],vh[2],kk);
            for (j = 0; j < 3; j++){
              dl[j] = vh[j]*dtt;
              xp[j] = xh[j] - dl[j];
            }

            tt = -1.0;
            u1 = xp[1]/aa[kk][2];
            u0 = (xp[0]-u1*aa[kk][1])/aa[kk][0];
            ix = static_cast<int> (u0*ndiv[kk][0]);
            iy = static_cast<int> (u1*ndiv[kk][1]);
            if (ix < 0) ix = 0;
            if (ix > (ndiv[kk][0] - 1)) ix = ndiv[kk][0] - 1;
            if (iy < 0) iy = 0;
            if (iy > (ndiv[kk][1] - 1)) iy = ndiv[kk][1] - 1;
            for (j = 0; j < 3; ++j)
              xh[j] -= vel[kk][ix][iy][j] * dtt;

            if (xh[2]*xp[2] < 0.0)
              if (xp[2]>0.0) //refl always = 1
                tt = xp[2]/(xp[2]-xh[2]);

            if (tt>0.0 && tt<=1.0){
                t_x[ind] = tt;
                i_x[ind] = kk;
                ind++;
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
            t_x[kk] = t_x[ind-j];
            i_x[kk] = i_x[ind-j];
            t_x[ind-j] = tt;
            i_x[ind-j] = cc;
          }
          while (ind){
            bounce = 1;
            tt = t_x[ind - 1];
            kk = i_x[ind - 1];
            for (k = 0; k < 3; k++){
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
                ix = 0;
                iy = 0;
                if (u0 < 0.0 || u1 < 0.0 || u0+u1 > 1.0)
                  bounce = 0;
                break;
              case 2 :
                u1 = dd[1]/aa[kk][2];
                u0 = (dd[0]-u1*aa[kk][1])/aa[kk][0];
                ix = static_cast<int> (u0*ndiv[kk][0]);
                iy = static_cast<int> (u1*ndiv[kk][1]);
                if (ix < 0) ix = 0;
                if (ix > (ndiv[kk][0] - 1)) ix = ndiv[kk][0] - 1;
                if (iy < 0) iy = 0;
                if (iy > (ndiv[kk][1] - 1)) iy = ndiv[kk][1] - 1;
                if (u0 < 0.0 || u1 < 0.0 || u0 > 1.0 || u1 > 1.0)
                  bounce = 0;
                break;
	    }

            if (bounce){
              ccount++;
              cplane = kk;
              if (mirror){
                vh[2] = 2.0 * vel[kk][ix][iy][2]-vh[2];
              }
              else {
                for (k = 0; k < 3; k++){
                  vh[k] = -vh[k] + 2.0*vel[kk][ix][iy][k];
                }
              }

              for (k = 0; k < 3; k++){
                xp[k] = dd[k];
                xh[k] = xp[k] + (1.0-tt)*dtt*vh[k];      // verlet
              }
              rot_back(xh[0],xh[1],xh[2],kk);
              rot_back(vh[0],vh[1],vh[2],kk);
              for (k = 0; k < 3; k++){
                x[i][k] = xh[k] + x0[kk][k];
                v[i][k] = vh[k];
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
}

/* ---------------------------------------------------------------------- */

void FixInflow::pre_exchange()
{
  int i,j,k,l,m,kk;
  double dl[3],t_x[NUM],dd[3],xp[3],xh[3],vh[3];
  double randnum1,randnum2,randnum3,xtemp,ytemp,ztemp,vxtemp,vytemp,vztemp,xx[3];
  bigint ntemp;

// Insert loop
  for (l = 0; l < num_flow; ++l){
    i = info_flow[l][0];
    j = info_flow[l][1];
    k = info_flow[l][2];
    //insert particles
    for (m = 0; m < num_at_types; ++m){
      ncount[l][m] += at_dens[m]*dtv*area[i][j][k]*vel[i][j][k][2];
      while (ncount[l][m] > 0.99999999){
        ncount[l][m] -= 1.0;
        randnum1 = ranmars0->uniform();
        randnum2 = ranmars0->uniform();
        randnum3 = ranmars0->uniform();
        if (ptype[i] == 1){
          while (randnum1 + randnum2 > 1.0){
            randnum1 = ranmars0->uniform();
            randnum2 = ranmars0->uniform();
          }
          xtemp = aa[i][0]*randnum1 + aa[i][1]*randnum2;
          ytemp = aa[i][2]*randnum2;
          ztemp = vel[i][j][k][2]*dtv*randnum3;
        } else {
          xtemp = aa[i][0]*(j + randnum1) / static_cast<double>(ndiv[i][0]) + aa[i][1]*(k + randnum2) / static_cast<double>(ndiv[i][1]);
          ytemp = aa[i][2]*(k + randnum2) / static_cast<double>(ndiv[i][1]);
          ztemp = vel[i][j][k][2]*dtv*randnum3;
	}
        rot_back(xtemp,ytemp,ztemp,i);
        xx[0] = xtemp + x0[i][0];
        xx[1] = ytemp + x0[i][1];
        xx[2] = ztemp + x0[i][2];
        vxtemp = vls[m]*ranmars0->gaussian() + vel[i][j][k][0];
        vytemp = vls[m]*ranmars0->gaussian() + vel[i][j][k][1];
        vztemp = vls[m]*ranmars0->gaussian() + vel[i][j][k][2];
        rot_back(vxtemp,vytemp,vztemp,i);


        atom->avec->create_atom(at_type[m],xx);
        kk = atom->nlocal - 1;
        atom->mask[kk] = 1 | groupbit;
        if (groupbit != at_groupbit[m])
          atom->mask[kk] |= at_groupbit[m];
        atom->v[kk][0] = vxtemp;
        atom->v[kk][1] = vytemp;
        atom->v[kk][2] = vztemp;
      }
    }
  }//inject loop

  ntemp = atom->nlocal;
  MPI_Allreduce(&ntemp,&atom->natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (atom->natoms < 0 || atom->natoms >= MAXBIGINT)
    error->all(FLERR,"Too many total atoms");

  if (atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }
}

/*------------------------------------------------------------------------*/

void FixInflow::post_force(int vflag)
{
  int i,m,kk,j,ix,iy,iz;
  double xh[3],ff[3],u0,u1;

  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  if (nlocal > nnmax){
    while (nnmax < nlocal)               
      nnmax += BUFMIN;
    if (max_shapes)
      memory->grow(ind_shapes,nnmax,max_shapes,"fix_inflow:ind_shapes");
    memory->grow(tot_shapes,nnmax,"fix_inflow:tot_shapes");
  }  

  if (neighbor->ago == 0)
    shape_decide();

  if (ind_press)
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit_p)
        for (m = 0; m < tot_shapes[i]; m++){
          kk = ind_shapes[i][m];
          for (j = 0; j < 3; j++)
            xh[j] = x[i][j] - x0[kk][j];
          rot_forward(xh[0],xh[1],xh[2],kk);
          switch (ptype[kk]){
            case 1 :
              u1 = xh[1]/aa[kk][2];
              u0 = (xh[0]-u1*aa[kk][1])/aa[kk][0];
              if (u0 >= 0.0 && u1 >= 0.0 && u0+u1 <= 1.0){
                if (xh[2]>0.0 && xh[2]<r_press){ 
                  iz = static_cast<int> (xh[2]*n_press/r_press);
                  ff[0] = 0.0;
                  ff[1] = 0.0;
                  ff[2] = f_press[iz];
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for (j = 0; j < 3; j++)
                    f[i][j] += ff[j];
		}
	      }
	      break;
            case 2 :
              u1 = xh[1]/aa[kk][2];
              u0 = (xh[0]-u1*aa[kk][1])/aa[kk][0];
              if (u0 >= 0.0 && u1 >= 0.0 && u0 < 1.0 && u1 < 1.0){
                ix = static_cast<int> (u0*ndiv[kk][0]);
                iy = static_cast<int> (u1*ndiv[kk][1]);
                if (xh[2]>0.0 && xh[2]<r_press){
                  iz = static_cast<int> (xh[2]*n_press/r_press);
                  ff[0] = 0.0;
                  ff[1] = 0.0;
                  ff[2] = f_press[iz];
                  rot_back(ff[0],ff[1],ff[2],kk);
                  for (j = 0; j < 3; j++)
                    f[i][j] += ff[j];
                }
	      }
              break;
	  }
	}
}

/*------------------------------------------------------------------------*/

void FixInflow::reset_target(double rho_new)
{
  int i,j,m,sh,ind;
  double dd[3],u0,u1,u2,zz,dl;

  double **x = atom->x;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double *rho = atom->rho;
  int nall = nlocal + atom->nghost;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      for (m = 0; m < tot_shapes[i]; m++){
        sh = ind_shapes[i][m];
        for (j = 0; j < 3; j++)
          dd[j] = x[i][j] - x0[sh][j];
        rot_forward(dd[0],dd[1],dd[2],sh);
        ind = 0;
        if (dd[0] > aa[sh][4]-r_cut_t && dd[0] < aa[sh][5]+r_cut_t && dd[1] > -r_cut_t && dd[1] < aa[sh][2]+r_cut_t && dd[2] >= 0.0 && dd[2] < r_cut_n){
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

  for (i = nlocal; i < nall; i++)
    if (mask[i] & groupbit)
      for (sh = 0; sh < num_shapes; sh++){
        for (j = 0; j < 3; j++)
          dd[j] = x[i][j] - x0[sh][j];
        rot_forward(dd[0],dd[1],dd[2],sh);
        ind = 0;
        if (dd[0] > aa[sh][4]-r_cut_t && dd[0] < aa[sh][5]+r_cut_t && dd[1] > -r_cut_t && dd[1] < aa[sh][2]+r_cut_t && dd[2] >= 0.0 && dd[2] < r_cut_n){
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

/* ---------------------------------------------------------------------- */

int FixInflow::check_shapes(int sh)
{
  int i,ch_ind,nn[3],jj[3];
  double rad, rad_n, rad_t, u0, u1, u2, zz, dl;
  double dd[3],ddx[3];

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

void FixInflow::setup_rot(int id, double a1[], double a2[])
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
    area[id][0][0] = 0.5*nr;
  } else{
    aa[id][5] = MAX(rx,rx + aa[id][1]); 
    for (i = 0; i < ndiv[id][0]; ++i)
      for (j = 0; j < ndiv[id][1]; ++j)
        area[id][i][j] = nr / ndiv[id][0] / ndiv[id][1];
  }
}

/* ---------------------------------------------------------------------- */

void FixInflow::rot_forward(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][0][1]*y1 + rot[id][0][2]*z1;
  y = rot[id][1][0]*x1 + rot[id][1][1]*y1 + rot[id][1][2]*z1;
  z = rot[id][2][0]*x1 + rot[id][2][1]*y1 + rot[id][2][2]*z1;
}

/* ---------------------------------------------------------------------- */

void FixInflow::rot_back(double &x, double &y, double &z, int id)
{
  double x1,y1,z1;

  x1 = x; y1 = y; z1 = z;
  x = rot[id][0][0]*x1 + rot[id][1][0]*y1 + rot[id][2][0]*z1;
  y = rot[id][0][1]*x1 + rot[id][1][1]*y1 + rot[id][2][1]*z1;
  z = rot[id][0][2]*x1 + rot[id][1][2]*y1 + rot[id][2][2]*z1;
}

/*------------------------------------------------------------------------*/

void FixInflow::shape_decide()
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

void FixInflow::grow_shape_arrays()
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

void FixInflow::setup_bins()
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

int FixInflow::check_bins(int bin, int sh)
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

int FixInflow::coord2bin(double *xx)
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
