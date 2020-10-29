/* ----------------------------------------------------------------------
  Dmitry Fedosov - 08/12/05  accumulation of statistics
 
  LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

------------------------------------------------------------------------- */

#include <cmath>
#include "statistic_com.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticCOM::StatisticCOM(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  int i, j, k;

  init_on = 0;
  memory->create(gx,nx,ny,nz,"statistic_com:gx");
  memory->create(gy,nx,ny,nz,"statistic_com:gy");
  memory->create(gz,nx,ny,nz,"statistic_com:gz");  
  memory->create(vcx,nx,ny,nz,"statistic_com:vcx");
  memory->create(vcy,nx,ny,nz,"statistic_com:vcy");
  memory->create(vcz,nx,ny,nz,"statistic_com:vcz");
  memory->create(fx,nx,ny,nz,"statistic_com:fx");
  memory->create(fy,nx,ny,nz,"statistic_com:fy");
  memory->create(fz,nx,ny,nz,"statistic_com:fz"); 

  for (i=0;i<nx;i++)
    for (j=0;j<ny;j++)
      for (k=0;k<nz;k++){
        gx[i][j][k] = 0.0;
        gy[i][j][k] = 0.0;
        gz[i][j][k] = 0.0;
        vcx[i][j][k] = 0.0;
        vcy[i][j][k] = 0.0;
        vcz[i][j][k] = 0.0;
        fx[i][j][k] = 0.0;
        fy[i][j][k] = 0.0;
        fz[i][j][k] = 0.0;
      }

  if (!comm->me){
    memory->create(c_dist,nx,ny,nz,"statistic_com:c_dist");

    for (i=0;i<nx;i++)
      for (j=0;j<ny;j++)
        for (k=0;k<nz;k++)
          c_dist[i][j][k] = 0.0;
  }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticCOM::~StatisticCOM()
{
  memory->destroy(gx);
  memory->destroy(gy);
  memory->destroy(gz);
  memory->destroy(vcx);
  memory->destroy(vcy);
  memory->destroy(vcz);
  memory->destroy(fx);
  memory->destroy(fy);
  memory->destroy(fz);   
  if (init_on){
    memory->sfree(c_m);
    memory->sfree(c_mt);
    memory->sfree(mol_list);
    if (cyl_ind) memory->destroy(rdh);
  }
  if (!comm->me)
    memory->destroy(c_dist);  
}


/* ---------------------------------------------------------------------- */

void StatisticCOM::calc_stat()
{
  tagint i,j;
  int k;
  double xh[3], gyy, gzz;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  tagint ind_h[atom->n_mol_max];
  tagint mol;
  tagint *n_atoms = atom->mol_size;
  
  if (init_on == 0){
    nm = atom->n_mol_max;
    c_m = (double *) memory->smalloc(3*nm*sizeof(double),"statistic_com:c_m");
    c_mt = (double *) memory->smalloc(3*nm*sizeof(double),"statistic_com:c_mt");   
    mol_list = (tagint *) memory->smalloc(nm*sizeof(tagint),"statistic_com:mol_list");
    if (cyl_ind) memory->create(rdh,nm,3,"statistic_com:rdh");
    for (i=0; i<nm; i++){
      ind_h[i] = 0;  
      mol_list[i] = 0; 
    }
    for (i=0; i<nlocal; i++){
      mol = molecule[i]; 
      if (mol && mask[i] & groupbit)
        ind_h[mol-1] = 1;
    }
    MPI_Allreduce(&ind_h,mol_list,nm,MPI_LMP_TAGINT,MPI_SUM,world);
    nm_h = 0;
    for (i=0; i<nm; i++)
      if (mol_list[i]){
        mol_list[nm_h] = i;
        nm_h++;
      }
    init_on = 1;  
    nm_hpm1 = 1.0/nm_h;
  }

  for(i=0; i<3*nm; i++) {   
    c_m[i] = 0.0;
    c_mt[i] = 0.0;
  }

  for (i=0; i<nlocal; i++){
    mol = molecule[i];     
    if (mol && mask[i] & groupbit){
      mol--; 
      domain->unmap(x[i],atom->image[i],xh); 
      for (j=0; j<3; j++) c_m[3*mol+j] += xh[j];
    }
  }  

  for (i=0; i<nm; i++){
    k = atom->mol_type[i];
    if (k > -1 && k < atom->n_mol_types)
      for (j=0; j<3; j++) 
        c_m[3*i+j] /= n_atoms[k];
  }   

  MPI_Allreduce(c_m,c_mt,3*nm,MPI_DOUBLE,MPI_SUM,world);

  if (cyl_ind && zhi > 0.5)
    for (i=0; i<nm; i++){
      rdh[i][0] = sqrt((c_mt[3*i+1]-ylo)*(c_mt[3*i+1]-ylo) + (c_mt[3*i+2]-zlo)*(c_mt[3*i+2]-zlo));
      if (rdh[i][0] > 0.0){
        rdh[i][1] = (c_mt[3*i+1]-ylo)/rdh[i][0];
        rdh[i][2] = (c_mt[3*i+2]-zlo)/rdh[i][0];
      }
    }
  
  for (i=0; i<nlocal; i++){
    mol = molecule[i];   
    if (mol && mask[i] & groupbit){
      mol--;
      domain->unmap(x[i],atom->image[i],xh);
      k = atom->mol_type[mol];
      if (k > -1 && k < atom->n_mol_types){
        if (map_index(c_mt[3*mol],c_mt[3*mol+1],c_mt[3*mol+2])){
          gx[is][js][ks] += (xh[0]-c_mt[3*mol])*(xh[0]-c_mt[3*mol])/n_atoms[k];
          vcx[is][js][ks] += v[i][0]/n_atoms[k];
          fx[is][js][ks] += f[i][0]/n_atoms[k];
          if (cyl_ind && zhi > 0.5){
            if (rdh[mol][0] > 0.0){
              gyy = (xh[1]-c_mt[3*mol+1])*rdh[mol][1] + (xh[2]-c_mt[3*mol+2])*rdh[mol][2];
              gzz = (xh[2]-c_mt[3*mol+2])*rdh[mol][1] - (xh[1]-c_mt[3*mol+1])*rdh[mol][2];
              gy[is][js][ks] += gyy*gyy/n_atoms[k];
              gz[is][js][ks] += gzz*gzz/n_atoms[k];
              vcy[is][js][ks] += (v[i][1]*rdh[mol][1] + v[i][2]*rdh[mol][2])/n_atoms[k];
              vcz[is][js][ks] += (v[i][2]*rdh[mol][1] - v[i][1]*rdh[mol][2])/n_atoms[k];
              fy[is][js][ks] += (f[i][1]*rdh[mol][1] + f[i][2]*rdh[mol][2])/n_atoms[k];
              fz[is][js][ks] += (f[i][2]*rdh[mol][1] - f[i][1]*rdh[mol][2])/n_atoms[k];
            }
          } else{
            gy[is][js][ks] += (xh[1]-c_mt[3*mol+1])*(xh[1]-c_mt[3*mol+1])/n_atoms[k];
            gz[is][js][ks] += (xh[2]-c_mt[3*mol+2])*(xh[2]-c_mt[3*mol+2])/n_atoms[k];
            vcy[is][js][ks] += v[i][1]/n_atoms[k];
            vcz[is][js][ks] += v[i][2]/n_atoms[k];
            fy[is][js][ks] += f[i][1]/n_atoms[k];
            fz[is][js][ks] += f[i][2]/n_atoms[k];
          }
        }
      }
    }
  }
  
  if (!comm->me)  
    for(i=0; i<nm_h; i++){
      j = mol_list[i]; 
      if (map_index(c_mt[3*j],c_mt[3*j+1],c_mt[3*j+2]))
        c_dist[is][js][ks] += nm_hpm1;
    }

  num_step++;
}

/* ---------------------------------------------------------------------- */

void StatisticCOM:: write_stat(bigint step)
{
  int i, j, k, l, m;
  double x, y, z, rr, theta, c_norm;
  int total = nx*ny*nz;
  char f_name[FILENAME_MAX];
  double rad[10];
  double *gxtmp, *gytmp, *gztmp, *tmp, *fxtmp, *fytmp, *fztmp;
  double *vcxtmp, *vcytmp, *vcztmp;

  gxtmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:gxtmp");
  gytmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:gytmp");
  gztmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:gztmp");
  vcxtmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:vcxtmp");
  vcytmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:vcytmp");
  vcztmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:vcztmp");
  fxtmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:fxtmp");
  fytmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:fytmp");
  fztmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:fztmp");
  tmp = (double *) memory->smalloc(total*sizeof(double),"statistic_com:tmp");

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=gx[i][j][k];
        gx[i][j][k] = 0.0;
        gxtmp[l] = 0.0;
        gytmp[l] = 0.0;
        gztmp[l] = 0.0;
        vcxtmp[l] = 0.0;
        vcytmp[l] = 0.0;
        vcztmp[l] = 0.0; 
        fxtmp[l] = 0.0;
        fytmp[l] = 0.0;
        fztmp[l] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,gxtmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=gy[i][j][k];
        gy[i][j][k] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,gytmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=gz[i][j][k];
        gz[i][j][k] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,gztmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=vcx[i][j][k];
        vcx[i][j][k] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,vcxtmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=vcy[i][j][k];
        vcy[i][j][k] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,vcytmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=vcz[i][j][k];
        vcz[i][j][k] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,vcztmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (i=0; i<nx; i++)
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++){
        tmp[l]=fx[i][j][k];
        fx[i][j][k] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,fxtmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (i=0; i<nx; i++)
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++){
        tmp[l]=fy[i][j][k];
        fy[i][j][k] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,fytmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (i=0; i<nx; i++)
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++){
        tmp[l]=fz[i][j][k];
        fz[i][j][k] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,fztmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  
  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step);
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"center_mass\",\"Rgx2\",\"Rgy2\",\"Rgz2\",\"Rg2\",\"Vcx\",\"Vcy\",\"Vcz\",\"Fx\",\"Fy\",\"Fz\" \n");
    if (cyl_ind){ 
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", ny, nz, nx);
      for (i=0; i<nx; i++)
        for (k=0; k<nz; k++)
          for (j=0; j<ny; j++){
            l = k*nx*ny + j*nx + i;
            x = xlo + (i+0.5)*dx;
            rr = (j+0.5)*dy;
            c_norm = M_PI*dy*dy*(2.0*j+1.0);
            theta = k*dz;
            y = ylo + rr*cos(theta);
            z = zlo + rr*sin(theta);
            if (c_dist[i][j][k]>0.0){
              rad[0] = gxtmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[1] = gytmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[2] = gztmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[3] = vcxtmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[4] = vcytmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[5] = vcztmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[6] = fxtmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[7] = fytmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[8] = fztmp[l]/c_dist[i][j][k]*nm_hpm1;
            } else {
              for (m=0; m<9; m++)
                rad[m] = 0.0;
	    } 
            fprintf(out_stat,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n",x, y, z, c_dist[i][j][k]/num_step/c_norm, rad[0], rad[1], rad[2], rad[0]+rad[1]+rad[2], rad[3], rad[4], rad[5], rad[6], rad[7], rad[8]);
            c_dist[i][j][k] = 0.0;
	  }
    } else{ 
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", nx, ny, nz);
      l = 0;
      for (k=0; k<nz; k++)
        for (j=0; j<ny; j++)
          for (i=0; i<nx; i++){
            x = xlo + (i+0.5)*dx;
            y = ylo + (j+0.5)*dy;
            z = zlo + (k+0.5)*dz;
            if (c_dist[i][j][k]>0.0){
              rad[0] = gxtmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[1] = gytmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[2] = gztmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[3] = vcxtmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[4] = vcytmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[5] = vcztmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[6] = fxtmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[7] = fytmp[l]/c_dist[i][j][k]*nm_hpm1;
              rad[8] = fztmp[l]/c_dist[i][j][k]*nm_hpm1;
	    } else {
              for (m=0; m<9; m++)
                rad[m] = 0.0;
	    }
            fprintf(out_stat,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n",x, y, z, c_dist[i][j][k]/num_step, rad[0], rad[1], rad[2], rad[0]+rad[1]+rad[2], rad[3], rad[4], rad[5], rad[6], rad[7], rad[8]);
            c_dist[i][j][k] = 0.0;
            l++;
	  }
    }
    fclose(out_stat);
  }
  num_step = 0;
  memory->sfree(gxtmp);
  memory->sfree(gytmp);
  memory->sfree(gztmp);
  memory->sfree(vcxtmp);
  memory->sfree(vcytmp);
  memory->sfree(vcztmp);
  memory->sfree(fxtmp);
  memory->sfree(fytmp);
  memory->sfree(fztmp);
  memory->sfree(tmp);
}

/* ---------------------------------------------------------------------- */

