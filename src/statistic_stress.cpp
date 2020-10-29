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
#include "statistic_stress.h"
#include "atom.h"
#include "force.h"
#include "memory.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticStress::StatisticStress(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  int i, j, k, l;

  cyl_ind = 0; 
  poly_ind = 0;
  if (force->bond || force->angle || force->dihedral) poly_ind = 1;

  ss = memory->create(ss,nx,ny,nz,6,"statistic_stress:ss");
  vv = memory->create(vv,nx,ny,nz,6,"statistic_stress:vv");
  ss1 = memory->create(ss1,nx,ny,nz,6,"statistic_stress:ss1");
  ss2 = memory->create(ss2,nx,ny,nz,6,"statistic_stress:ss2");
  if (poly_ind){
    ss_p = memory->create(ss_p,nx,ny,nz,6,"statistic_stress:ss_p");
    vv_p = memory->create(vv_p,nx,ny,nz,6,"statistic_stress:vv_p");
    ss_p1 = memory->create(ss_p1,nx,ny,nz,6,"statistic_stress:ss_p1");
    ss_p2 = memory->create(ss_p2,nx,ny,nz,6,"statistic_stress:ss_p2");
  }

  for (i=0; i<nx; i++)
    for (j=0; j<ny; j++)
      for (k=0; k<nz; k++)
        for (l=0; l<6; l++){
          ss[i][j][k][l] = 0.0;
          vv[i][j][k][l] = 0.0;
          ss1[i][j][k][l] = 0.0;
          ss2[i][j][k][l] = 0.0;
          if (poly_ind){
            ss_p[i][j][k][l] = 0.0;
            vv_p[i][j][k][l] = 0.0;
            ss_p1[i][j][k][l] = 0.0;
            ss_p2[i][j][k][l] = 0.0;
          }
        }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticStress::~StatisticStress()
{
  memory->destroy(ss);
  memory->destroy(vv);
  memory->destroy(ss1);
  memory->destroy(ss2);
  if (poly_ind){
    memory->destroy(ss_p);
    memory->destroy(vv_p);
    memory->destroy(ss_p1);
    memory->destroy(ss_p2);
  }
}

/* ---------------------------------------------------------------------- */

void StatisticStress::calc_stat()
{
  int i,j,k,l;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *mol = atom->molecule;
  double *mass = atom->mass;
  double **x = atom->x;
  double **v = atom->v;
  double h_mass;
  double v_avg[nx][ny][nz][4], v_avg_tmp[nx][ny][nz][4];
  int nlocal = atom->nlocal;

  for (i=0; i<nx; i++)
    for (j=0; j<ny; j++)
      for (k=0; k<nz; k++)
        for (l=0; l<4; l++){
          v_avg[i][j][k][l] = 0.0;
          v_avg_tmp[i][j][k][l] = 0.0;
        }

  for (l=0; l<nlocal; l++)
    if (mask[l] & groupbit)
      if (map_index(x[l][0],x[l][1],x[l][2])){
        v_avg_tmp[is][js][ks][0] += v[l][0];
        v_avg_tmp[is][js][ks][1] += v[l][1];
        v_avg_tmp[is][js][ks][2] += v[l][2];
        v_avg_tmp[is][js][ks][3] += 1.0;
      }

  MPI_Allreduce(&v_avg_tmp,&v_avg,nx*ny*nz*4,MPI_DOUBLE,MPI_SUM,world);

  for (i=0; i<nx; i++)
    for (j=0; j<ny; j++)
      for (k=0; k<nz; k++)
        for (l=0; l<3; l++)
          v_avg[i][j][k][l] /= v_avg[i][j][k][3];


  for (l=0; l<nlocal; l++)
    if (mask[l] & groupbit)
      if (map_index(x[l][0],x[l][1],x[l][2])){
        h_mass = mass[type[l]];
        vv[is][js][ks][0] += h_mass*(v[l][0]-v_avg[is][js][ks][0])*(v[l][0]-v_avg[is][js][ks][0]);
        vv[is][js][ks][1] += h_mass*(v[l][1]-v_avg[is][js][ks][1])*(v[l][1]-v_avg[is][js][ks][1]);
        vv[is][js][ks][2] += h_mass*(v[l][2]-v_avg[is][js][ks][2])*(v[l][2]-v_avg[is][js][ks][2]);
        vv[is][js][ks][3] += h_mass*(v[l][0]-v_avg[is][js][ks][0])*(v[l][1]-v_avg[is][js][ks][1]);
        vv[is][js][ks][4] += h_mass*(v[l][0]-v_avg[is][js][ks][0])*(v[l][2]-v_avg[is][js][ks][2]);
        vv[is][js][ks][5] += h_mass*(v[l][1]-v_avg[is][js][ks][1])*(v[l][2]-v_avg[is][js][ks][2]);
        if (poly_ind && mol[l]){
          vv_p[is][js][ks][0] += h_mass*(v[l][0]-v_avg[is][js][ks][0])*(v[l][0]-v_avg[is][js][ks][0]);
          vv_p[is][js][ks][1] += h_mass*(v[l][1]-v_avg[is][js][ks][1])*(v[l][1]-v_avg[is][js][ks][1]);
          vv_p[is][js][ks][2] += h_mass*(v[l][2]-v_avg[is][js][ks][2])*(v[l][2]-v_avg[is][js][ks][2]);
          vv_p[is][js][ks][3] += h_mass*(v[l][0]-v_avg[is][js][ks][0])*(v[l][1]-v_avg[is][js][ks][1]);
          vv_p[is][js][ks][4] += h_mass*(v[l][0]-v_avg[is][js][ks][0])*(v[l][2]-v_avg[is][js][ks][2]);
          vv_p[is][js][ks][5] += h_mass*(v[l][1]-v_avg[is][js][ks][1])*(v[l][2]-v_avg[is][js][ks][2]);
        }
      }
  num_step++;
}

/* ---------------------------------------------------------------------- */

void StatisticStress::virial1(int ii)
{
  map1 = map_index(atom->x[ii][0],atom->x[ii][1],atom->x[ii][2]);

  if (map1 && (atom->mask[ii] & groupbit)){
    is1 = is; js1 = js; ks1 = ks;
  }
  else
    map1 = 0;

}

/* ---------------------------------------------------------------------- */

void StatisticStress::virial2(int ii, double ff[6], int indd)
{
  int i;
  double ****ss_h = ss;

  if (indd == 1) ss_h = ss1;
  else if (indd == 2) ss_h = ss2;

  if (map1){
    for (i=0; i<6; i++)
      ss_h[is1][js1][ks1][i] += 0.5*ff[i];
  }

  if (map_index(atom->x[ii][0],atom->x[ii][1],atom->x[ii][2]) && (atom->mask[ii] & groupbit)){
    for (i=0; i<6; i++)
      ss_h[is][js][ks][i] += 0.5*ff[i];
  }
}

/* ---------------------------------------------------------------------- */

void StatisticStress::virial3(int ii, int jj, double ff[6])
{
  int i;

  if (map_index(atom->x[ii][0],atom->x[ii][1],atom->x[ii][2]) && (atom->mask[ii] & groupbit)){
    for (i=0; i<6; i++)
      ss_p[is][js][ks][i] += 0.5*ff[i];
  }

  if (map_index(atom->x[jj][0],atom->x[jj][1],atom->x[jj][2]) && (atom->mask[jj] & groupbit)){
    for (i=0; i<6; i++)
      ss_p[is][js][ks][i] += 0.5*ff[i];
  }
}

/* ---------------------------------------------------------------------- */

void StatisticStress::virial4(int ii, int jj, int kk, double ff[6])
{
  int i;

  if (map_index(atom->x[ii][0],atom->x[ii][1],atom->x[ii][2]) && (atom->mask[ii] & groupbit)){
    for (i=0; i<6; i++)
      ss_p1[is][js][ks][i] += ff[i]/3.0;
  }

  if (map_index(atom->x[jj][0],atom->x[jj][1],atom->x[jj][2]) && (atom->mask[jj] & groupbit)){
    for (i=0; i<6; i++)
      ss_p1[is][js][ks][i] += ff[i]/3.0;
  }

  if (map_index(atom->x[kk][0],atom->x[kk][1],atom->x[kk][2]) && (atom->mask[kk] & groupbit)){
    for (i=0; i<6; i++)
      ss_p1[is][js][ks][i] += ff[i]/3.0;
  }
}

/* ---------------------------------------------------------------------- */

void StatisticStress::virial5(int ii, int jj, int kk, int ll, double ff[6])
{
  int i;

  if (map_index(atom->x[ii][0],atom->x[ii][1],atom->x[ii][2]) && (atom->mask[ii] & groupbit)){
    for (i=0; i<6; i++)
      ss_p2[is][js][ks][i] += 0.25*ff[i];
  }

  if (map_index(atom->x[jj][0],atom->x[jj][1],atom->x[jj][2]) && (atom->mask[jj] & groupbit)){
    for (i=0; i<6; i++)
      ss_p2[is][js][ks][i] += 0.25*ff[i];
  }

  if (map_index(atom->x[kk][0],atom->x[kk][1],atom->x[kk][2]) && (atom->mask[kk] & groupbit)){
    for (i=0; i<6; i++)
      ss_p2[is][js][ks][i] += 0.25*ff[i];
  }

  if (map_index(atom->x[ll][0],atom->x[ll][1],atom->x[ll][2]) && (atom->mask[ll] & groupbit)){
    for (i=0; i<6; i++)
      ss_p2[is][js][ks][i] += 0.25*ff[i];
  }
}

/* ---------------------------------------------------------------------- */

void StatisticStress::virial6(int ii, double xx[3], double ff[6])
{
  int i;

  if (map_index(atom->x[ii][0],atom->x[ii][1],atom->x[ii][2]) && (atom->mask[ii] & groupbit)){
    for (i=0; i<6; i++)
      ss[is][js][ks][i] += 0.5*ff[i];
  }

  if (map_index(xx[0],xx[1],xx[2])){
    for (i=0; i<6; i++)
      ss[is][js][ks][i] += 0.5*ff[i];
  }
}

/* ---------------------------------------------------------------------- */

void StatisticStress:: write_stat(bigint step)
{
  int i,j,k,l;
  double stmp[nx][ny][nz][6], sptmp[nx][ny][nz][6], tmp[nx][ny][nz][6]; 
  double vtmp[nx][ny][nz][6], vptmp[nx][ny][nz][6], sp2tmp[nx][ny][nz][6];
  double s1tmp[nx][ny][nz][6], s2tmp[nx][ny][nz][6], sp1tmp[nx][ny][nz][6];
  double vol = xs*ys*zs/nx/ny/nz;
  double x,y,z;  
  char f_name[FILENAME_MAX];

  for (i=0; i<nx; i++)
    for (j=0; j<ny; j++)
      for (k=0; k<nz; k++)
        for (l=0; l<6; l++){
          tmp[i][j][k][l] = - vv[i][j][k][l]/num_step/vol;
          vv[i][j][k][l] = 0.0;
          vtmp[i][j][k][l] = 0.0;
	}
  MPI_Reduce(&tmp,&vtmp,nx*ny*nz*6,MPI_DOUBLE,MPI_SUM,0,world);

  for (i=0; i<nx; i++)
    for (j=0; j<ny; j++)
      for (k=0; k<nz; k++)
        for (l=0; l<6; l++){
          tmp[i][j][k][l] = - ss[i][j][k][l]/num_step/vol;
          ss[i][j][k][l] = 0.0;
          stmp[i][j][k][l] = 0.0;
	}
  MPI_Reduce(&tmp,&stmp,nx*ny*nz*6,MPI_DOUBLE,MPI_SUM,0,world);

  for (i=0; i<nx; i++)
    for (j=0; j<ny; j++)
      for (k=0; k<nz; k++)
        for (l=0; l<6; l++){
          tmp[i][j][k][l] = - ss1[i][j][k][l]/num_step/vol;
          ss1[i][j][k][l] = 0.0;
          s1tmp[i][j][k][l] = 0.0;
	}
  MPI_Reduce(&tmp,&s1tmp,nx*ny*nz*6,MPI_DOUBLE,MPI_SUM,0,world);

  for (i=0; i<nx; i++)
    for (j=0; j<ny; j++)
      for (k=0; k<nz; k++)
        for (l=0; l<6; l++){
          tmp[i][j][k][l] = - ss2[i][j][k][l]/num_step/vol;
          ss2[i][j][k][l] = 0.0;
          s2tmp[i][j][k][l] = 0.0;
	}
  MPI_Reduce(&tmp,&s2tmp,nx*ny*nz*6,MPI_DOUBLE,MPI_SUM,0,world);

  if (poly_ind){
    for (i=0; i<nx; i++)
      for (j=0; j<ny; j++)
        for (k=0; k<nz; k++)
          for (l=0; l<6; l++){
            tmp[i][j][k][l] = - vv_p[i][j][k][l]/num_step/vol;
            vv_p[i][j][k][l] = 0.0;
            vptmp[i][j][k][l] = 0.0;
	  }
    MPI_Reduce(&tmp,&vptmp,nx*ny*nz*6,MPI_DOUBLE,MPI_SUM,0,world);

    for (i=0; i<nx; i++)
      for (j=0; j<ny; j++)
        for (k=0; k<nz; k++)
          for (l=0; l<6; l++){
            tmp[i][j][k][l] = - ss_p[i][j][k][l]/num_step/vol;
            ss_p[i][j][k][l] = 0.0;
            sptmp[i][j][k][l] = 0.0;
	  }
    MPI_Reduce(&tmp,&sptmp,nx*ny*nz*6,MPI_DOUBLE,MPI_SUM,0,world);

    for (i=0; i<nx; i++)
      for (j=0; j<ny; j++)
        for (k=0; k<nz; k++)
          for (l=0; l<6; l++){
            tmp[i][j][k][l] = - ss_p1[i][j][k][l]/num_step/vol;
            ss_p1[i][j][k][l] = 0.0;
            sp1tmp[i][j][k][l] = 0.0;
	  }
    MPI_Reduce(&tmp,&sp1tmp,nx*ny*nz*6,MPI_DOUBLE,MPI_SUM,0,world);

    for (i=0; i<nx; i++)
      for (j=0; j<ny; j++)
        for (k=0; k<nz; k++)
          for (l=0; l<6; l++){
            tmp[i][j][k][l] = - ss_p2[i][j][k][l]/num_step/vol;
            ss_p2[i][j][k][l] = 0.0;
            sp2tmp[i][j][k][l] = 0.0;
	  }
    MPI_Reduce(&tmp,&sp2tmp,nx*ny*nz*6,MPI_DOUBLE,MPI_SUM,0,world);
  }      
  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"atom1_%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"Sxx_k\",\"Syy_k\",\"Szz_k\",\"Sxy_k\",\"Sxz_k\",\"Syz_k\",\"Sxx_v\",\"Syy_v\",\"Szz_v\",\"Sxy_v\",\"Sxz_v\",\"Syz_v\"  \n");
    fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", nx, ny, nz);

    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
        for (i=0; i<nx; i++){
          x = xlo + (i+0.5)*dx;
          y = ylo + (j+0.5)*dy;
          z = zlo + (k+0.5)*dz;
          fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf \n",x, y, z,vtmp[i][j][k][0],vtmp[i][j][k][1],vtmp[i][j][k][2],vtmp[i][j][k][3],vtmp[i][j][k][4],vtmp[i][j][k][5],stmp[i][j][k][0],stmp[i][j][k][1],stmp[i][j][k][2],stmp[i][j][k][3],stmp[i][j][k][4],stmp[i][j][k][5]);
	}
    fclose(out_stat);

    sprintf(f_name,"atom2_%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"Sxx_1\",\"Syy_1\",\"Szz_1\",\"Sxy_1\",\"Sxz_1\",\"Syz_1\",\"Sxx_2\",\"Syy_2\",\"Szz_2\",\"Sxy_2\",\"Sxz_2\",\"Syz_2\"  \n");
    fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", nx, ny, nz);

    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
        for (i=0; i<nx; i++){
          x = xlo + (i+0.5)*dx;
          y = ylo + (j+0.5)*dy;
          z = zlo + (k+0.5)*dz;
          fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf \n",x, y, z,s1tmp[i][j][k][0],s1tmp[i][j][k][1],s1tmp[i][j][k][2],s1tmp[i][j][k][3],s1tmp[i][j][k][4],s1tmp[i][j][k][5],s2tmp[i][j][k][0],s2tmp[i][j][k][1],s2tmp[i][j][k][2],s2tmp[i][j][k][3],s2tmp[i][j][k][4],s2tmp[i][j][k][5]);
	}
    fclose(out_stat);

    if (poly_ind){
      sprintf(f_name,"poly1_%s."BIGINT_FORMAT".plt",fname,step); 
      out_stat=fopen(f_name,"w");
      fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"Sxx_k\",\"Syy_k\",\"Szz_k\",\"Sxy_k\",\"Sxz_k\",\"Syz_k\",\"Sxx_v\",\"Syy_v\",\"Szz_v\",\"Sxy_v\",\"Sxz_v\",\"Syz_v\" \n");
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", nx, ny, nz);

      for (k=0; k<nz; k++)
        for (j=0; j<ny; j++)
          for (i=0; i<nx; i++){
            x = xlo + (i+0.5)*dx;
            y = ylo + (j+0.5)*dy;
            z = zlo + (k+0.5)*dz;
            fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf \n",x, y, z,vptmp[i][j][k][0],vptmp[i][j][k][1],vptmp[i][j][k][2],vptmp[i][j][k][3],vptmp[i][j][k][4],vptmp[i][j][k][5],sptmp[i][j][k][0],sptmp[i][j][k][1],sptmp[i][j][k][2],sptmp[i][j][k][3],sptmp[i][j][k][4],sptmp[i][j][k][5]);
          }
      fclose(out_stat);

      sprintf(f_name,"poly2_%s."BIGINT_FORMAT".plt",fname,step); 
      out_stat=fopen(f_name,"w");
      fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"Sxx_1\",\"Syy_1\",\"Szz_1\",\"Sxy_1\",\"Sxz_1\",\"Syz_1\",\"Sxx_2\",\"Syy_2\",\"Szz_2\",\"Sxy_2\",\"Sxz_2\",\"Syz_2\"  \n");
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", nx, ny, nz);

      for (k=0; k<nz; k++)
        for (j=0; j<ny; j++)
          for (i=0; i<nx; i++){
            x = xlo + (i+0.5)*dx;
            y = ylo + (j+0.5)*dy;
            z = zlo + (k+0.5)*dz;
            fprintf(out_stat,"%lf %lf %lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf %15.10lf \n",x, y, z,sp1tmp[i][j][k][0],sp1tmp[i][j][k][1],sp1tmp[i][j][k][2],sp1tmp[i][j][k][3],sp1tmp[i][j][k][4],sp1tmp[i][j][k][5],sp2tmp[i][j][k][0],sp2tmp[i][j][k][1],sp2tmp[i][j][k][2],sp2tmp[i][j][k][3],sp2tmp[i][j][k][4],sp2tmp[i][j][k][5]);
	  }
      fclose(out_stat);
    }     
  } 
}

/* ---------------------------------------------------------------------- */

