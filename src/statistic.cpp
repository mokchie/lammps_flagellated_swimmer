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
#include <cstring>
#include <cstdlib>
#include "statistic.h"
#include "domain.h"
#include "group.h"
#include "error.h"
#include "comm.h"
#include "force.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
Statistic::Statistic(LAMMPS *lmp, int narg, char **arg) : Pointers(lmp)
{
  int n = strlen(arg[0]) + 1;
  style = new char[n];
  strcpy(style,arg[0]);

  int igroup = group->find(arg[1]);
  if (igroup == -1) error->one(FLERR,"Group ID for Statistic does not exist");  
  groupbit = group->bitmask[igroup];
  cyl_ind = force->inumeric(FLERR,arg[2]);
  nx = force->inumeric(FLERR,arg[3]);
  ny = force->inumeric(FLERR,arg[4]);
  nz = force->inumeric(FLERR,arg[5]);
  st_start = force->inumeric(FLERR,arg[6]);
  step_each = force->inumeric(FLERR,arg[7]); 
  dump_each = force->inumeric(FLERR,arg[8]);  
  if ((nx < 1)||(ny < 1)||(nz < 1)) error->all(FLERR,"Illegal division for statistics"); 
  if (narg == 16){
    xlo = force->numeric(FLERR,arg[9]);
    xhi = force->numeric(FLERR,arg[10]);
    ylo = force->numeric(FLERR,arg[11]);
    yhi = force->numeric(FLERR,arg[12]);
    zlo = force->numeric(FLERR,arg[13]);
    zhi = force->numeric(FLERR,arg[14]);
    sprintf(fname,arg[15]);    
  }
  else{
    xlo = domain->boxlo[0];
    xhi = domain->boxhi[0];
    ylo = domain->boxlo[1];
    yhi = domain->boxhi[1];
    zlo = domain->boxlo[2];
    zhi = domain->boxhi[2];
    sprintf(fname,arg[9]);     
  }
  if (cyl_ind){
    if ((xlo >= xhi)||(yhi <= 0.0)) error->all(FLERR,"Illegal coordinates for statistics");
  } else{
    if ((xlo >= xhi)||(ylo >= yhi)||(zlo >= zhi)) error->all(FLERR,"Illegal coordinates for statistics");
  }

  if (st_start < 0) error->all(FLERR,"Invalid statistic time start");
  if (step_each <= 0) error->all(FLERR,"Invalid statistic calculation frequency");
  if (dump_each <= 0) error->all(FLERR,"Invalid statisitic output frequency");
}

/* ---------------------------------------------------------------------- */

Statistic::~Statistic()
{
  delete [] style;
}

/* ---------------------------------------------------------------------- */

void Statistic::init()
{
  if (cyl_ind){
    if ((xlo >= xhi)||(yhi <= 0.0)) error->all(FLERR,"Illegal coordinates for statistics");
  } else{
    if ((xlo >= xhi)||(ylo >= yhi)||(zlo >= zhi)) error->all(FLERR,"Illegal coordinates for statistics");
  }

  xs = xhi - xlo;
  ys = yhi - ylo;
  zs = zhi - zlo;
  dxpm1 = nx/xs;
  if (cyl_ind){ 
    dypm1 = ny/yhi;
    dzpm1 = 0.5*nz/M_PI;
  } else{
    dypm1 = ny/ys;
    dzpm1 = nz/zs;
  }
  dx = 1.0/dxpm1;
  dy = 1.0/dypm1;
  dz = 1.0/dzpm1;
  xper = domain->xperiodic;
  yper = domain->yperiodic;
  zper = domain->zperiodic;
  dxlo = domain->boxlo[0];
  dxhi = domain->boxhi[0];
  dylo = domain->boxlo[1];
  dyhi = domain->boxhi[1];
  dzlo = domain->boxlo[2];
  dzhi = domain->boxhi[2];
  dxs = domain->xprd;
  dys = domain->yprd;
  dzs = domain->zprd;
}


/* ---------------------------------------------------------------------- */

int Statistic::map_index(double x, double y, double z)
{
  double theta, theta1,rr;
  int ind = 0;
  jv = 0;
  
  if (x<dxlo || x>=dxhi || y<dylo || y>=dyhi || z<dzlo || z>=dzhi){
    if (xper) {
      while (x >= dxhi){
        x -= dxs;
        if (comm->le){
          y -= comm->shift;
          jv--;
          while (y >= dyhi)
            y -= dys;
          while (y < dylo)
            y += dys;
        }
      }   
      while (x < dxlo){
        x += dxs;
        if (comm->le){
          y += comm->shift;
          jv++;
          while (y >= dyhi)
            y -= dys;
          while (y < dylo)
            y += dys;
        }
      }
    } 
    if (yper) {
      while (y >= dyhi) 
        y -= dys;
      while (y < dylo) 
        y += dys;
    }
    if (zper) {
      while (z >= dzhi) 
        z -= dzs;
      while (z < dzlo) 
        z += dzs;
    }
  }

  if (cyl_ind){
    rr = sqrt((y-ylo)*(y-ylo) + (z-zlo)*(z-zlo));
    if (x>=xlo && x<xhi && rr<yhi){
      theta = acos((y-ylo)/rr);
      theta1 = asin((z-zlo)/rr);
      if (theta1 < 0.0)
        theta = 2.0*M_PI - theta;
      is = static_cast<int> ((x - xlo)*dxpm1);
      js = static_cast<int> (rr*dypm1);
      ks = static_cast<int> (theta*dzpm1);
      if (ks>=nz) ks = nz-ks;
      xx = x; yy = y; zz = z;
      ind = 1;
    }
  } else{ 
    if (x>=xlo && x<xhi && y>=ylo && y<yhi && z>=zlo && z<zhi){  
      is = static_cast<int> ((x - xlo)*dxpm1);
      js = static_cast<int> ((y - ylo)*dypm1);
      ks = static_cast<int> ((z - zlo)*dzpm1);
      xx = x; yy = y; zz = z;
      ind = 1;
    }
  }
  return ind;
} 

/* ---------------------------------------------------------------------- */

