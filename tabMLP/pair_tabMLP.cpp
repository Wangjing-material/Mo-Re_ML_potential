/* -------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
-------------------------------------------------------------------- */

/* --------------------------------------------------------------------
   Contributing author: Jesper Byggmastar (Univ. of Helsinki)
   Based on MGP pair style from flare https://github.com/mir-group/flare by:
        Contributing authors: Lixin Sun (Harvard) Yu Xie (Harvard)
-------------------------------------------------------------------- */

#include "pair_tabMLP.h"
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "tokenizer.h"
#include "potential_file_reader.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace std;
using namespace LAMMPS_NS;

/* ----------------------------------------------------------------- */

PairTabMLP::PairTabMLP(LAMMPS *lmp) : Pair(lmp) {
  // no restart info
  restartinfo = 0;
  // give a warning if bonds/angles/dihedrals are used with mmf
  manybody_flag = 1;
  single_enable = 1;
  // not sure this is good
  reinitflag = 0;
  embedstep = -1;

  setflag = NULL;
  e0 = NULL;
  map2b = NULL;
  map3b = NULL;
  map_rho =	NULL;
  compute2b = true;
  compute3b = true;
  compute_rho = true;

  elements = NULL;
  n_2body = 0;
  n_3body = 0;
  n_rho = 0;
  lo_2body = NULL;
  hi_2body = NULL;
  lo_3body = NULL;
  hi_3body = NULL;
  lo_rho = NULL;
  hi_rho = NULL;
  grid_2body = NULL;
  grid_3body = NULL;
  grid_rho = NULL;
  ecoeff_2body = NULL;
  fcoeff_2body = NULL;
  ecoeff_3body = NULL;
  fcoeff_3body = NULL;
  ecoeff_rho = NULL;
  fcoeff_rho = NULL;
  
  rho_value=NULL;
  param=NULL;
  nmax = 0;
  
  cutmax = 0.0;
  cutsq = NULL;
  cut2bsq = NULL;
  cut3bsq = NULL;
  cutrhosq = NULL;
  cutshortsq = 0;

  maxshort = 20;
  neighshort = NULL;
  
  comm_forward = 24;
  comm_reverse = 0;

  Ad[0][0] = -1.0 / 6.0;
  Ad[0][1] = 3.0 / 6.0;
  Ad[0][2] = -3.0 / 6.0;
  Ad[0][3] = 1.0 / 6.0;
  Ad[1][0] = 3.0 / 6.0;
  Ad[1][1] = -6.0 / 6.0;
  Ad[1][2] = 0.0 / 6.0;
  Ad[1][3] = 4.0 / 6.0;
  Ad[2][0] = -3.0 / 6.0;
  Ad[2][1] = 3.0 / 6.0;
  Ad[2][2] = 3.0 / 6.0;
  Ad[2][3] = 1.0 / 6.0;
  Ad[3][0] = 1.0 / 6.0;
  Ad[3][1] = 0.0 / 6.0;
  Ad[3][2] = 0.0 / 6.0;
  Ad[3][3] = 0.0 / 6.0;
  // {-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0},
  // { 3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0},
  // {-3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0},
  // { 1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0}

  // initialize with 0.0
  for (int j = 0; j < 4; j++) {
    Bd[j] = 0.0;
    Cd[j] = 0.0;
    basis[j] = 0.0;
    for (int i = 0; i < 4; i++) {
      dAd[j][i] = 0.0;
      d2Ad[j][i] = 0.0;
    }
  }

  // calculate values
  for (int j = 0; j < 4; j++) {
    for (int i = 1; i < 4; i++) {
      dAd[j][i] = Ad[j][i - 1] * (4 - i);
    }
  }

  for (int j = 0; j < 4; j++) {
    for (int i = 1; i < 4; i++) {
      d2Ad[j][i] = dAd[j][i - 1] * (4 - i);
    }
  }

  for (int i = 1; i < 4; i++) {
    Bd[i] = 3 * Ad[i][0] + 2 * Ad[i][1] + Ad[i][2];
    Cd[i] = Ad[i][0] + Ad[i][1] + Ad[i][2] + Ad[i][3];
  }
}

/* --------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
-------------------------------------------------------------------- */

PairTabMLP::~PairTabMLP() {
  int me;
  // why comm->me does not work here?
  MPI_Comm_rank(world, &me);

  if (copymode)
    return;

  if (elements) {
    for (int i = 1; i <= atom->ntypes; i++) {
      if (elements[i])
        delete[] elements[i];
    }
    delete[] elements;
  }

  memory->destroy(lo_2body);
  memory->destroy(hi_2body);
  memory->destroy(grid_2body);

  memory->destroy(lo_3body);
  memory->destroy(hi_3body);
  memory->destroy(grid_3body);
  
  memory->destroy(lo_rho);
  memory->destroy(hi_rho);
  memory->destroy(grid_rho);
  
  memory->destroy(rho_value);
  memory->destroy(param);

  // more to do
  if (fcoeff_2body) {
    for (int i = 0; i < n_2body; i++) {
      memory->destroy(fcoeff_2body[i]);
      //   memory->destroy(ecoeff_2body[i]);
    }
    memory->sfree(fcoeff_2body);
    // memory->sfree(ecoeff_2body);
  }

  if (fcoeff_3body) {
    for (int i = 0; i < n_3body; i++) {
      memory->destroy(fcoeff_3body[i]);
      //     memory->destroy(ecoeff_3body[i]);
    }
    memory->sfree(fcoeff_3body);
    // memory->sfree(ecoeff_3body);
  }
  
  if (fcoeff_rho) {
    for (int i = 0; i < n_rho; i++) {
      memory->destroy(fcoeff_rho[i]);
      //   memory->destroy(ecoeff_2body[i]);
    }
    memory->sfree(fcoeff_rho);
    // memory->sfree(ecoeff_2body);
  }

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
  }
  memory->destroy(cut2bsq);
  memory->destroy(cut3bsq);
  memory->destroy(cutrhosq);
  memory->destroy(e0);
  memory->destroy(map2b);
  memory->destroy(map3b);
  memory->destroy(map_rho);
}

/* ----------------------------------------------------------------- */

void PairTabMLP::compute(int eflag, int vflag) {
  int me;
  MPI_Comm_rank(world, &me);
  
  /// Grow per-atom array if necessary
  if (atom->nmax > nmax) {
	memory->destroy(rho_value);
    nmax = atom->nmax;
    memory->create(rho_value,nmax,24,"pair:rho_value");	
  }

  int i, j, k, inum, jnum, itype, jtype, ktype;
  double xtmp, ytmp, ztmp, del[3];
  double evdwl = 0;
  double rsq, r;
  double cutoff;
  int *ilist, *jlist, *numneigh, **firstneigh;
  //double Param[24]={-26826.2,550.639,-89.1629,33.7798,-15.8106,3.88996,41486.7,-848.469,110.874,-30.6446,12.7121,-3.1685,29083,-518.837,27.8574
//,53.6607,-62.3615,25.4932,-26261.3,485.615,-57.9012,-2.4866,20.6598,-9.01532};
  tagint *tag = atom->tag;
  tagint itag, jtag;

  // set up flags for total and per-atom energies/virials
  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  // int nall = nlocal + atom->nghost;
  // int ntype_p1 = atom->ntypes + 1;

  bool newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute forces on each atom
  // loop over neighbors of my atoms
  for (int ii = 0; ii < inum; ii++) {
	  double evdwl_rho=0;
	  
    i = ilist[ii];
    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
	for(int m=0;m<24;m++)
	  {
	    rho_value[i][m]=0;
	  }

    // energy of isolated atom
    // misuse ev_tally with zero forces to avoid virial tally (TODO is there a better way?)
    if (evflag)
      ev_tally(i, i, nlocal, newton_pair, e0[itype], 0.0, 0.0, 0.0, 0.0, 0.0);

    jlist = firstneigh[i];
    jnum = numneigh[i];

    int numshort = 0;
    for (int jj = 0; jj < jnum; jj++) {
      double fpair = 0;
      j = jlist[jj];

      j &= NEIGHMASK;
      jtype = type[j];
      jtag = tag[j];

      del[0] = x[j][0] - xtmp;
      del[1] = x[j][1] - ytmp;
      del[2] = x[j][2] - ztmp;
      rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];

      if (compute3b) {
        // accumulate short list for 3body interactions
        if (rsq < cutshortsq) {
          neighshort[numshort++] = j;
          if (numshort >= maxshort) {
            maxshort += maxshort / 2;
            memory->grow(neighshort, maxshort, "pair:neighshort");
          }
        }
      }

      if (compute2b) {
        int mapid = map2b[itype][jtype];
        if (mapid != -1) {
          if (rsq < cut2bsq[mapid]) {

            jtype = type[j];
            r = sqrt(rsq);

            // energy and forces are both computed
            eval_cubic_splines_1d(lo_2body[mapid], hi_2body[mapid],
                                  grid_2body[mapid], fcoeff_2body[mapid], r,
                                  &evdwl, &fpair);

            // evdwl *= 1;
            fpair *= 1 / r;

            double fx, fy, fz;
            fx = del[0] * fpair;
            fy = del[1] * fpair;
            fz = del[2] * fpair;

            // newton on always: sum up forces on both i and j
            f[i][0] += fx;
            f[i][1] += fy;
            f[i][2] += fz;
            f[j][0] -= fx;
            f[j][1] -= fy;
            f[j][2] -= fz;

            // tally energy compute virials
            if (evflag)
              ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, -fx, -fy, -fz,
                           del[0], del[1], del[2]);
          }
        }
      }
	        if (compute_rho) {
        int mapid = map_rho[itype][jtype];
        if (mapid != -1) {
          if (rsq < cutrhosq[mapid]) {

            jtype = type[j];
            r = sqrt(rsq);
			for(int k=0;k<6;k++)
			{
            // energy computed
            eval_cubic_splines_1d_energy(lo_rho[mapid], hi_rho[mapid],
                                  grid_rho[mapid], fcoeff_rho[mapid*6+k], r,
                                  &evdwl);
			
			rho_value[i][mapid*6+k]+=evdwl;
			
            }
          
          }
        }
      }

    } // j
	if(compute_rho)
	{
	 for(int k=0;k<6;k++)
    {	

     double rho_sum = rho_value[i][k]+ 1.0;
     evdwl_rho+= (rho_sum*(log(rho_sum)-1.0)+1.0)*param[k];
	  
     double rho_sum1 = rho_value[i][k+18]+ 1.0;
     evdwl_rho+= (rho_sum1*(log(rho_sum1)-1.0)+1.0)*param[k+6];

     double rho_sum2 =rho_value[i][k]+rho_value[i][k+6]+ 1.0;
     evdwl_rho+= (rho_sum2*(log(rho_sum2)-1.0)+1.0)*param[k+12];

     double rho_sum3 = rho_value[i][k+12]+rho_value[i][k+18]+ 1.0;
     evdwl_rho+= (rho_sum3*(log(rho_sum3)-1.0)+1.0)*param[k+18];
	 rho_value[i][k] = param[k]*log(rho_sum)+param[k+12]*log(rho_sum2);
	 rho_value[i][k+18] = param[k+6]*log(rho_sum1)+param[k+18]*log(rho_sum3);;
     rho_value[i][k+6] = param[k+12]*log(rho_sum2);
     rho_value[i][k+12] = param[k+18]*log(rho_sum3);
	}
	
	if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl_rho,0.0,0.0,0.0,0.0, 0.0);
	
	}
	
    if (compute3b) {
      // three-body interactions
      // skip immediately if ij is not within cutoff
      double delr1[3], delr2[3], delr12[3];
      double rsq1, rsq2, rsq12, rij, rik, cos_angle;
      double fji[3], fki[3];
      double fcosthetaijk_ij[3], fcosthetaijk_ik[3];

      for (int jj = 0; jj < numshort; jj++) {
        j = neighshort[jj];
        jtype = type[j];

        delr1[0] = x[j][0] - xtmp;
        delr1[1] = x[j][1] - ytmp;
        delr1[2] = x[j][2] - ztmp;
        rsq1 = delr1[0] * delr1[0] + delr1[1] * delr1[1] + delr1[2] * delr1[2];

        for (int kk = jj + 1; kk < numshort; kk++) {
          double ftriplet[3] = {0, 0, 0};
          k = neighshort[kk];
          ktype = type[k];

          // TODO remove mapid2, it's not even used and read_file ensures ijk and ikj both exist
          int mapid1 = map3b[itype][jtype][ktype];
          int mapid2 = map3b[itype][ktype][jtype];
          // 3 body coef not found, then continue
          if (mapid1 == -1 || mapid2 == -1)
            error->all(FLERR, "tabMLP coeff: mapid is not found.");

          cutoff = cut3bsq[mapid1];
          if (rsq1 >= cutoff)
            continue;

          delr2[0] = x[k][0] - xtmp;
          delr2[1] = x[k][1] - ytmp;
          delr2[2] = x[k][2] - ztmp;
          rsq2 =
              delr2[0] * delr2[0] + delr2[1] * delr2[1] + delr2[2] * delr2[2];
          if (rsq2 >= cutoff)
            continue;

          delr12[0] = x[k][0] - x[j][0];
          delr12[1] = x[k][1] - x[j][1];
          delr12[2] = x[k][2] - x[j][2];
          rsq12 = delr12[0] * delr12[0] + delr12[1] * delr12[1] +
                  delr12[2] * delr12[2];

          // TODO flag jk_cutoff[i][j][k] ?
          //if (rsq12 >= cutoff)
            //continue;

          // compute bonds
          rij = sqrt(rsq1);
          rik = sqrt(rsq2);
          // rjk = sqrt(rsq12);

          // compute angle
          cos_angle = (rsq1+rsq2-rsq12)/2./(rij*rik);
          if (cos_angle > 1) cos_angle = 1; // prevent numerical error
          if (cos_angle < -1) cos_angle = -1;

          // compute spline
          eval_cubic_splines_3d(lo_3body[mapid1], hi_3body[mapid1],
                                grid_3body[mapid1], fcoeff_3body[mapid1], rij,
                                rik, cos_angle, &evdwl, ftriplet);
          double f_ij, f_ik;


          f_ij = ftriplet[0] / rij;
          f_ik = ftriplet[1] / rik;

          // using fji = -fij to get correct sign for input to virial calculation ev_tally*
          fji[0] = -f_ij * delr1[0]; // delr1, delr2, not unit vector
          fji[1] = -f_ij * delr1[1];
          fji[2] = -f_ij * delr1[2];
          fki[0] = -f_ik * delr2[0];
          fki[1] = -f_ik * delr2[1];
          fki[2] = -f_ik * delr2[2];
          fcosthetaijk_ij[0] = ftriplet[2] / rij * (delr2[0] / rik - delr1[0] / rij * cos_angle);
          fcosthetaijk_ij[1] = ftriplet[2] / rij * (delr2[1] / rik - delr1[1] / rij * cos_angle);
          fcosthetaijk_ij[2] = ftriplet[2] / rij * (delr2[2] / rik - delr1[2] / rij * cos_angle);
          fcosthetaijk_ik[0] = ftriplet[2] / rik * (delr1[0] / rij - delr2[0] / rik * cos_angle);
          fcosthetaijk_ik[1] = ftriplet[2] / rik * (delr1[1] / rij - delr2[1] / rik * cos_angle);
          fcosthetaijk_ik[2] = ftriplet[2] / rik * (delr1[2] / rij - delr2[2] / rik * cos_angle);

          // final forces on j and k due to i. force on i is now: -fji - fki
          fji[0] -= fcosthetaijk_ij[0];
          fji[1] -= fcosthetaijk_ij[1];
          fji[2] -= fcosthetaijk_ij[2];
          fki[0] -= fcosthetaijk_ik[0];
          fki[1] -= fcosthetaijk_ik[1];
          fki[2] -= fcosthetaijk_ik[2];

          f[i][0] += -fji[0] - fki[0];
          f[i][1] += -fji[1] - fki[1];
          f[i][2] += -fji[2] - fki[2];
          f[j][0] += fji[0];
          f[j][1] += fji[1];
          f[j][2] += fji[2];
          f[k][0] += fki[0];
          f[k][1] += fki[1];
          f[k][2] += fki[2];

          if (evflag) {
            // tally full energy of i
            // misuse ev_tally with zero forces to avoid virial tally (is there a better way?)
            ev_tally(i, i, nlocal, newton_pair, evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);
			
          }
          if (vflag_atom) {
            v_tally3(i, j, k, fji, fki, delr1, delr2);
          }

        } // k

      } // j
    } // compute3b
  } // i
  
    comm->forward_comm(this);
	
 if (compute_rho)
	{

  for (int ii = 0; ii < inum; ii++) {
	  
	  i = ilist[ii];
    itag = tag[i];
	xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
	
    jlist = firstneigh[i];
    jnum = numneigh[i];
	double fpair1=0.0;
 
    for (int jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      j &= NEIGHMASK;
      jtype = type[j];
      jtag = tag[j];
	  del[0] = x[j][0] - xtmp;
      del[1] = x[j][1] - ytmp;
      del[2] = x[j][2] - ztmp;
      rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];


 
		int mapid = map_rho[itype][jtype];
        if (mapid != -1) {
          if (rsq < cutrhosq[mapid]) {
			double frho_total=0.0;
			r = sqrt(rsq);
	  for(int k=0;k<6;k++)
    {	
     double frho=0.0;
	 double Urho_prime_ij=0.0;
       eval_cubic_splines_1d_force(lo_rho[mapid], hi_rho[mapid],
                                  grid_rho[mapid], fcoeff_rho[mapid*6+k], r,
                                  &frho);
     if(mapid==0)
		Urho_prime_ij=0.5*(rho_value[i][k]+rho_value[j][k]);
	
	  else if(mapid==3)
        Urho_prime_ij=0.5*(rho_value[i][k+18]+rho_value[j][k+18]);
	  
	  else if(mapid==1)
        Urho_prime_ij=rho_value[i][k+6];

	  else if(mapid==2)
        Urho_prime_ij=rho_value[i][k+12];
	
	frho_total += Urho_prime_ij*frho;
	}
		  
	  frho_total *= 1 / r;
      fpair1+=frho_total;
            double fx, fy, fz;
            fx = del[0] * frho_total;
            fy = del[1] * frho_total;
            fz = del[2] * frho_total;
			
            // newton on always: sum up forces on both i and j
            f[i][0] += fx;
            f[i][1] += fy;
            f[i][2] += fz;
            f[j][0] -= fx;
            f[j][1] -= fy;
            f[j][2] -= fz;

            // tally energy compute virials	
           if (evflag)
		   {
              ev_tally_xyz(i, j, nlocal, newton_pair, 0.0, 0.0, -fx, -fy, -fz,
                           del[0], del[1], del[2]);
		   }
            
    }
	}
	}//j
	}//i
	
  }
  if (vflag_fdotr)
    virial_fdotr_compute();
}

/* --------------------------------------------------------------------
   allocate all arrays
-------------------------------------------------------------------- */

void PairTabMLP::allocate() {
  // int me = comm->me;
  allocated = 1;

  int np = atom->ntypes + 1;
  memory->create(setflag, np, np, "pair:setflag");
  memory->create(cutsq, np, np, "pair:cutsq");
  memory->create(neighshort, maxshort, "pair:neighshort");

  memset(&setflag[0][0], 0, np * np * sizeof(int));
  memset(&cutsq[0][0], 0, np * np * sizeof(double));
}

/* -------------------------------------------------------------------
   global settings
-------------------------------------------------------------------- */

void PairTabMLP::settings(int narg, char ** /*arg*/) {
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

/* --------------------------------------------------------------------
   set coeffs for one or more type pairs
-------------------------------------------------------------------- */

void PairTabMLP::coeff(int narg, char **arg) {
  int me = comm->me;

  int n3 = 3 + atom->ntypes;
  if (narg < n3)
    error->all(FLERR, "tabMLP coeff: Incorrect args for pair coefficients");

  // TO DO, this should be compatible with other potentials?
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "tabMLP coeff: Incorrect args for pair coefficients");

  if (!allocated)
    allocate();

  // parse pair of atom types
  elements = new char *[atom->ntypes + 1];

  int n;
  for (int i = 3; i < n3; i++) {
    n = strlen(arg[i]);
    elements[i - 2] = new char[n];
    strcpy(elements[i - 2], arg[i]);
    if (me == 0 && screen)
      fprintf(screen, "tabMLP coeff: type %d is element %s\n", i - 2,
              elements[i - 2]);
  }

  if (narg >= (4 + atom->ntypes)) {
    if (strcmp(arg[n3], "yes") == 0)
      compute2b = true;
    else if (strcmp(arg[n3], "no") == 0)
      compute2b = false;
    else {
      error->all(FLERR, "tabMLP coeff:Incorrect args for pair coefficients. The "
                        "argument should be either yes or no");
    }
  }

  if (narg > (4 + atom->ntypes)) {
    if (strcmp(arg[n3 + 1], "yes") == 0)
      compute3b = true;
    else if (strcmp(arg[n3 + 1], "no") == 0)
      compute3b = false;
    else {
      error->all(FLERR, "tabMLP coeff:Incorrect args for pair coefficients. The "
                        "argument should be either yes or no");
    }
  }
  
  if (narg > (5 + atom->ntypes)) {
    if (strcmp(arg[n3+2], "yes") == 0)
      compute_rho = true;
    else if (strcmp(arg[n3+2], "no") == 0)
      compute_rho = false;
    else {
      error->all(FLERR, "tabMLP coeff:Incorrect args for pair coefficients. The "
                        "argument should be either yes or no");
    }
  }

  // read_file(arg[2]);
  if (me == 0)
    read_file(arg[2]);
  bcast_table();

  if (me == 0 && screen)
    fprintf(screen, "tabMLP coeff: done reading coefficients\n");
}


void PairTabMLP::read_file(char *filename) {

  if (screen)
    fprintf(screen, "tabMLP read_file: reading potential file...\n");

  PotentialFileReader reader(lmp, filename, "tabmlp", unit_convert_flag);
  double e0_init = 1e9;
  int ntype_p1 = atom->ntypes + 1;
  try {
    reader.skip_line(); // 1st line is a comment
    reader.skip_line(); // 2nd line as a comment

    //// extract e0 for each element from 3rd line
    ValueTokenizer values = reader.next_values(1);
    n_e0 = values.next_int();
    memory->create(e0, atom->ntypes + 1, "pair:e0");
    for (int i = 0; i < atom->ntypes + 1; i++) {
      e0[i] = e0_init;  // initialise crazy value
    }
    // read "element e0" pairs one by one
    for (int i = 0; i < n_e0; i++) {
      const std::string word = values.next_string();
      double e0_ele;
      e0_ele = values.next_double();
      // map e0 to element type
      for (int itype = 1; itype <= atom->ntypes; itype++) {
        if (strcmp(elements[itype], word.c_str()) == 0) {
          e0[itype] = e0_ele;
          // TODO print more decimals for e0 to avoid confusion?
          if (screen)
            fprintf(screen,
                "tabMLP read_file: e0 for element %s type %d is %lg\n",
                elements[itype], itype, e0[itype]);
        }
      }
    }
    // check that we have e0 for all elements
    for (int itype = 1; itype <= atom->ntypes; itype++) {
      if (e0[itype] > e0_init - 1) {
        char str[128];
        snprintf(str, 128, "tabMLP read_file: e0 for type %d element %s not found",
                 itype, elements[itype]);
        error->one(FLERR, str);
      }
    }
	

    // 4th line is number of 2b and 3b potentials to read
    values = reader.next_values(2);
    n_2body = values.next_int();
    n_3body = values.next_int();
	n_rho = values.next_int();
    if (screen)
      fprintf(screen,
              "tabMLP read_file: expecting to read in %d 2b potentials and %d 3b "
              "potentials\n", n_2body, n_3body);

    // initialise 2b details
    if (n_2body > 0) {
      memory->create(lo_2body, n_2body, "pair:lo_2body");
      memory->create(hi_2body, n_2body, "pair:hi_2body");
      memory->create(grid_2body, n_2body, "pair:grid_2body");
      memory->create(map2b, ntype_p1, ntype_p1, "pair:map2b");
      memory->create(cut2bsq, n_2body, "pair:cut2bsq");

      for (int i = 1; i < ntype_p1; i++) {
        for (int j = 1; j < ntype_p1; j++) {
          map2b[i][j] = -1;
        }
      }

      // ????????????????????/pair: 3body?
      fcoeff_2body = (double **)memory->smalloc(n_2body * sizeof(double *),
                                                "pair:fcoeff_3body");
     // ecoeff_2body = (double **) memory->smalloc(n_2body*sizeof(double
      // *),"pair:ecoeff_3body");
    } else {
      compute2b = false;
    }

    // initialise 3b details
    if (n_3body > 0) {
      memory->create(lo_3body, n_3body, 3, "pair:lo_3body");
      memory->create(hi_3body, n_3body, 3, "pair:hi_3body");
      memory->create(grid_3body, n_3body, 3, "pair:grid_3body");
      memory->create(map3b, ntype_p1, ntype_p1, ntype_p1, "pair:map3b");
      memory->create(cut3bsq, n_3body, "pair:cut3bsq");
      for (int i = 1; i < ntype_p1; i++) {
        for (int j = 1; j < ntype_p1; j++) {
          for (int k = 1; k < ntype_p1; k++) {
            map3b[i][j][k] = -1;
          }
        }
      }
      fcoeff_3body = (double **)memory->smalloc(n_3body * sizeof(double *),
                                                "pair:coeff_2body");
      // ecoeff_3body = (double **) memory->smalloc(n_3body*sizeof(double
      // *),"pair:ecoeff_2body");
    } else {
      compute3b = false;
    }
	
	if (n_rho > 0) {
      memory->create(lo_rho, n_rho/6, "pair:lo_rho");
      memory->create(hi_rho, n_rho/6, "pair:hi_rho");
      memory->create(grid_rho, n_rho/6, "pair:grid_rho");
      memory->create(map_rho, ntype_p1, ntype_p1, "pair:map_rho");
      memory->create(cutrhosq, n_rho/6, "pair:cutrhosq");
	  

      for (int i = 1; i < ntype_p1; i++) {
        for (int j = 1; j < ntype_p1; j++) {
          map_rho[i][j] = -1;
        }
      }
      fcoeff_rho = (double **)memory->smalloc(n_rho * sizeof(double *),
                                                "pair:coeff_rho");   
      // ecoeff_2body = (double **) memory->smalloc(n_2body*sizeof(double
      // *),"pair:ecoeff_3body");
    } else {
      compute_rho = false;
    }
	//5th line leaning param of rho
	values = reader.next_values(24);
	memory->create(param, n_rho, "pair:param");
	for(int i=0;i<n_rho;i++)
	{
		param[i]=values.next_double();
		if(screen)
			fprintf(screen,"Param[%d]=%lf",i,param[i]);
	}

    // read 2b potentials
    cutmax = 0;
    for (int idx = 0; idx < n_2body; idx++) {
      //char ele1[10], ele2[10];
      double rmin, rmax;
      int order;
  
      // read header
      values = reader.next_values(5);
      std::string ele1 = values.next_string();
      std::string ele2 = values.next_string();
      rmin = values.next_double();
      rmax = values.next_double();
      order = values.next_int();
      if (screen)
        fprintf(screen, "tabMLP read_file: reading 2b potential %d: %s %s\n",
                idx, ele1.c_str(), ele2.c_str());

      bool type1[atom->ntypes + 1];
      bool type2[atom->ntypes + 1];
      for (int itype = 1; itype <= atom->ntypes; itype++) {
        if (strcmp(elements[itype], ele1.c_str()) == 0)
          type1[itype] = true;
        else
          type1[itype] = false;
        if (strcmp(elements[itype], ele2.c_str()) == 0)
          type2[itype] = true;
        else
          type2[itype] = false;
      }

      double rsq = rmax * rmax;
      cut2bsq[idx] = rsq;
      for (int itype = 1; itype <= atom->ntypes; itype++) {
        for (int jtype = 1; jtype <= atom->ntypes; jtype++) {
          if (type1[itype] && type2[jtype]) {
            map2b[itype][jtype] = idx;
            if (compute2b) {
              setflag[itype][jtype] = 1;
              setflag[jtype][itype] = 1;
            }
            cutsq[itype][jtype] = rsq;
            cutsq[jtype][itype] = rsq;
          }
        }
      }

      if (cutmax < rsq)
        cutmax = rsq;

      lo_2body[idx] = rmin;
      hi_2body[idx] = rmax;
      grid_2body[idx] = order;
      memory->create(fcoeff_2body[idx], order + 2, "pair:2body_coeff");
      // memory->create(ecoeff_2body[idx],order+2,"pair:2body_ecoeff");

      // read in energies
      reader.next_dvector(&fcoeff_2body[idx][0], order + 2);
    }

    // read 3b potentials
    for (int idx = 0; idx < n_3body; idx++) {
      //char ele1[10], ele2[10], ele3[10];
      double a[3], b[3];
      int order[3];

      // read header
      values = reader.next_values(12);
      std::string ele1 = values.next_string();
      std::string ele2 = values.next_string();
      std::string ele3 = values.next_string();
      a[0] = values.next_double();
      a[1] = values.next_double();
      a[2] = values.next_double();
      b[0] = values.next_double();
      b[1] = values.next_double();
      b[2] = values.next_double();
      order[0] = values.next_int();
      order[1] = values.next_int();
      order[2] = values.next_int();
      if (screen)
        fprintf(screen, "tabMLP read_file: reading 3b potential %d: %s %s %s\n",
                idx, ele1.c_str(), ele2.c_str(), ele3.c_str());

      bool type1[atom->ntypes + 1];
      bool type2[atom->ntypes + 1];
      bool type3[atom->ntypes + 1];
      for (int itype = 1; itype <= atom->ntypes; itype++) {
        if (strcmp(elements[itype], ele1.c_str()) == 0)
          type1[itype] = true;
        else
          type1[itype] = false;
        if (strcmp(elements[itype], ele2.c_str()) == 0)
          type2[itype] = true;
        else
          type2[itype] = false;
        if (strcmp(elements[itype], ele3.c_str()) == 0)
          type3[itype] = true;
        else
          type3[itype] = false;
      }

      double rsq = b[0] * b[0];
      cut3bsq[idx] = rsq;
      for (int itype = 1; itype <= atom->ntypes; itype++) {
        for (int jtype = 1; jtype <= atom->ntypes; jtype++) {
          for (int ktype = 1; ktype <= atom->ntypes; ktype++) {
            if (type1[itype] && type2[jtype] && type3[ktype]) {
              map3b[itype][jtype][ktype] = idx;
              // no permutation symmetry, we check later and assume symmetry if ikj is missing
              // map3b[itype][ktype][jtype] = idx;
              if (compute3b) {
                setflag[itype][jtype] = 1;
                setflag[jtype][itype] = 1;
                setflag[itype][ktype] = 1;
                setflag[ktype][itype] = 1;
                setflag[jtype][ktype] = 1;
                setflag[ktype][jtype] = 1;
              }
            }
          }
        }
      }

      if (cutmax < rsq)
        cutmax = rsq;
      if (rsq > cutshortsq)
        cutshortsq = rsq;

      lo_3body[idx][0] = a[0];
      lo_3body[idx][1] = a[1];
      lo_3body[idx][2] = a[2];
      hi_3body[idx][0] = b[0];
      hi_3body[idx][1] = b[1];
      hi_3body[idx][2] = b[2];
      grid_3body[idx][0] = order[0];
      grid_3body[idx][1] = order[1];
      grid_3body[idx][2] = order[2];

      int len = (order[0] + 2) * (order[1] + 2) * (order[2] + 2);
      memory->create(fcoeff_3body[idx], len, "pair:3body_coeff");
      // memory->create(ecoeff_3body[idx], len, "pair:3body_ecoeff");

      // read in energies
      reader.next_dvector(&fcoeff_3body[idx][0], len);
    }
	
	// read rho potentials
    for (int idx = 0; idx < (n_rho/6); idx++) {
      //char ele1[10], ele2[10];
      double r_rho_min, r_rho_max;
      int order_rho;
  
      // read header
      values = reader.next_values(5);
      std::string ele1 = values.next_string();
      std::string ele2 = values.next_string();
      r_rho_min = values.next_double();
      r_rho_max = values.next_double();
      order_rho = values.next_int();
      if (screen)
        fprintf(screen, "tabMLP read_file: reading rho potential %d: %s %s\n",
                idx, ele1.c_str(), ele2.c_str());

      bool type1[atom->ntypes + 1];
      bool type2[atom->ntypes + 1];
      for (int itype = 1; itype <= atom->ntypes; itype++) {
        if (strcmp(elements[itype], ele1.c_str()) == 0)
          type1[itype] = true;
        else
          type1[itype] = false;
        if (strcmp(elements[itype], ele2.c_str()) == 0)
          type2[itype] = true;
        else
          type2[itype] = false;
      }

      double rsq_rho = r_rho_max * r_rho_max;
      cutrhosq[idx] = rsq_rho;
      for (int itype = 1; itype <= atom->ntypes; itype++) {
        for (int jtype = 1; jtype <= atom->ntypes; jtype++) {
          if (type1[itype] && type2[jtype]) {
            map_rho[itype][jtype] = idx;
			if(screen)
			  fprintf(screen,"map_rho[%d][%d]=%d\n",itype,jtype,idx);
            if (compute_rho) {
              setflag[itype][jtype] = 1;
            }
            cutsq[itype][jtype] = rsq_rho;
          }
        }
      }

      if (cutmax < rsq_rho)
        cutmax = rsq_rho;

      lo_rho[idx] = r_rho_min;
      hi_rho[idx] = r_rho_max;
      grid_rho[idx] = order_rho;
	  for(int k=0;k<6;k++)
	  {
      memory->create(fcoeff_rho[idx*6+k], order_rho + 2, "pair:rho_coeff");
      // memory->create(ecoeff_2body[idx],order+2,"pair:2body_ecoeff");

      // read in energies
      reader.next_dvector(&fcoeff_rho[idx*6+k][0], order_rho + 2);
	  }
    }
	
  } catch (TokenizerException & e) {
      error->one(FLERR, e.what());
  }


  cutmax = sqrt(cutmax);

  // if (screen) fprintf(screen, "tabMLP read_file: sanity check on setflag\n");
  for (int i = 1; i < ntype_p1; i++) {
    for (int j = 1; j < ntype_p1; j++) {
      if (setflag[i][j] == 0) {
        char str[128]; // read files, warning if some potential is not covered
        snprintf(str, 128,
                 "tabMLP read_file: no tabMLP for type %d %d is defined", i,
                 j);
        error->all(FLERR, str);
        // setflag[i][j] == 1; // can be covered by other potentials
      }
    }
  }

  // if (screen) fprintf(screen, "tabMLP read_file: sanity check on 2b
  // potential\n");
  if (compute2b) {
    for (int i = 1; i < ntype_p1; i++) {
      for (int j = 1; j < ntype_p1; j++) {
        if (map2b[i][j] == -1) {
          char str[128];
          snprintf(
              str, 128,
              "tabMLP read_file: 2b tabMLP for types %d %d is not defined",
              i, j);
          error->warning(FLERR, str);
        }
      }
    }
  }

  // if (screen) fprintf(screen, "tabMLP read_file: sanity check on 3b
  // potential\n");
  if (compute3b) {
    for (int i = 1; i < ntype_p1; i++) {
      for (int j = 1; j < ntype_p1; j++) {
        for (int k = 1; k < ntype_p1; k++) {
          if (map3b[i][j][k] == -1) {
            if (map3b[i][k][j] == -1) {
              char str[128];
              snprintf(str, 128,
                       "tabMLP read_file: 3b tabMLP for types %d %d %d is "
                       "not defined",
                       i, j, k);
              error->warning(FLERR, str);
            } else {
              map3b[i][j][k] = map3b[i][k][j];
            }
          }
        }
      }
    }
  }
  
  if (compute_rho) {
    for (int i = 1; i < ntype_p1; i++) {
      for (int j = 1; j < ntype_p1; j++) {
        if (map_rho[i][j] == -1) {
          char str[128];
          snprintf(
              str, 128,
              "tabMLP read_file: rho tabMLP for types %d %d is not defined",
              i, j);
          error->warning(FLERR, str);
        }
      }
    }
  }

  if (screen)
    fprintf(screen, "tabMLP read_file: done\n");
}

void PairTabMLP::bcast_table() {
  int me; // = comm->me;
  MPI_Comm_rank(world, &me);

  MPI_Bcast(&n_2body, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_3body, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_rho, 1, MPI_INT, 0, world);
  
  int ntype_p1 = atom->ntypes + 1;
  if (me > 0) {
    memory->create(e0, ntype_p1, "pair:e0");
	memory->create(param, n_rho, "pair:param");
    if (n_2body > 0) {
      memory->create(lo_2body, n_2body, "pair:lo_2body");
      memory->create(hi_2body, n_2body, "pair:hi_2body");
      memory->create(grid_2body, n_2body, "pair:grid_2body");
      memory->create(cut2bsq, n_2body, "pair:cut2bsq");
      memory->create(map2b, ntype_p1, ntype_p1, "pair:map2b");
      fcoeff_2body = (double **)memory->smalloc(n_2body * sizeof(double *),
                                                "pair:fcoeff_2body");
      // ecoeff_2body = (double **) memory->smalloc(n_2body*sizeof(double
      // *),"pair:ecoeff_2body");
    }
    if (n_3body > 0) {
      memory->create(lo_3body, n_3body, 3, "pair:lo_3body");
      memory->create(hi_3body, n_3body, 3, "pair:hi_3body");
      memory->create(grid_3body, n_3body, 3, "pair:grid_3body");
      memory->create(cut3bsq, n_3body, "pair:cut3bsq");
      memory->create(map3b, ntype_p1, ntype_p1, ntype_p1, "pair:map3b");
      fcoeff_3body = (double **)memory->smalloc(n_3body * sizeof(double *),
                                                "pair:fcoeff_3body");
      // ecoeff_3body = (double **) memory->smalloc(n_3body*sizeof(double
      // *),"pair:ecoeff_3body");
    }
	if (n_rho > 0) {
      memory->create(lo_rho, n_rho/6, "pair:lo_rho");
      memory->create(hi_rho, n_rho/6, "pair:hi_rho");
      memory->create(grid_rho, n_rho/6, "pair:grid_rho");
      memory->create(cutrhosq, n_rho/6, "pair:cutrhosq");
      memory->create(map_rho, ntype_p1, ntype_p1, "pair:map_rho");
      fcoeff_rho = (double **)memory->smalloc(n_rho * sizeof(double *),
                                                "pair:fcoeff_rho");
      // ecoeff_2body = (double **) memory->smalloc(n_2body*sizeof(double
      // *),"pair:ecoeff_2body");
    }
  }

  MPI_Bcast(cutsq[0], ntype_p1 * ntype_p1, MPI_DOUBLE, 0, world);
  MPI_Bcast(setflag[0], ntype_p1 * ntype_p1, MPI_INT, 0, world);
  MPI_Bcast(&cutmax, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cutshortsq, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(e0, ntype_p1, MPI_DOUBLE, 0, world);
  MPI_Bcast(param, n_rho, MPI_DOUBLE, 0, world);
  


  char temp[100];
  if (n_2body > 0) {
    MPI_Bcast(lo_2body, n_2body, MPI_DOUBLE, 0, world);
    MPI_Bcast(hi_2body, n_2body, MPI_DOUBLE, 0, world);
    MPI_Bcast(grid_2body, n_2body, MPI_INT, 0, world);
    MPI_Bcast(cut2bsq, n_2body, MPI_DOUBLE, 0, world);
    MPI_Bcast(map2b[0], ntype_p1 * ntype_p1, MPI_INT, 0, world);
    for (int idx = 0; idx < n_2body; idx++) {
      if (me > 0) {
        sprintf(temp, "pair:fcoeff_2body%d", idx);
        memory->create(fcoeff_2body[idx], grid_2body[idx] + 2, temp);
        // sprintf(temp, "pair:ecoeff_2body%d", idx);
        // memory->create(fcoeff_2body[idx],grid_2body[idx]+2,temp);
      }
      MPI_Bcast(fcoeff_2body[idx], grid_2body[idx] + 2, MPI_DOUBLE, 0, world);
      // MPI_Bcast(ecoeff_2body[idx],order+2,MPI_DOUBLE,0,world);
    }
  }

  if (n_3body > 0) {
    MPI_Bcast(lo_3body[0], n_3body * 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(hi_3body[0], n_3body * 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(grid_3body[0], n_3body * 3, MPI_INT, 0, world);
    MPI_Bcast(cut3bsq, n_3body, MPI_DOUBLE, 0, world);
    MPI_Bcast(map3b[0][0], ntype_p1 * ntype_p1 * ntype_p1, MPI_INT, 0, world);
    for (int idx = 0; idx < n_3body; idx++) {
      int *order = grid_3body[idx];
      int len = (order[0] + 2) * (order[1] + 2) * (order[2] + 2);
      if (me > 0) {
        sprintf(temp, "pair:fcoeff_3body%d", idx);
        memory->create(fcoeff_3body[idx], len, temp);
        // sprintf(temp, "pair:ecoeff_3body%d", idx);
        // memory->create(fcoeff_3body[idx], len, temp);
      }
      MPI_Bcast(fcoeff_3body[idx], len, MPI_DOUBLE, 0, world);
      // MPI_Bcast(ecoeff_3body[idx],len,MPI_DOUBLE,0,world);
    }
  }
  if (n_rho > 0) {
    MPI_Bcast(lo_rho, n_rho/6, MPI_DOUBLE, 0, world);
    MPI_Bcast(hi_rho, n_rho/6, MPI_DOUBLE, 0, world);
    MPI_Bcast(grid_rho, n_rho/6, MPI_INT, 0, world);
    MPI_Bcast(cutrhosq, n_rho/6, MPI_DOUBLE, 0, world);
    MPI_Bcast(map_rho[0], ntype_p1 * ntype_p1, MPI_INT, 0, world);
    for (int idx = 0; idx < n_rho; idx++) {
      if (me > 0) {
        sprintf(temp, "pair:fcoeff_rho%d", idx);		
          memory->create(fcoeff_rho[idx], grid_rho[idx/6] + 2, temp);
      }
        MPI_Bcast(fcoeff_rho[idx], grid_rho[idx/6] + 2, MPI_DOUBLE, 0, world);
    }
  }
}

/* -------------------------------------------------------------------
   init specific to this pair style
-------------------------------------------------------------------- */

void PairTabMLP::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style tabMLP requires newton pair on");

  // need a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ------------------------------------------------------------------
 *     init for one type pair i,j and corresponding j,i
 * ---------------------------------------------------------------- */

double PairTabMLP::init_one(int i, int j) {
  if (setflag[i][j] == 0)
    error->all(FLERR, "All pair coeffs are not set");
  return cutmax;
  // return sqrt(cutsq[i][j]);
}


/* ---------------------------------------------------------------------- */



void PairTabMLP::eval_cubic_splines_1d(double a, double b, int orders,
                                    double *coefs, double r, double *val,
                                    double *dval) {
  // some coefficients
  int i, j, ii;
  double dinv, u, i0, tt;
  *val = 0;
  dinv = (orders - 1.0) / (b - a);
  u = (r - a) * dinv;
  i0 = floor(u);
  ii = fmax(fmin(i0, orders - 2), 0);
  tt = u - ii;

  // interpolation points
  double tp[4];
  tp[3] = 1.0;
  tp[2] = tt;
  tp[1] = tt*tt;
  tp[0] = tp[2]*tp[1];

  // value of cubic spline function
  double Phi[4], dPhi[4];
  double dt;
  int k;

  if (tt < 0) {
    for (i = 0; i < 4; i++) {
      Phi[i] = dAd[i][3] * tt + Ad[i][3];
    }
  } else if (tt > 1) {
    dt = tt - 1;
    for (i = 0; i < 4; i++) {
      Phi[i] = Bd[i] * dt + Cd[i];
    }
  } else {
    for (i = 0; i < 4; i++) {
      Phi[i] = 0;
      for (k = 0; k < 4; k++) {
        Phi[i] += Ad[i][k] * tp[k];
      }
    }
  }

  // value of derivative of spline
  for (i = 0; i < 4; i++) {
    dPhi[i] = 0;
    for (k = 0; k < 4; k++) {
      dPhi[i] += dAd[i][k] * tp[k];
    }
    dPhi[i] *= dinv;
  }

  // added by coefficients
  // int N = orders + 2;
  // double pc = 0;
  // double ppc = 0;

  for (j = 0; j < 4; j++) {
    *val += Phi[j] * coefs[ii + j];
    *dval += dPhi[j] * coefs[ii + j];
  }
}

void PairTabMLP::eval_cubic_splines_1d_energy(double a, double b, int orders,
                                    double *coefs, double r, double *val) {
  // some coefficients
  int i, j, ii;
  double dinv, u, i0, tt;
  *val = 0;
  dinv = (orders - 1.0) / (b - a);
  u = (r - a) * dinv;
  i0 = floor(u);
  ii = fmax(fmin(i0, orders - 2), 0);
  tt = u - ii;

  // interpolation points
  double tp[4];
  tp[3] = 1.0;
  tp[2] = tt;
  tp[1] = tt*tt;
  tp[0] = tp[2]*tp[1];

  // value of cubic spline function
  double Phi[4];
  double dt;
  int k;

  if (tt < 0) {
    for (i = 0; i < 4; i++) {
      Phi[i] = dAd[i][3] * tt + Ad[i][3];
    }
  } else if (tt > 1) {
    dt = tt - 1;
    for (i = 0; i < 4; i++) {
      Phi[i] = Bd[i] * dt + Cd[i];
    }
  } else {
    for (i = 0; i < 4; i++) {
      Phi[i] = 0;
      for (k = 0; k < 4; k++) {
        Phi[i] += Ad[i][k] * tp[k];
      }
    }
  }

  for (j = 0; j < 4; j++) {
    *val += Phi[j] * coefs[ii + j];
  }
}

void PairTabMLP::eval_cubic_splines_1d_force(double a, double b, int orders,
                                    double *coefs, double r,double *dval) {
  // some coefficients
  int i, j, ii;
  double dinv, u, i0, tt;
  dinv = (orders - 1.0) / (b - a);
  u = (r - a) * dinv;
  i0 = floor(u);
  ii = fmax(fmin(i0, orders - 2), 0);
  tt = u - ii;

  // interpolation points
  double tp[4];
  tp[3] = 1.0;
  tp[2] = tt;
  tp[1] = tt*tt;
  tp[0] = tp[2]*tp[1];

  // value of cubic spline function
  double dPhi[4];
  int k;
  // value of derivative of spline
  for (i = 0; i < 4; i++) {
    dPhi[i] = 0;
    for (k = 0; k < 4; k++) {
      dPhi[i] += dAd[i][k] * tp[k];
    }
    dPhi[i] *= dinv;
  }

  for (j = 0; j < 4; j++) {
    *dval += dPhi[j] * coefs[ii + j];
  }
}

void PairTabMLP::eval_cubic_splines_3d(double *a, double *b, int *orders,
                                    double *coefs, double r1, double r2,
                                    double a12, double *val, double *dval) {
  int dim = 3;
  int i;
  int j;
  double point[3] = {r1, r2, a12};
  double dinv[dim];
  double u[dim];
  double i0[dim];
  int ii[dim];
  double tt[dim];

  *val = 0;
  // coefficients
  for (i = 0; i < dim; i++) {
    dinv[i] = (orders[i] - 1.0) / (b[i] - a[i]);
    u[i] = (point[i] - a[i]) * dinv[i];
    i0[i] = floor(u[i]);
    ii[i] = fmax(fmin(i0[i], orders[i] - 2), 0);
    tt[i] = u[i] - ii[i];
  }

  // points
  double tp[dim][4];
  for (i = 0; i < dim; i++) {
    tp[i][3] = 1.0;
    tp[i][2] = tt[i];
    tp[i][1] = tt[i]*tt[i];
    tp[i][0] = tp[i][2]*tp[i][1];
  }

  double Phi[dim][4], dPhi[dim][4];
  double dt;
  int k;

  for (j = 0; j < dim; j++) {

    // evaluate spline function
    if (tt[j] < 0) {
      for (i = 0; i < 4; i++) {
        Phi[j][i] = dAd[i][3] * tt[j] + Ad[i][3];
      }
    } else if (tt[j] > 1) {
      dt = tt[j] - 1;
      for (i = 0; i < 4; i++) {
        Phi[j][i] = Bd[i] * dt + Cd[i];
      }
    } else {
      for (i = 0; i < 4; i++) {
        Phi[j][i] = 0;
        for (k = 0; k < 4; k++) {
          Phi[j][i] += Ad[i][k] * tp[j][k];
        }
      }
    }

    // evaluate derivatives
    for (i = 0; i < 4; i++) {
      dPhi[j][i] = 0;
      for (k = 0; k < 4; k++) {
        dPhi[j][i] += dAd[i][k] * tp[j][k];
      }
      dPhi[j][i] *= dinv[j];
    }
  }

  // added by coefficients
  int N[dim];
  for (i = 0; i < dim; i++) {
    N[i] = orders[i] + 2;
  }

  for (i = 0; i < 4; i++) {
    double ppc = 0;
    double dppc1 = 0;
    double dppc2 = 0;
    for (j = 0; j < 4; j++) {
      double pc = 0;
      double dpc = 0;
      for (k = 0; k < 4; k++) {
        double c = coefs[((ii[0] + i) * N[1] + ii[1] + j) * N[2] + ii[2] + k];
        pc += Phi[2][k] * c;
        dpc += dPhi[2][k] * c;
      }
      ppc += Phi[1][j] * pc;
      dppc1 += dPhi[1][j] * pc;
      dppc2 += Phi[1][j] * dpc;
    }
    *val += Phi[0][i] * ppc;
    dval[0] += dPhi[0][i] * ppc;
    dval[1] += Phi[0][i] * dppc1;
    dval[2] += Phi[0][i] * dppc2;
  }
}

int PairTabMLP::pack_forward_comm(int n, int *list, double *buf,
                               int pbc_flag, int * pbc)
{
  int i,j,m,k;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (k = 0; k < 24; k++) 
	  buf[m++] = rho_value[j][k];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairTabMLP::unpack_forward_comm(int n, int first, double *buf)
{
  int i,k,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) 
  {
	for (k = 0; k < 24; k++) 
      rho_value[i][k] = buf[m++];   
  }
}


/* ---------------------------------------------------------------------- */

int PairTabMLP::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) 
	  for (k = 0; k < 24; k++) 
	  buf[m++] = rho_value[i][k];
  return m;
}

void PairTabMLP::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (k = 0; k < 24; k++) 
      rho_value[j][k] = buf[m++];
  }
}
double PairTabMLP::memory_usage()
{
  double bytes = (double)maxeatom * sizeof(double);
  bytes += (double)maxvatom*6 * sizeof(double);
  bytes += (double)2 * nmax * sizeof(double);
  return bytes;
}
