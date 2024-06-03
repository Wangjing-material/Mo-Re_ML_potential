/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Hongxiang Zong (XJTU& UoE), zonghust@mail.xjtu.edu.cn
   The University of Edinburgh
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_ml_energy.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include <iostream>
#include <vector>


using namespace std;
using namespace LAMMPS_NS;
#define MAXLINE 1024
/* ---------------------------------------------------------------------- */

PairMLEnergy::PairMLEnergy(LAMMPS *lmp) : Pair(lmp)
{
  nfeature = 1;
  ntarget = 1;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 0;

  nelements = 0;
  elements = NULL;
  
  eta_list = NULL;
  eta2_list = NULL;  
  rho_values = NULL;
  twoBodyInfo = NULL;
  
  fvect_unit_Eg = NULL;
  norm_dev = NULL;
  norm_mean = NULL;
  fvect_dev = NULL;

  nmax = 0;
  eta_num = 0;
  maxNeighbors = 0;
  zero_atom_energy = 0.0;

  comm_forward = 32;
  comm_reverse = 0;//二体项计算不用再算j到i，节约时间
}

/* ---------------------------------------------------------------------- */

PairMLEnergy::~PairMLEnergy()
{
  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;

  memory->destroy(Fvect_mpi_arr);
  memory->destroy(Target_mpi_arr);
  
  delete[] twoBodyInfo;
  memory->destroy(rho_values);  
  
  if(allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }

  memory->destroy(eta_list); 
  memory->destroy(eta2_list);  
  
  memory->destroy(fvect_unit_Eg); 
  memory->destroy(norm_dev); 
  memory->destroy(norm_mean);   
  memory->destroy(fvect_dev);
}

/* ---------------------------------------------------------------------- */

void PairMLEnergy::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr =
         eflag_global = vflag_global = eflag_atom = vflag_atom = 0;

  double cutforcesq = cutoff*cutoff;

  /// Grow per-atom array if necessary
  if (atom->nmax > nmax) {
	memory->destroy(rho_values);
    nmax = atom->nmax;
    memory->create(rho_values,nmax,32,"pair:rho_values");
  }

  double** const x = atom->x;
  double** forces = atom->f;
  int *type = atom->type;  
  int nlocal = atom->nlocal;
  bool newton_pair = force->newton_pair;

  int inum_full = listfull->inum;
  int* ilist_full = listfull->ilist;
  int* numneigh_full = listfull->numneigh;
  int** firstneigh_full = listfull->firstneigh;

  int newMaxNeighbors = 0;
  for(int ii = 0; ii < inum_full; ii++) {
    int jnum = numneigh_full[ilist_full[ii]];
    if(jnum > newMaxNeighbors) newMaxNeighbors = jnum;
  }

  /// Allocate array for temporary bond info

  if(newMaxNeighbors > maxNeighbors) {
    maxNeighbors = newMaxNeighbors;
    delete[] twoBodyInfo;
    twoBodyInfo = new MEAM2Body[maxNeighbors];
  }

double mu_list[8] ={2.60,3.0,3.234,3.58,4.41,5.06,5.60,6.20};
double Rm = cutoff*0.5;
double Rm_sq = Rm*Rm;
Pi_value = acos(-1.0);


 for(int ii = 0; ii < inum_full; ii++) {
    int i = ilist_full[ii];
    int itype = type[i];
	
   for(int k=0;k<32;k++)
	   rho_values[i][k] = 0.0;	
	
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int* jlist = firstneigh_full[i];
    int jnum = numneigh_full[i];
	

    int numBonds = 0;
    double fagl = 0;
    double  evdwl = 0.0;	
    MEAM2Body* nextTwoBodyInfo = twoBodyInfo;   
    double PotEg = 0.0;

    for(int k=0;k<eta_num;k++)
	fvect_unit_Eg[k] = 0.0;
  
    	
    for(int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double jdelx = x[j][0] - xtmp;
      double jdely = x[j][1] - ytmp;
      double jdelz = x[j][2] - ztmp;
      double rij_sq = jdelx*jdelx + jdely*jdely + jdelz*jdelz;
 
      if(rij_sq < cutforcesq) {		  
	int jtype = type[j];
        double rij = sqrt(rij_sq);		

        PotEg += EAM_Epair(rij,itype,jtype);

        double fc_rij= fun_cutoff(rij,cutoff);
       	int pos2b = pos2bs_fun(itype,jtype,6);
       	int pair_Sij = pos2b_fun(itype,jtype,1);

        for(int k=0;k<6;k++){
	        double lastp = exp(-1.0*pow(rij/eta_list[k],2))*fc_rij;			 
                fvect_unit_Eg[k+pos2b] += lastp;
		rho_values[i][k] += lastp;
		rho_values[i][k+pos2b+6] += lastp;		

           fvect_unit_Eg[k+24+pos2b] += Hx(mu_list[k]-rij)*pow(mu_list[k]-rij,3)*exp(-1.0*pow((rij-2.5)/cutoff,2))*fc_rij;
		   fvect_unit_Eg[k+48+pos2b] += sin(k*rij)*exp(-rij/eta_list[k])*fc_rij;
		   fvect_unit_Eg[k+72+pos2b] += cos(k*rij)*exp(-rij/eta_list[k])*fc_rij;

               }

        //if(rij > 6.5) continue;
        nextTwoBodyInfo->tag = j;
        nextTwoBodyInfo->type = jtype;		
        nextTwoBodyInfo->r = rij;
        nextTwoBodyInfo->fcut = fun_cutoff(rij,6.5,6.0);
        nextTwoBodyInfo->fcut_dev = fun_cutoff_dev(rij,6.5,6.0);		
        nextTwoBodyInfo->del[0] = jdelx / rij;
        nextTwoBodyInfo->del[1] = jdely / rij;
        nextTwoBodyInfo->del[2] = jdelz / rij;



        for(int kk = 0; kk < numBonds; kk++) {
          const MEAM2Body& bondk = twoBodyInfo[kk];
          double cos_theta = (nextTwoBodyInfo->del[0]*bondk.del[0] +
                              nextTwoBodyInfo->del[1]*bondk.del[1] +
                              nextTwoBodyInfo->del[2]*bondk.del[2]);
        double cos_theta2=cos_theta*cos_theta;
        double fkk_cut=nextTwoBodyInfo->fcut*bondk.fcut;
	double rik = bondk.r;
	int ktype = bondk.type;
        double rsij2 = pow(rij+rik,2);
        double rvij2 = pow(rij-rik,2);

        double gcos1 =  1.0;
        double gcos2 = (3*cos_theta2-1);
        double gcos3 = cos_theta*(5*cos_theta2 -3);	

       int pos3b = pos3b_fun(itype,jtype,ktype,1)+96;		
       double prefactor=fkk_cut*exp(-1.0*rvij2/Rm_sq); 
       fvect_unit_Eg[pos3b] += prefactor*gcos1;
       fvect_unit_Eg[pos3b+6] += prefactor*gcos2;
       fvect_unit_Eg[pos3b+12] += prefactor*gcos3;

       pos3b = pos3b+18;
       prefactor=fkk_cut*exp(-1.0*rsij2/cutoff2);
       fvect_unit_Eg[pos3b] += prefactor*gcos1;
       fvect_unit_Eg[pos3b+6] += prefactor*gcos2;
       fvect_unit_Eg[pos3b+12] += prefactor*gcos3;
	   
	   pos3b = pos3b+18;
       prefactor=fkk_cut*exp(-1.0*rvij2/(0.75*Rm_sq));
       fvect_unit_Eg[pos3b] += prefactor*gcos1;
       fvect_unit_Eg[pos3b+6] += prefactor*gcos2;
       fvect_unit_Eg[pos3b+12] += prefactor*gcos3;
	   
	   pos3b = pos3b+18;
       prefactor=fkk_cut*exp(-1.0*rvij2/(0.5*Rm_sq));
       fvect_unit_Eg[pos3b] += prefactor*gcos1;
       fvect_unit_Eg[pos3b+6] += prefactor*gcos2;
       fvect_unit_Eg[pos3b+12] += prefactor*gcos3;
	   
	   pos3b = pos3b+18;
       prefactor=fkk_cut*exp(-1.0*rvij2/(0.25*Rm_sq));
       fvect_unit_Eg[pos3b] += prefactor*gcos1;
       fvect_unit_Eg[pos3b+6] += prefactor*gcos2;
       fvect_unit_Eg[pos3b+12] += prefactor*gcos3;
	   
	   pos3b = pos3b+18;
       prefactor=fkk_cut*exp(-1.0*rvij2/(0.1*Rm_sq));
       fvect_unit_Eg[pos3b] += prefactor*gcos1;
       fvect_unit_Eg[pos3b+6] += prefactor*gcos2;
       fvect_unit_Eg[pos3b+12] += prefactor*gcos3;

        }		  
		  
		  
	 numBonds++;
         nextTwoBodyInfo++;

    }

//    printf("Run to here!\n");
 }


//printf("nbond =%d\n",numBonds); 

for(int k=0;k<6;k++)
 {	

     double rho_sum = rho_values[i][k+6]+ 1.0;
     fvect_unit_Eg[k+204]= rho_sum*(log(rho_sum)-1.0)+1.0;

     double rho_sum1 = rho_values[i][k+24]+ 1.0;
     fvect_unit_Eg[k+210]= rho_sum1*(log(rho_sum1)-1.0)+1.0;

     double rho_sum2 =rho_values[i][k+6]+rho_values[i][k+12]+ 1.0;
     fvect_unit_Eg[k+216]= rho_sum2*(log(rho_sum2)-1.0)+1.0;


     double rho_sum3 = rho_values[i][k+18]+rho_values[i][k+24]+ 1.0;
     fvect_unit_Eg[k+222]= rho_sum3*(log(rho_sum3)-1.0)+1.0;

     rho_values[i][k+6] = fvect_dev[k+204]*log(rho_sum)+fvect_dev[k+216]*log(rho_sum2);
	 rho_values[i][k+24] = fvect_dev[k+210]*log(rho_sum1)+fvect_dev[k+222]*log(rho_sum3);
     rho_values[i][k+12] = fvect_dev[k+216]*log(rho_sum2);
     rho_values[i][k+18] = fvect_dev[k+222]*log(rho_sum3);
		  }
  

//debugging

/*
if(i<4)
{
//	printf("%lg %lg %lg\n",x[i][0],x[i][1],x[i][2]);	

  //     printf("nbond =%d\n",numBonds); 	

     for(int t=0;t<eta_num;t++)
       printf("%lg ",fvect_unit_Eg[t]);
      printf("\n");
}
*/	

    //comm->forward_comm_pair(this);
	
 PotEg = PotEg*0.5; 

  for(int tj = 0; tj<eta_num;tj++)
      PotEg += fvect_dev[tj]*fvect_unit_Eg[tj];

  PotEg +=zero_atom_energy;

 //printf("PotEg = %lg\n",PotEg);
	 
	if (eflag) {
       if (eflag_global) eng_vdwl += PotEg;
       if (eflag_atom) eatom[i] += PotEg;
        }
 
 
 //calculating forces for pre-atoms

 
      for(int jj = 0; jj < numBonds; jj++) {
        const MEAM2Body bondj = twoBodyInfo[jj];
        double rij = bondj.r;
        int j = bondj.tag;
        int jtype = bondj.type;	  
      
      MEAM2Body const* bondk = twoBodyInfo;
      for(int kk = 0; kk < jj; kk++, ++bondk) {
	double rik = bondk->r;
        int k = bondk->tag;
	int ktype = bondk->type;

        double cos_theta = (bondj.del[0]*bondk->del[0] +
                            bondj.del[1]*bondk->del[1] +
                            bondj.del[2]*bondk->del[2]);

        double cos_theta2=cos_theta*cos_theta;
        double fjk_cut=bondj.fcut*bondk->fcut;
	double frjp_cut=bondj.fcut_dev*bondk->fcut;
	double frkp_cut=bondj.fcut*bondk->fcut_dev;
        double dcosdrj[3],dcosdrk[3];
        costheta_d(cos_theta,bondj.del,rij,bondk->del,rik,dcosdrj,dcosdrk);

         double gcos1 = 1.0;
         double gcos1_dev = 0.0;

         double gcos2 = 3*cos_theta2-1;
         double gcos2_dev = 6*cos_theta;

         double gcos3 = cos_theta*(5*cos_theta2 -3);
         double gcos3_dev = 15.0*cos_theta2 -3 ;

	double fj[3]= {0, 0, 0};
	double fk[3]= {0, 0, 0};		
        double fcos_factor_sum = 0.0;
        double fij_sum = 0.0;
        double fik_sum = 0.0;

        double rsij2 = pow(rij+rik,2);
        double rvij2 = pow(rij-rik,2);
        int pos3b = pos3b_fun(itype,jtype,ktype,1)+96;			
		
	//double fjk_eta2 = fjk_cut/Rm_sq;
        double gausp= exp(-1.0*rvij2/Rm_sq);
        double prefactor_gij = frjp_cut-(2.0*(rij-rik)/Rm_sq)*fjk_cut;
        double prefactor_gik = frkp_cut-(2.0*(rik-rij)/Rm_sq)*fjk_cut;

	 double gcos_sum = 0.0;
	 double gcos_dev_sum = 0.0;			 

         double gs1_factor = -1.0*fvect_dev[pos3b];			
         gcos_sum += gs1_factor*gcos1;
	 gcos_dev_sum += gs1_factor*gcos1_dev;

         double gs2_factor = -1.0*fvect_dev[pos3b+6];		
         gcos_sum += gs2_factor*gcos2;
	 gcos_dev_sum += gs2_factor*gcos2_dev;

         double gs3_factor = -1.0*fvect_dev[pos3b+12];		
         gcos_sum += gs3_factor*gcos3;
	 gcos_dev_sum += gs3_factor*gcos3_dev;	

        double fr_g_factor = gcos_sum*gausp;
	double fcos_g_factor = gcos_dev_sum*gausp;			

        fcos_factor_sum += fcos_g_factor*fjk_cut;
	    fij_sum += prefactor_gij*fr_g_factor;
        fik_sum += prefactor_gik*fr_g_factor;
		

        pos3b = pos3b+18;			
        gausp= exp(-1.0*rsij2/cutoff2);
        prefactor_gij = frjp_cut-(2.0*(rij+rik)/cutoff2)*fjk_cut;
        prefactor_gik = frkp_cut-(2.0*(rik+rij)/cutoff2)*fjk_cut;

        gcos_sum = 0.0;
        gcos_dev_sum = 0.0;

         gs1_factor = -1.0*fvect_dev[pos3b];
         gcos_sum += gs1_factor*gcos1;
         gcos_dev_sum += gs1_factor*gcos1_dev;
         
         gs2_factor = -1.0*fvect_dev[pos3b+6];
         gcos_sum += gs2_factor*gcos2;
         gcos_dev_sum += gs2_factor*gcos2_dev;
         
         gs3_factor = -1.0*fvect_dev[pos3b+12];
         gcos_sum += gs3_factor*gcos3;
         gcos_dev_sum += gs3_factor*gcos3_dev;

        fr_g_factor = gcos_sum*gausp;
        fcos_g_factor = gcos_dev_sum*gausp;

        fcos_factor_sum += fcos_g_factor*fjk_cut;
        fij_sum += prefactor_gij*fr_g_factor;
        fik_sum += prefactor_gik*fr_g_factor;
		
		pos3b = pos3b+18;			
        gausp= exp(-1.0*rvij2/(0.75*Rm_sq));
        prefactor_gij = frjp_cut-(2.0*(rij-rik)/(0.75*Rm_sq))*fjk_cut;
        prefactor_gik = frkp_cut-(2.0*(rik-rij)/(0.75*Rm_sq))*fjk_cut;

        gcos_sum = 0.0;
	   gcos_dev_sum = 0.0;			 

         gs1_factor = -1.0*fvect_dev[pos3b];			
         gcos_sum += gs1_factor*gcos1;
	 gcos_dev_sum += gs1_factor*gcos1_dev;

         gs2_factor = -1.0*fvect_dev[pos3b+6];		
         gcos_sum += gs2_factor*gcos2;
	 gcos_dev_sum += gs2_factor*gcos2_dev;

         gs3_factor = -1.0*fvect_dev[pos3b+12];		
         gcos_sum += gs3_factor*gcos3;
	 gcos_dev_sum += gs3_factor*gcos3_dev;	

         fr_g_factor = gcos_sum*gausp;
	     fcos_g_factor = gcos_dev_sum*gausp;			

        fcos_factor_sum += fcos_g_factor*fjk_cut;
	    fij_sum += prefactor_gij*fr_g_factor;
        fik_sum += prefactor_gik*fr_g_factor;
		
		pos3b = pos3b+18;			
        gausp= exp(-1.0*rvij2/(0.5*Rm_sq));
        prefactor_gij = frjp_cut-(2.0*(rij-rik)/(0.5*Rm_sq))*fjk_cut;
        prefactor_gik = frkp_cut-(2.0*(rik-rij)/(0.5*Rm_sq))*fjk_cut;

        gcos_sum = 0.0;
	   gcos_dev_sum = 0.0;			 

         gs1_factor = -1.0*fvect_dev[pos3b];			
         gcos_sum += gs1_factor*gcos1;
	 gcos_dev_sum += gs1_factor*gcos1_dev;

         gs2_factor = -1.0*fvect_dev[pos3b+6];		
         gcos_sum += gs2_factor*gcos2;
	 gcos_dev_sum += gs2_factor*gcos2_dev;

         gs3_factor = -1.0*fvect_dev[pos3b+12];		
         gcos_sum += gs3_factor*gcos3;
	 gcos_dev_sum += gs3_factor*gcos3_dev;	

         fr_g_factor = gcos_sum*gausp;
	     fcos_g_factor = gcos_dev_sum*gausp;			

        fcos_factor_sum += fcos_g_factor*fjk_cut;
	    fij_sum += prefactor_gij*fr_g_factor;
        fik_sum += prefactor_gik*fr_g_factor;
		
		pos3b = pos3b+18;			
        gausp= exp(-1.0*rvij2/(0.25*Rm_sq));
        prefactor_gij = frjp_cut-(2.0*(rij-rik)/(0.25*Rm_sq))*fjk_cut;
        prefactor_gik = frkp_cut-(2.0*(rik-rij)/(0.25*Rm_sq))*fjk_cut;

        gcos_sum = 0.0;
	    gcos_dev_sum = 0.0;			 

         gs1_factor = -1.0*fvect_dev[pos3b];			
         gcos_sum += gs1_factor*gcos1;
	 gcos_dev_sum += gs1_factor*gcos1_dev;

         gs2_factor = -1.0*fvect_dev[pos3b+6];		
         gcos_sum += gs2_factor*gcos2;
	 gcos_dev_sum += gs2_factor*gcos2_dev;

         gs3_factor = -1.0*fvect_dev[pos3b+12];		
         gcos_sum += gs3_factor*gcos3;
	 gcos_dev_sum += gs3_factor*gcos3_dev;	

        fr_g_factor = gcos_sum*gausp;
	    fcos_g_factor = gcos_dev_sum*gausp;			

        fcos_factor_sum += fcos_g_factor*fjk_cut;
	    fij_sum += prefactor_gij*fr_g_factor;
        fik_sum += prefactor_gik*fr_g_factor;
		
		pos3b = pos3b+18;			
        gausp= exp(-1.0*rvij2/(0.1*Rm_sq));
        prefactor_gij = frjp_cut-(2.0*(rij-rik)/(0.1*Rm_sq))*fjk_cut;
        prefactor_gik = frkp_cut-(2.0*(rik-rij)/(0.1*Rm_sq))*fjk_cut;

        gcos_sum = 0.0;
	   gcos_dev_sum = 0.0;			 

         gs1_factor = -1.0*fvect_dev[pos3b];			
         gcos_sum += gs1_factor*gcos1;
	 gcos_dev_sum += gs1_factor*gcos1_dev;

         gs2_factor = -1.0*fvect_dev[pos3b+6];		
         gcos_sum += gs2_factor*gcos2;
	 gcos_dev_sum += gs2_factor*gcos2_dev;

         gs3_factor = -1.0*fvect_dev[pos3b+12];		
         gcos_sum += gs3_factor*gcos3;
	 gcos_dev_sum += gs3_factor*gcos3_dev;	

         fr_g_factor = gcos_sum*gausp;
	     fcos_g_factor = gcos_dev_sum*gausp;			

        fcos_factor_sum += fcos_g_factor*fjk_cut;
	    fij_sum += prefactor_gij*fr_g_factor;
        fik_sum += prefactor_gik*fr_g_factor;
	
	fj[0]+=fij_sum*bondj.del[0]+fcos_factor_sum*dcosdrj[0];
	fj[1]+=fij_sum*bondj.del[1]+fcos_factor_sum*dcosdrj[1];
	fj[2]+=fij_sum*bondj.del[2]+fcos_factor_sum*dcosdrj[2];
			
	fk[0]+=fik_sum*bondk->del[0]+fcos_factor_sum*dcosdrk[0];
	fk[1]+=fik_sum*bondk->del[1]+fcos_factor_sum*dcosdrk[1];
	fk[2]+=fik_sum*bondk->del[2]+fcos_factor_sum*dcosdrk[2]; 	
	
        forces[i][0] -= fj[0];
        forces[i][1] -= fj[1];
        forces[i][2] -= fj[2];			
        forces[j][0] += fj[0];
        forces[j][1] += fj[1];
        forces[j][2] += fj[2];			
			
			
        forces[i][0] -= fk[0];
        forces[i][1] -= fk[1];
        forces[i][2] -= fk[2];			
        forces[k][0] += fk[0];
        forces[k][1] += fk[1];
        forces[k][2] += fk[2];			

     if(evflag) {
          double delta_ij[3];
          double delta_ik[3];
          delta_ij[0] = bondj.del[0] * rij;
          delta_ij[1] = bondj.del[1] * rij;
          delta_ij[2] = bondj.del[2] * rij;
          delta_ik[0] = bondk->del[0] * rik;
          delta_ik[1] = bondk->del[1] * rik;
          delta_ik[2] = bondk->del[2] * rik;
          ev_tally3(i, j, k, 0.0, 0.0, fj, fk, delta_ij, delta_ik);
           }			
        }
     }	
 }
 // Communicate U'(rho) values

  comm->forward_comm_pair(this);
  
  
  for(int ii = 0; ii < inum_full; ii++) {
    int i = ilist_full[ii];
    int itype = type[i];	
	
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int* jlist = firstneigh_full[i];
    int jnum = numneigh_full[i]; 

		   
    for(int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double jdelx = x[j][0] - xtmp;
      double jdely = x[j][1] - ytmp;
      double jdelz = x[j][2] - ztmp;
      double rij_sq = jdelx*jdelx + jdely*jdely + jdelz*jdelz;
 
      if(rij_sq < cutforcesq) {
        double rij = sqrt(rij_sq);
	int jtype = type[j];
		
	     double fpair = 0.0; 	
        double fc_rij = fun_cutoff(rij,cutoff);
        double fcut_coeff = -1.0*sin(rij*Pi_value/cutoff)*Pi_value/(2*rij*cutoff);//fc_rij求导/rij
	    int pos2b = pos2bs_fun(itype,jtype,6);
        int pos_pair_type = pos2bs_fun(itype,jtype,1);		

        fpair +=EAM_fpair(rij,itype,jtype);

        for(int k=0;k<6;k++)
	  {

	   double fpair1_coeff = -2.0*fc_rij/eta2_list[k];
	   double pair_pot1_deriv = (fpair1_coeff+fcut_coeff)*exp(-1.0*pow(rij/eta_list[k],2));	//lasp求导		
           fpair += fvect_dev[k+pos2b]*pair_pot1_deriv;

            double ex = pow((rij-2.5)/cutoff,2);
            double gx = pow(mu_list[k]-rij,3);
            double gxp = -3.0*pow(mu_list[k]-rij,2);
            double fpair2_coeff =(gxp-2*gx*(rij-2.5)/cutoff2)*fc_rij/rij;
            double fcut2_coeff = fcut_coeff*gx;
            double pair_pot2_deriv = (fpair2_coeff+fcut2_coeff)*exp(-1.0*ex)*Hx(mu_list[k]-rij);

            fpair += fvect_dev[k+24+pos2b]*pair_pot2_deriv;
			
			double fpair3_coeff =(k*cos(k*rij)-sin(k*rij)/eta_list[k])*fc_rij/rij;
			double fcut3_coeff=sin(k*rij)*fcut_coeff;
			double pair_pot3_deriv=(fpair3_coeff+fcut3_coeff)*exp(-rij/eta_list[k]);
			fpair += fvect_dev[k+48+pos2b]*pair_pot3_deriv;
			
			double fpair4_coeff =(-k*sin(k*rij)-cos(k*rij)/eta_list[k])*fc_rij/rij;
			double fcut4_coeff=cos(k*rij)*fcut_coeff;
			double pair_pot4_deriv=(fpair4_coeff+fcut4_coeff)*exp(-rij/eta_list[k]);
			fpair += fvect_dev[k+72+pos2b]*pair_pot4_deriv;
			
			
		  //计算多体项力
		  double Urho_prime_ij=0.0;		
          
		  if(pos_pair_type==0)//mo-mo
			   Urho_prime_ij= 0.5*(rho_values[i][k+6]+rho_values[j][k+6]);
		else if(pos_pair_type==3)
			Urho_prime_ij= 0.5*(rho_values[i][k+24]+rho_values[j][k+24]);
		else if(pos_pair_type==1)
			Urho_prime_ij= rho_values[i][k+12];
		else if(pos_pair_type==2)
			Urho_prime_ij= rho_values[i][k+18];
       
           fpair +=Urho_prime_ij*pair_pot1_deriv;

               }			 
			 
          forces[i][0] += jdelx*fpair;
          forces[i][1] += jdely*fpair;
          forces[i][2] += jdelz*fpair;

          forces[j][0] -= jdelx*fpair;
          forces[j][1] -= jdely*fpair;
          forces[j][2] -= jdelz*fpair;	
		  
        //if (evflag) ev_tally_full(i,0.0, 0.0, -fpair, jdelx, jdely, jdelz);
          if (evflag) ev_tally(i,j,nlocal,newton_pair,0.0,0.0,-fpair,jdelx, jdely, jdelz);	 
	      }

	  }
  }

 if(vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairMLEnergy::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  map = new int[n+1];

  
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMLEnergy::settings(int narg, char **arg)
{
  if(narg != 2) error->all(FLERR,"Illegal pair_style command");
  train_flag = atoi(arg[0]);
  zero_atom_energy = atof(arg[1]);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLEnergy::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }

  // read potential file
  if(train_flag==0)
   {
     read_file(arg[2]);
     Data_Fitting();
      }
   else 
     read_param(arg[2]);
 
 
   // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;

    if (j < nelements) map[i-2] = j+1;
    else error->all(FLERR,"No matching element in ML potential file");
  }
 

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLEnergy::read_file(const char* filename)
{
        if(comm->me == 0) {
                FILE *fp = force->open_potential(filename);
                if(fp == NULL) {
                        char str[1024];
                        sprintf(str,"Cannot open machine learning trainning database file %s", filename);
                        error->one(FLERR,str);
                }

                // Skip first two line of file.
                char line[MAXLINE];

                fgets(line, MAXLINE, fp);
		fgets(line, MAXLINE, fp);
				
		fgets(line, MAXLINE, fp);
        sscanf(line,"%d %d %d",&nfeature,&ntarget,&nlines);
		fclose(fp);
		}	

        MPI_Bcast(&nfeature, 1, MPI_INT, 0, world);
        MPI_Bcast(&ntarget, 1, MPI_INT, 0, world);
        MPI_Bcast(&nlines, 1, MPI_INT, 0, world);		
	    
        // double target_unit;
        Fvect_mpi_arr= (double *)malloc((nlines*nfeature+1)*sizeof(double));
	    Target_mpi_arr= (double *)malloc((nlines*ntarget+1)*sizeof(double));
			 
      if(comm->me == 0) {
         	int iat=0;
		int jat=0;	
		char line[MAXLINE];	
		char *ptr;
		
		FILE *fp = force->open_potential(filename);
        if(fp == NULL) {
            char str[1024];
            sprintf(str,"Cannot open machine learning trainning database file %s", filename);
            error->one(FLERR,str);
                }
		

        fgets(line, MAXLINE, fp);
		fgets(line, MAXLINE, fp);				
		fgets(line, MAXLINE, fp);
		
         for(int i=0;i<nlines;i++)
          {
              fgets(line, MAXLINE, fp);
		 
              ptr = strtok(line," \t\n\r\f");
	
         for(int j=0;j<nfeature;j++)			 
		  {
                    //fvect_unit(j) = atof(ptr);
		    Fvect_mpi_arr[iat++] = atof(ptr);
		    ptr = strtok(NULL," \t\n\r\f");
		  }
		  
		  Target_mpi_arr[jat++] = atof(ptr);
			
            //fvect_arr.push_back(fvect_unit);
            //target_arr.push_back(target_unit);
                } 
        fclose(fp);
        }

        // Transfer training data from master processor to all other processors.
         MPI_Bcast(Fvect_mpi_arr, nlines*nfeature+1, MPI_DOUBLE, 0, world);
	     MPI_Bcast(Target_mpi_arr, nlines*ntarget+1, MPI_DOUBLE, 0, world);

	
        // Calculate 'zero-point energy' of single atom in vacuum.
        //zero_atom_energy = 0.0;

        // Determine maximum cutoff radius of all relevant spline functions.
        cutoff = 6.5;
        cutoff2 = cutoff*cutoff;        
 
	eta_num = nfeature;
    memory->create(eta_list,eta_num+1,"pair_ml:eta_list");	
    memory->create(eta2_list,eta_num+1,"pair_ml:eta_list");		
    for(int k=0;k<eta_num;k++)
        {  
	      eta_list[k]=1.0*pow(1.3459,1.0*k);
	      eta2_list[k]= eta_list[k]*eta_list[k];
        } 
		
		
	   
        // Set LAMMPS pair interaction flags.
        for(int i = 1; i <= atom->ntypes; i++) {
                for(int j = 1; j <= atom->ntypes; j++) {
                        setflag[i][j] = 1;
                        cutsq[i][j] = cutoff;
                }
        }
}


void PairMLEnergy::read_param(const char* filename)
{

  FILE *fp;
  char line[MAXLINE];
  
        if(comm->me == 0) {
                fp = force->open_potential(filename);
                if(fp == NULL) {
                        char str[1024];
                        sprintf(str,"Cannot open machine learning parameters file %s", filename);
                        error->one(FLERR,str);
                }

		}
		
  // read and broadcast header
  // extract element names from nelements line

  int n;
  if (comm->me == 0) {		
        fgets(line, MAXLINE, fp);
	fgets(line, MAXLINE, fp);				
        n = strlen(line) + 1;
	}	
  MPI_Bcast(&n,1,MPI_INT,0,world);
  MPI_Bcast(line,n,MPI_CHAR,0,world);	
		
  sscanf(line,"%d",&nelements);
  int nwords = atom->count_words(line);
  //printf("%d %d\n",nelements,nwords);
 
  if (nwords != nelements + 1)
    error->all(FLERR,"Incorrect element names in ML potential file");

  char **words = new char*[nelements+1];
  nwords = 0;
  strtok(line," \t\n\r\f");
  while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

  elements = new char*[nelements];
  for (int i = 0; i < nelements; i++) {
    n = strlen(words[i]) + 1;
    elements[i] = new char[n];
    strcpy(elements[i],words[i]);
  }
  delete [] words;		
		
if(comm->me == 0) {
		fgets(line, MAXLINE, fp);
        sscanf(line,"%d %d",&nfeature,&ntarget);
        printf("Reading Param_files: %d %d\n",nfeature,ntarget);		
  }
		
        MPI_Bcast(&nfeature, 1, MPI_INT, 0, world);
        MPI_Bcast(&ntarget, 1, MPI_INT, 0, world);		

 
        Fvect_mpi_arr= (double *)malloc((nfeature+1)*sizeof(double));
	    Target_mpi_arr= (double *)malloc((ntarget+1)*sizeof(double));
			 
      if(comm->me == 0) {
        int iat=0;
		int jat=0;	
		char line[MAXLINE];	
		char *ptr;
		
         for(int i=0;i<nfeature;i++)
          {
              fgets(line, MAXLINE, fp);		 
              sscanf(line,"%lf",&Fvect_mpi_arr[i]);	
		  }
		  
              fgets(line, MAXLINE, fp);
              sscanf(line,"%lf",&Target_mpi_arr[0]);
              
        fclose(fp);
        }
   
        // Transfer training data from master processor to all other processors.
         MPI_Bcast(Fvect_mpi_arr, nfeature+1, MPI_DOUBLE, 0, world);
	 MPI_Bcast(Target_mpi_arr, ntarget+1, MPI_DOUBLE, 0, world);

       memory->create(fvect_dev,nfeature+121,"pair_ml:fvect_dev");
       memory->create(fvect_unit_Eg,nfeature+121,"pair_ml:fvect_unit_Eg");

       for(int j=0;j<nfeature+121;j++)
              fvect_dev[j] = 0.0;

       for(int j=0;j<nfeature;j++)
	       fvect_dev[j] = Fvect_mpi_arr[j];
       zero_atom_energy = -1.0*Target_mpi_arr[0];
       printf("Isolated atom energy = %lg eV\n",zero_atom_energy);

        // Determine maximum cutoff radius of all relevant spline functions.
        cutoff = 6.5;
        cutoff2 = cutoff*cutoff;        
 
       eta_num = nfeature;
       memory->create(eta_list,eta_num+1,"pair_ml:eta_list");
       memory->create(eta2_list,eta_num+1,"pair_ml:eta_list");	   

       for(int k=0;k<eta_num;k++)
	   {
	        eta_list[k]=1.0*pow(1.3459,1.0*k);
		eta2_list[k]= eta_list[k]*eta_list[k];
	   }
	   
        // Set LAMMPS pair interaction flags.
        for(int i = 1; i <= atom->ntypes; i++) {
                for(int j = 1; j <= atom->ntypes; j++) {
                        setflag[i][j] = 1;
                        cutsq[i][j] = cutoff;
                }
        }

   //cout<<fvect_dev<<endl;
}

void PairMLEnergy::grab(FILE *fptr, int n, double *list)
{
  char *ptr;
  char line[MAXLINE];

  int i = 0;
  while (i < n) {
    fgets(line,MAXLINE,fptr);
    ptr = strtok(line," \t\n\r\f");
    list[i++] = atof(ptr);
    while (ptr = strtok(NULL," \t\n\r\f")) list[i++] = atof(ptr);
  }
}


void PairMLEnergy::Data_Fitting()
{
  /* randomize_samples(fvect_arr, target_arr);	
   normalizer.train(fvect_arr);
 
   for (unsigned long i = 0; i < fvect_arr.size(); ++i)
       fvect_arr[i] = normalizer(fvect_arr[i]);
	
  if(comm->me == 0)
        cout << "doing a grid cross-validation" << endl;

    matrix<double> params = logspace(log10(1e-6),log10(1e0),7);
     
    matrix<double> best_result(2,1);
    best_result = 0;
    double best_lambda = 0.000001;
    for(long col =0 ;col <params.nc(); ++col)
    {
        // tell the trainer the parameters we want to use
          const double lambda = params(0,col);


        krr_trainer<kernel_type> trainer_cv;
        trainer_cv.set_lambda(lambda);

        matrix<double> result = cross_validate_regression_trainer(trainer_cv, fvect_arr, target_arr,5);

       if(sum(result)> sum(best_result))
          {
             best_result = result;
             best_lambda = lambda;
              }
    }
     if(comm->me == 0) {
             cout <<"\n best result of grid search: " <<sum(best_result) <<endl;
             cout <<"  best lambda: "<<best_lambda<<endl;
	}
    trainer.set_lambda(best_lambda);
    final_pot_trainer = trainer.train(fvect_arr, target_arr);

   //calculate the derivate of kernel	
   	for(int j=0;j<nfeature;j++)
	   fvect_dev(j) = 0.0;

    for(int i=0;i<final_pot_trainer.basis_vectors.nr();i++)
       fvect_dev += final_pot_trainer.alpha(i)*final_pot_trainer.basis_vectors(i);
   
   double inter_p = final_pot_trainer.b;
   for(int j=0;j<nfeature;j++){
       fvect_dev(j) = fvect_dev(j)*normalizer.std_devs()(j);
     inter_p += normalizer.means()(j)*fvect_dev(j);
   }   

    zero_atom_energy = -1.0*inter_p;   

  if(comm->me == 0) {
    FILE * fp= fopen("Param_ML_pot.txt","w");
    fprintf(fp,"# Fitted ML parameters\n");
    fprintf(fp,"# Zr 91.22\n");
    fprintf(fp,"%d %d\n",nfeature, ntarget);    
    for(int j=0;j<nfeature;j++)
      fprintf(fp,"%lg\n",fvect_dev[j]);
    fprintf(fp,"%lg\n",inter_p);
    fclose(fp);
   }  */
   //cout<<fvect_dev<<endl;
  // cout<<final_pot_trainer.b<<endl;
}

int PairMLEnergy::pos2b_fun(int type1, int type2, int nv)
{
	int map1 = map[type1];
	int map2 = map[type2];
	
        return (map1+map2-2)*nv;
}
int PairMLEnergy::pos2bs_fun(int type1, int type2, int nv)
{
	int map1 = map[type1];
	int map2 = map[type2];
	return (2*map1+map2-3)*nv;	
}

int PairMLEnergy::pos3b_fun(int type1, int type2, int type3, int nv)
{
	int map1 = map[type1];
	int map2 = map[type2];
	int map3 = map[type3];
        int value;

//        printf("%d %d , %d %d, %d %d\n",type1,map1,type2,map2,type3,map3);
 	
            value = (map2+map3-2)*nv;

	if(map1==1)
            return value;
	else if(map1==2)
	    return value + 3*nv;
}

double PairMLEnergy::fun_cutoff(double r,double Rc)
{
        double Pi_value = 3.14159626;

	if(r>Rc)
		return 0.0;
	else
		return 0.5*(cos(Pi_value*r/Rc)+1.0);
}


double PairMLEnergy::fun_cutoff(double r,double Rc,double gramma)
{
	
	if(r>Rc)
		return 0.0;
	else
		return 1+ (gramma*r/Rc-gramma -1)*pow(r/Rc,gramma);
}


double PairMLEnergy::fun_cutoff_dev(double r,double Rc,double gramma)
{
	
	if(r>Rc)
		return 0.0;
	else
		return gramma*(gramma+1)*pow(r/Rc,gramma-1)*(-1.0+r/Rc)/Rc;
}

double PairMLEnergy::Hx(double r)
{
    if(r>0)
       return 1.0;
     else
       return 0.0;
}

void PairMLEnergy::costheta_d(double cos_theta, const double rij_hat[3], double rij,
			     const double rik_hat[3], double rik,
			     double *cos_drj, double *cos_drk)
{
  // first element is devative  wrt Rj, second wrt Rk

  vec3_scaleadd(-cos_theta,rij_hat,rik_hat,cos_drj);
  vec3_scale(1.0/rij,cos_drj,cos_drj);
  vec3_scaleadd(-cos_theta,rik_hat,rij_hat,cos_drk);
  vec3_scale(1.0/rik,cos_drk,cos_drk);

}



double PairMLEnergy::EAM_fpair(double r,int type1, int type2)
{
   double value;
   
   if(type1==1&&type2==1)
      value = 0.0;
   else if(type1==2&&type2==2)
      value = 0.0;
   else 
      value = 0.0;

   return value/r;
}

double PairMLEnergy::EAM_Epair(double r,int type1, int type2)
{
    double value;
   
   if(type1==1&&type2==1)
      value = 0.0;
   else if(type1==2&&type2==2)
      value = 0.0;
   else 
      value = 0.0;

   return value;
}


double PairMLEnergy::embed_fun(double rho, double rho_cent)
{
    return -0.5+1.0/(1+exp(-1.0+rho/rho_cent));
}


double PairMLEnergy::embed_fun_dev(double rho,double rho_cent)
{
     double value = 1.0/(1+exp(-1.0+rho/rho_cent));
     return value*(value-1.0)/rho_cent;
}

/*
double PairMLEnergy::LJ_fpair(double r)
{

   double value;
    value = -250.64*pow(r,-5);
    //value = -105258.61*exp(-1.0*r/0.2631);
   return value/r;

}

double PairMLEnergy::LJ_Epair(double r)
{
  double value;
    value = 62.66*pow(r,-4);
   // value = 27693.54*exp(-1.0*r/0.2613);
   return value;
}*/
/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
void PairMLEnergy::init_style()
{
        if(force->newton_pair == 0)
                error->all(FLERR,"Pair style ml/energy requires newton pair on");

        // Need both full and half neighbor list.
        int irequest_full = neighbor->request(this,instance_me);
        neighbor->requests[irequest_full]->id = 1;
        neighbor->requests[irequest_full]->half = 0;
        neighbor->requests[irequest_full]->full = 1;
        int irequest_half = neighbor->request(this,instance_me);
        neighbor->requests[irequest_half]->id = 2;
        neighbor->requests[irequest_half]->half = 0;
        //neighbor->requests[irequest_half]->half_from_full = 1;
        //neighbor->requests[irequest_half]->otherlist = irequest_full;

}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   half or full
------------------------------------------------------------------------- */
void PairMLEnergy::init_list(int id, NeighList *ptr)
{
        if(id == 1) listfull = ptr;
        else if(id == 2) listhalf = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairMLEnergy::init_one(int i, int j)
{
        return cutoff;
}

/* ---------------------------------------------------------------------- */

int PairMLEnergy::pack_forward_comm(int n, int *list, double *buf, 
                                      int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
	for (k = 0; k < 32; k++) 
	  buf[m++] = rho_values[j][k];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairMLEnergy::unpack_forward_comm(int n, int first, double *buf)
{
    int i,k,m,last;

  m = 0;
 last = first + n;
  for (i = first; i < last; i++){
   for (k = 0; k < 32; k++) 
      rho_values[i][k] = buf[m++];
    }

  
}

/* ---------------------------------------------------------------------- */

int PairMLEnergy::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

   m = 0;
  last = first + n;
  for (i = first; i < last; i++)
	 for (k = 0; k < 32; k++) 
	  buf[m++] = rho_values[i][k];
  return m;
}

/* ---------------------------------------------------------------------- */

void PairMLEnergy::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
      j = list[i];
   for (k = 0; k < 32; k++) 
      rho_values[j][k] = buf[m++];
    }
}

/* ----------------------------------------------------------------------
   Returns memory usage of local atom-based arrays
------------------------------------------------------------------------- */
double PairMLEnergy::memory_usage()
{
        return nmax *32* sizeof(double);
}

