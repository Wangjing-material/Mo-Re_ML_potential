/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(tabMLP, PairTabMLP)

#else

#ifndef LMP_PAIR_TABMLP_H
#define LMP_PAIR_TABMLP_H

#include "pair.h"
#include <cstdio>

namespace LAMMPS_NS {

class PairTabMLP : public Pair {
public:
  PairTabMLP(class LAMMPS *);
  virtual ~PairTabMLP();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual void init_style();
  double init_one(int, int);
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;
  // double single(int, int, int, int, double, double, double, double &);
  // virtual void *extract(const char *, int &);

protected:
  int **map2b;  // which element each atom type maps to,
                // [0,0],[0,1],[1,1],[0,2],...
  int ***map3b; // which element each atom type maps to
  int **map_rho;
  char **elements;

  bool compute2b, compute3b,compute_rho;
  int n_2body, n_3body,n_rho;
  int nmax; 

  int n_e0;
  double *e0;

  double *lo_2body, *hi_2body,*lo_rho, *hi_rho;
  double **lo_3body, **hi_3body;
  int *grid_2body, **grid_3body,*grid_rho;
  double **fcoeff_2body, **fcoeff_3body,**fcoeff_rho;
  double **ecoeff_2body, **ecoeff_3body,**ecoeff_rho;
  double **rho_value,***frho_value,*param;

  int maxshort;
  int *neighshort;
   bigint embedstep;

  // extra cutoff for 3d and short list
  double cutmax;
  double *cut2bsq;
  double *cut3bsq;
  double *cutrhosq;
  double cutshortsq;

  // parameters for spline
  double Ad[4][4];
  double dAd[4][4];
  double d2Ad[4][4];
  double Bd[4];
  double Cd[4];
  double basis[4];

  void allocate();

  void eval_cubic_splines_1d(double, double, int, double *, double, double *,
                             double *);
  void eval_cubic_splines_1d_energy(double, double, int, double *, double, double *);
  void eval_cubic_splines_1d_force(double, double, int, double *, double, double *);
  void eval_cubic_splines_3d(double *, double *, int *, double *, double,
                             double, double, double *, double *);

  void read_file(char *);
  void bcast_table();
  void grab(FILE *, int, double *);
};

} // namespace LAMMPS_NS

#endif
#endif

    /* ERROR/WARNING messages:

    E: Illegal ... command

    Self-explanatory.  Check the input script syntax and compare to the
    documentation for the command.  You can use -echo screen as a
    command-line option when running LAMMPS to see the offending line.

    E: Incorrect args for pair coefficients

    Self-explanatory.  Check the input script or data file.

    E: Cannot open tabMLP potential file %s

    The specified tabMLP potential file cannot be opened.  Check that the
    path and name are correct.

    E: Invalid tabMLP potential file

    UNDOCUMENTED

    */
