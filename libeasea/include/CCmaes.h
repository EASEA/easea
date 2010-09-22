/*
 * CCmaes.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef CCMAES_H_
#define CCMAES_H_
#include "define.h"
class Caleatoire 
{
public:
	/* Variables for Uniform() */
	long int startseed;
	long int aktseed;
	long int aktalea;
	long int rgalea[32];
  
	/* Variables for Gauss() */
	short flgstored;
	double hold;
public:
	long alea_Start(long unsigned inseed);
	long alea_init(long unsigned inseed);
	double alea_Gauss();
	double alea_Uniform();
};

class CCmaes{
	//random_t rand; /* random number generator */
public:	
	int dim;	
	double sigma;  /* step size */
	double *rgxmean;  /* mean x vector, "parent" */
	double chiN; 
	double **C;  /* lower triangular matrix: i>=j for C[i][j] */
	double **B;  /* matrix with normalize eigenvectors in columns */
	double *rgD; /* axis lengths */
	double *rgpc;
	double *rgps;
	double *rgxold; 
	double *rgout; 
	double *rgBDz;   /* for B*D*z */
	double *rgdTmp;  /* temporary (random) vector used in different places */
	short flgEigensysIsUptodate;
	short flgCheckEigen; /* control via signals.par */
  	double genOfEigensysUpdate;

	short flgIniphase;
	
	int lambda;          /* -> mu, <- N */
	int mu;              /* -> weights, (lambda) */
	double mucov, mueff; /* <- weights */
	double *weights;     /* <- mu, -> mueff, mucov, ccov */
	double damps;        /* <- cs, maxeval, lambda */
	double cs;           /* -> damps, <- N */
	double ccumcov;      /* <- N */
	double ccov;         /* <- mucov, <- N */
	int gen;

	double * xstart; 
	double * typicalX; 
	int typicalXcase;
	double * rgInitialStds;
	double * rgDiffMinChange;

	struct { int flgalways; double modulo; double maxtime; } updateCmode;
	double facupdateCmode;
	
	Caleatoire alea; /* random number generator */
	int seed;
public:
	CCmaes(int lambda, int mu, int problemdim);
	~CCmaes();
	void cmaes_update(double **popparent, double *fitness);
	void Cmaes_init_param(int lambda, int mu);
	void TestMinStdDevs();
	void cmaes_UpdateEigensystem(int flgforce);
	void Adapt_C2(int hsig, double **parents);
};

int Check_Eigen(int taille,  double **C, double *diag, double **Q);
double myhypot(double a, double b);
void Householder2(int n, double **V, double *d, double *e);
void QLalgo2 (int n, double *d, double *e, double **V);
void Eigen( int taille,  double **C, double *diag, double **Q, double *rgtmp);


#endif /* CCCmaes*ES_H_ */
