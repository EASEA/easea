/*
 * CCmaesCuda.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef CCMAES_H_
#define CCMAES_H_
#include "define.h"
class CaleatoireCuda 
{
public:
	/* Variables for Uniform() */
	long int startseed;
	long int aktseed;
	long int aktalea;
	long int rgalea[32];
  
	/* Variables for Gauss() */
	short flgstored;
	float hold;
public:
	long alea_Start(long unsigned inseed);
	long alea_init(long unsigned inseed);
	float alea_Gauss();
	float alea_Uniform();
};

class CCmaesCuda{
	//random_t rand; /* random number generator */
public:	
	int dim;	
	float sigma;  /* step size */
	float *rgxmean;  /* mean x vector, "parent" */
	float chiN; 
	float **C;  /* lower triangular matrix: i>=j for C[i][j] */
	float **B;  /* matrix with normalize eigenvectors in columns */
	float *rgD; /* axis lengths */
	float *rgpc;
	float *rgps;
	float *rgxold; 
	float *rgout; 
	float *rgBDz;   /* for B*D*z */
	float *rgdTmp;  /* temporary (random) vector used in different places */
	short flgEigensysIsUptodate;
	short flgCheckEigen; /* control via signals.par */
  	float genOfEigensysUpdate;

	short flgIniphase;
	
	int lambda;          /* -> mu, <- N */
	int mu;              /* -> weights, (lambda) */
	float mucov, mueff; /* <- weights */
	float *weights;     /* <- mu, -> mueff, mucov, ccov */
	float damps;        /* <- cs, maxeval, lambda */
	float cs;           /* -> damps, <- N */
	float ccumcov;      /* <- N */
	float ccov;         /* <- mucov, <- N */
	int gen;

	float * xstart; 
	float * typicalX; 
	int typicalXcase;
	float * rgInitialStds;
	float * rgDiffMinChange;

	struct { int flgalways; float modulo; float maxtime; } updateCmode;
	float facupdateCmode;
	
	CaleatoireCuda alea; /* random number generator */
	int seed;
public:
	CCmaesCuda(int lambda, int mu, int problemdim);
	~CCmaesCuda();
	void cmaes_update(float **popparent, float *fitness);
	void Cmaes_init_param(int lambda, int mu);
	void TestMinStdDevs();
	void cmaes_UpdateEigensystem(int flgforce);
	void Adapt_C2(int hsig, float **parents);
};

int Check_Eigen(int taille,  float **C, float *diag, float **Q);
float myhypot(float a, float b);
void Householder2(int n, float **V, float *d, float *e);
void QLalgo2 (int n, float *d, float *e, float **V);
void Eigen( int taille,  float **C, float *diag, float **Q, float *rgtmp);


#endif /* CCCmaesCuda*ES_H_ */
