#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/CCmaes.h"

//Functions for cma
long Caleatoire::alea_Start(long unsigned inseed)
{
	long tmp;
	int i;

	this->flgstored = 0;
	this->startseed = inseed;
	if (inseed < 1)
		inseed = 1; 
	this->aktseed = inseed;
	for (i = 39; i >= 0; --i)
	{
		tmp = this->aktseed/127773;
		this->aktseed = 16807 * (this->aktseed - tmp * 127773)
				- 2836 * tmp;
		if (this->aktseed < 0) this->aktseed += 2147483647;
		if (i < 32)
			this->rgalea[i] = this->aktseed;
	}
	this->aktalea = this->rgalea[0];
	return inseed;
}

long Caleatoire::alea_init(long unsigned inseed)
{
	clock_t cloc = clock();

	this->flgstored = 0;
	if (inseed < 1) {
		while ((long) (cloc - clock()) == 0)
			; /* TODO: remove this for time critical applications? */
		inseed = (long)abs((long)(100*time(NULL)+clock()));
	}
	return this->alea_Start(inseed);
}

double Caleatoire::alea_Uniform()
{
	long tmp;

	tmp = this->aktseed/127773;
	this->aktseed = 16807 * (this->aktseed - tmp * 127773)
			- 2836 * tmp;
	if (this->aktseed < 0) 
		this->aktseed += 2147483647;
	tmp = this->aktalea / 67108865;
	this->aktalea = this->rgalea[tmp];
	this->rgalea[tmp] = this->aktseed;
	return (double)(this->aktalea)/(2.147483647e9);
}

double Caleatoire::alea_Gauss()
{
	double x1, x2, rquad, fac;

	if (this->flgstored)
	{    
		this->flgstored = 0;
		return this->hold;
	}
	do 
	{
		x1 = 2.0 * this->alea_Uniform() - 1.0;
		x2 = 2.0 * this->alea_Uniform() - 1.0;
		rquad = x1*x1 + x2*x2;
	} while(rquad >= 1 || rquad <= 0);
	fac = sqrt(-2.0*log(rquad)/rquad);
	this->flgstored = 1;
	this->hold = fac * x1;
	return fac * x2;
}

void CCmaes::Cmaes_init_param(int lambda, int mu){
double s1, s2;
double t1, t2;
int i;
clock_t cloc = clock();

this->lambda = lambda;
this->mu = mu;
/*set weights*/
this->weights = (double*)malloc(this->mu*sizeof(double));
for (i=0; i<this->mu; ++i) 
      this->weights[i] = log(this->mu+1.)-log(i+1.);
/* normalize weights vector and set mueff */
s1=0., s2=0.;
for (i=0; i<this->mu; ++i) {
    s1 += this->weights[i];
    s2 += this->weights[i]*this->weights[i];
}
this->mueff = s1*s1/s2;
for (i=0; i<this->mu; ++i) 
    this->weights[i] /= s1;
if(this->mu < 1 || this->mu > this->lambda || (this->mu==this->lambda && this->weights[0]==this->weights[this->mu-1])){
    printf("readpara_SetWeights(): invalid setting of mu or lambda\n");
	exit(0);
}

/*supplement defaults*/
this->cs = (this->mueff + 2.) / (this->dim + this->mueff + 3.);
this->ccumcov = 4. / (this->dim + 4);
this->mucov = this->mueff;

t1 = 2. / ((this->dim+1.4142)*(this->dim+1.4142));
t2 = (2.*this->mueff-1.) / ((this->dim+2.)*(this->dim+2.)+this->mueff);
t2 = (t2 > 1) ? 1 : t2;
t2 = (1./this->mucov) * t1 + (1.-1./this->mucov) * t2;

this->ccov = t2;

//this->stopMaxIter = ceil((double)(this->stopMaxFunEvals / this->lambda));

this->damps = 1;

this->damps = this->damps * (1 + 2*MAX(0., sqrt((this->mueff-1.)/(this->dim+1.)) - 1)) * 0.3 + this->cs;

this->updateCmode.modulo = 1./this->ccov/(double)(this->dim)/10.;
this->updateCmode.modulo *= this->facupdateCmode;

while ((int) (cloc - clock()) == 0)
; /* TODO: remove this for time critical applications!? */
	this->seed = (unsigned int)abs((long)(100*time(NULL)+clock()));
}

void CCmaes::TestMinStdDevs()
/* increases sigma */
{
	int i; 
	if (this->rgDiffMinChange == NULL)
		return;
	else{
	for (i = 0; i < this->dim; ++i)
		while (this->sigma * sqrt(this->C[i][i]) < this->rgDiffMinChange[i]) 
			this->sigma *= exp(0.05+this->cs/this->damps);
	}

} /* cmaes_TestMinStdDevs() */

int Check_Eigen(int taille,  double **C, double *diag, double **Q) 
{
	/* compute Q diag Q^T and Q Q^T to check */
	int i, j, k, res = 0;
	double cc, dd; 

	for (i=0; i < taille; ++i)
		for (j=0; j < taille; ++j) {
		for (cc=0.,dd=0., k=0; k < taille; ++k) {
			cc += diag[k] * Q[i][k] * Q[j][k];
			dd += Q[i][k] * Q[j][k];
		}
		/* check here, is the normalization the right one? */
		if (fabs(cc - C[i>j?i:j][i>j?j:i])/sqrt(C[i][i]*C[j][j]) > 1e-10 && fabs(cc - C[i>j?i:j][i>j?j:i]) > 3e-14) 
		{
			printf("cmaes_t:Eigen(): imprecise result detected \n");
			++res; 
				  }
				  if (fabs(dd - (i==j)) > 1e-10) {
					  printf("cmaes_t:Eigen(): imprecise result detected (Q not orthog.)\n");
					  ++res;
				  }
		}
	return res; 
}

double myhypot(double a, double b) 
/* sqrt(a^2 + b^2) numerically stable. */
{
	double r = 0;
	if (fabs(a) > fabs(b)) {
		r = b/a;
		r = fabs(a)*sqrt(1+r*r);
	} else if (b != 0) {
		r = a/b;
		r = fabs(b)*sqrt(1+r*r);
	}
	return r;
}

void Householder2(int n, double **V, double *d, double *e) {
	int i,j,k; 

	for (j = 0; j < n; j++) {
		d[j] = V[n-1][j];
	}

	/* Householder reduction to tridiagonal form */
   
	for (i = n-1; i > 0; i--) {
   
		/* Scale to avoid under/overflow */
   
		double scale = 0.0;
		double h = 0.0;
		for (k = 0; k < i; k++) {
			scale = scale + fabs(d[k]);
		}
		if (scale == 0.0) {
			e[i] = d[i-1];
			for (j = 0; j < i; j++) {
				d[j] = V[i-1][j];
				V[i][j] = 0.0;
				V[j][i] = 0.0;
			}
		} else {
   
			/* Generate Householder vector */

			double f, g, hh;
   
			for (k = 0; k < i; k++) {
				d[k] /= scale;
				h += d[k] * d[k];
			}
			f = d[i-1];
			g = sqrt(h);
			if (f > 0) {
				g = -g;
			}
			e[i] = scale * g;
			h = h - f * g;
			d[i-1] = f - g;
			for (j = 0; j < i; j++) {
				e[j] = 0.0;
			}
   
			/* Apply similarity transformation to remaining columns */
   
			for (j = 0; j < i; j++) {
				f = d[j];
				V[j][i] = f;
				g = e[j] + V[j][j] * f;
				for (k = j+1; k <= i-1; k++) {
					g += V[k][j] * d[k];
					e[k] += V[k][j] * f;
				}
				e[j] = g;
			}
			f = 0.0;
			for (j = 0; j < i; j++) {
				e[j] /= h;
				f += e[j] * d[j];
			}
			hh = f / (h + h);
			for (j = 0; j < i; j++) {
				e[j] -= hh * d[j];
			}
			for (j = 0; j < i; j++) {
				f = d[j];
				g = e[j];
				for (k = j; k <= i-1; k++) {
					V[k][j] -= (f * e[k] + g * d[k]);
				}
				d[j] = V[i-1][j];
				V[i][j] = 0.0;
			}
		}
		d[i] = h;
	}
   
	/* Accumulate transformations */
   
	for (i = 0; i < n-1; i++) {
		double h; 
		V[n-1][i] = V[i][i];
		V[i][i] = 1.0;
		h = d[i+1];
		if (h != 0.0) {
			for (k = 0; k <= i; k++) {
				d[k] = V[k][i+1] / h;
			}
			for (j = 0; j <= i; j++) {
				double g = 0.0;
				for (k = 0; k <= i; k++) {
					g += V[k][i+1] * V[k][j];
				}
				for (k = 0; k <= i; k++) {
					V[k][j] -= g * d[k];
				}
			}
		}
		for (k = 0; k <= i; k++) {
			V[k][i+1] = 0.0;
		}
	}
	for (j = 0; j < n; j++) {
		d[j] = V[n-1][j];
		V[n-1][j] = 0.0;
	}
	V[n-1][n-1] = 1.0;
	e[0] = 0.0;

} /* Housholder() */

void QLalgo2 (int n, double *d, double *e, double **V) {
	int i, k, l, m;
	double f = 0.0;
	double tst1 = 0.0;
	double eps = 2.22e-16; /* Math.pow(2.0,-52.0);  == 2.22e-16 */
  
	/* shift input e */
	for (i = 1; i < n; i++) {
		e[i-1] = e[i];
	}
	e[n-1] = 0.0; /* never changed again */
   
	for (l = 0; l < n; l++) { 

		/* Find small subdiagonal element */
   
		if (tst1 < fabs(d[l]) + fabs(e[l]))
			tst1 = fabs(d[l]) + fabs(e[l]);
		m = l;
		while (m < n) {
			if (fabs(e[m]) <= eps*tst1) {
				/* if (fabs(e[m]) + fabs(d[m]+d[m+1]) == fabs(d[m]+d[m+1])) { */
				break;
			}
			m++;
		}
   
		/* If m == l, d[l] is an eigenvalue, */
		/* otherwise, iterate. */
   
		if (m > l) { 
			int iter = 0;
			do { /* while (fabs(e[l]) > eps*tst1); */
				double dl1, h;
				double g = d[l];
				double p = (d[l+1] - g) / (2.0 * e[l]); 
				double r = myhypot(p, 1.); 

				iter = iter + 1;  /* Could check iteration count here */
   
				/* Compute implicit shift */
   
				if (p < 0) {
					r = -r;
				}
				d[l] = e[l] / (p + r);
				d[l+1] = e[l] * (p + r);
				dl1 = d[l+1];
				h = g - d[l];
				for (i = l+2; i < n; i++) {
					d[i] -= h;
				}
				f = f + h;
   
				/* Implicit QL transformation. */
   
				p = d[m];
				{
					double c = 1.0;
					double c2 = c;
					double c3 = c;
					double el1 = e[l+1];
					double s = 0.0;
					double s2 = 0.0;
					for (i = m-1; i >= l; i--) {
						c3 = c2;
						c2 = c;
						s2 = s;
						g = c * e[i];
						h = c * p;
						r = myhypot(p, e[i]);
						e[i+1] = s * r;
						s = e[i] / r;
						c = p / r;
						p = c * d[i] - s * g;
						d[i+1] = h + s * (c * g + s * d[i]);
   
						/* Accumulate transformation. */
   
						for (k = 0; k < n; k++) {
							h = V[k][i+1];
							V[k][i+1] = s * V[k][i] + c * h;
							V[k][i] = c * V[k][i] - s * h;
						}
					}
					p = -s * s2 * c3 * el1 * e[l] / dl1;
					e[l] = s * p;
					d[l] = c * p;
				}
   
				/* Check for convergence. */
   
			} while (fabs(e[l]) > eps*tst1);
		}
		d[l] = d[l] + f;
		e[l] = 0.0;
	}     
/* Sort eigenvalues and corresponding vectors. */

	int j; 
	double p;
	for (i = 0; i < n-1; i++) {
		k = i;
		p = d[i];
		for (j = i+1; j < n; j++) {
			if (d[j] < p) {
				k = j;
				p = d[j];
			}
		}
		if (k != i) {
			d[k] = d[i];
			d[i] = p;
			for (j = 0; j < n; j++) {
				p = V[j][i];
				V[j][i] = V[j][k];
				V[j][k] = p;
			}
		}
	}
} /* QLalgo2 */ 

void Eigen( int taille,  double **C, double *diag, double **Q, double *rgtmp)
{
	int i, j;
	if (rgtmp == NULL) /* was OK in former versions */
		printf("cmaes_t:Eigen(): input parameter double *rgtmp must be non-NULL\n");

	/* copy C to Q */
	if (C != Q) {
		for (i=0; i < taille; ++i)
			for (j = 0; j <= i; ++j)
				Q[i][j] = Q[j][i] = C[i][j];
	}
	Householder2( taille, Q, diag, rgtmp);
	QLalgo2( taille, diag, rgtmp, Q);
}

void CCmaes::cmaes_UpdateEigensystem(int flgforce)
{
  int i;

  if(flgforce == 0) {
    if (this->flgEigensysIsUptodate == 1)
      return; 

    /* return on modulo generation number */ 
    if (this->gen < this->genOfEigensysUpdate + this->updateCmode.modulo)
      return;
  }

  Eigen( this->dim, this->C, this->rgD, this->B, this->rgdTmp);

  if (this->flgCheckEigen)
    /* needs O(n^3)! writes, in case, error message in error file */ 
    i = Check_Eigen( this->dim, this->C, this->rgD, this->B);
  
  for (i = 0; i < this->dim; ++i)
    this->rgD[i] = sqrt(this->rgD[i]);
  
  this->flgEigensysIsUptodate = 1;
  this->genOfEigensysUpdate = this->gen; 
  
  return;
} /* cmaes_UpdateEigensystem() */

/*Function for CMA*/
CCmaes::CCmaes(int lambda, int mu, int problemdim){
	int i, j;
	double trace;
	/*read param*/
	this->gen = 0;
	this->xstart = NULL; 
	this->typicalX = NULL;
	this->typicalXcase = 0;
	this->rgInitialStds = NULL; 
	this->rgDiffMinChange = NULL;
	this->lambda = lambda;
	this->dim = problemdim;
	this->mu = -1;
	this->mucov = -1;
	this->weights = NULL;
	this->cs = -1;
	this->ccumcov = -1;
	this->damps = -1;
	this->ccov = -1;
	this->updateCmode.modulo = -1;  
	this->updateCmode.maxtime = -1;
	this->updateCmode.flgalways = 0;
	this->facupdateCmode = 1;
	this->flgIniphase = 0;

	this->xstart = (double*)malloc(this->dim*sizeof(double));
	
	this->typicalXcase = 1;
	for (i=0; i<this->dim; ++i)
		this->xstart[i] = 0.5;

	this->rgInitialStds = (double*)malloc(this->dim*sizeof(double));
	for (i=0; i<this->dim; ++i)
	      this->rgInitialStds[i] = 0.3;

	this->Cmaes_init_param(lambda,mu);

	this->seed = this->alea.alea_init((unsigned) this->seed);

	/* initialization  */
	for (i = 0, trace = 0.; i < this->dim; ++i)
		trace += this->rgInitialStds[i]*this->rgInitialStds[i];
	this->sigma = sqrt(trace/this->dim); /* this->sp.mueff/(0.2*this->mueff+sqrt(this->dim)) * sqrt(trace/this->dim); */

	this->chiN = sqrt((double) this->dim) * (1. - 1./(4.*this->dim) + 1./(21.*this->dim*this->dim));
	this->flgEigensysIsUptodate = 1;
	this->flgCheckEigen = 0;
	this->genOfEigensysUpdate = 0;

	this->rgpc = (double*)malloc(this->dim*sizeof(double));
	this->rgps = (double*)malloc(this->dim*sizeof(double));
	this->rgdTmp = (double*)malloc((this->dim+1)*sizeof(double));
	this->rgBDz = (double*)malloc(this->dim*sizeof(double));
	this->rgxmean = (double*)malloc(this->dim*sizeof(double));
	this->rgxold = (double*)malloc(this->dim*sizeof(double));  
	this->rgD = (double*)malloc(this->dim*sizeof(double));
	this->C = (double**)malloc(this->dim*sizeof(double*));
	this->B = (double**)malloc(this->dim*sizeof(double*));

	for (i = 0; i < this->dim; ++i) {
		this->C[i] = (double*)malloc((i+1)*sizeof(double));;
		this->B[i] = (double*)malloc(this->dim*sizeof(double));
	}
	/* Initialize newed space  */

	for (i = 0; i < this->dim; ++i)
		for (j = 0; j < i; ++j){
			this->C[i][j] = this->B[i][j] = this->B[j][i] = 0.;
		}
	for (i = 0; i < this->dim; ++i)
	{
		this->B[i][i] = 1.;
		this->C[i][i] = this->rgD[i] = this->rgInitialStds[i] * sqrt(this->dim / trace);
		this->C[i][i] *= this->C[i][i];
		this->rgpc[i] = this->rgps[i] = this->rgdTmp[i] = 0.;
	}

  //initialise mean;
  	for ( i = 0; i < this->dim; ++i){
		this->rgxmean[i] = 0.5 + (this->sigma * this->rgD[i] * this->alea.alea_Gauss());
    		this->rgxold[i] = this->rgxmean[i];
  	}
}

void CCmaes::Adapt_C2(int hsig, double **parents)
{
	int i, j, k;
	if (this->ccov != 0. && this->flgIniphase == 0) {
		
		/* definitions for speeding up inner-most loop */
		double ccovmu = this->ccov * (1-1./this->mucov); 
		double sigmasquare = this->sigma * this->sigma; 

		this->flgEigensysIsUptodate = 0;

		/* update covariance matrix */
		for (i = 0; i < this->dim; ++i)
			for (j = 0; j <=i; ++j) {
				this->C[i][j] = (1 - this->ccov) * this->C[i][j] + this->ccov * (1./this->mucov) * (this->rgpc[i] * this->rgpc[j] + (1-hsig)*this->ccumcov*(2.-this->ccumcov) * this->C[i][j]);
			for (k = 0; k < this->mu; ++k) { /* additional rank mu update */
				this->C[i][j] += ccovmu * this->weights[k] * (parents[k][i] - this->rgxold[i]) * (parents[k][j] - this->rgxold[j]) / sigmasquare;
			}
		}

	}
}

CCmaes::~CCmaes()
{
	int i;
	free( this->rgpc);
	free( this->rgps);
	free( this->rgdTmp);
	free( this->rgBDz);
	free( this->rgxmean);
	free( this->rgxold); 
	free( this->rgD);
	for (i = 0; i < this->dim; ++i) {
		free( this->C[i]);
		free( this->B[i]);
	}
	free( this->C);
	free( this->B);
} /* cmaes_exit() */

void CCmaes::cmaes_update(double **popparent, double *fitpar){
    	int i, j, iNk, hsig;
	double sum; 
	double psxps;
	if (fitpar[0] == fitpar[(int)this->mu/2]){
		this->sigma *= exp(0.2+this->cs/this->damps);
		printf("Warning: sigma increased due to equal function values\n");
		printf("   Reconsider the formulation of the objective function\n");
	}
	for (i = 0; i < this->dim; ++i) {
		this->rgxold[i] = this->rgxmean[i]; 
		this->rgxmean[i] = 0.;
		for (iNk = 0; iNk < this->mu; ++iNk) 
			this->rgxmean[i] += this->weights[iNk] * popparent[iNk][i];
		this->rgBDz[i] = sqrt(this->mueff)*(this->rgxmean[i] - this->rgxold[i])/this->sigma; 
	}
	/* calculate z := D^(-1) * B^(-1) * rgBDz into rgdTmp */
	for (i = 0; i < this->dim; ++i) {
		sum = 0.;
		for (j = 0; j < this->dim; ++j)
			sum += this->B[j][i] * this->rgBDz[j];
		this->rgdTmp[i] = sum / this->rgD[i];
	}
	/* cumulation for sigma (ps) using B*z */
	for (i = 0; i < this->dim; ++i) {
		sum = 0.;
		for (j = 0; j < this->dim; ++j)
			sum += this->B[i][j] * this->rgdTmp[j];
		this->rgps[i] = (1. - this->cs) * this->rgps[i] + 
				sqrt(this->cs * (2. - this->cs)) * sum;
	}
	/* calculate norm(ps)^2 */
	psxps = 0.;
	for (i = 0; i < this->dim; ++i)
		psxps += this->rgps[i] * this->rgps[i];
	/* cumulation for covariance matrix (pc) using B*D*z~N(0,C) */
	hsig = (sqrt(psxps) / sqrt(1. - pow(1.-this->cs, 2*this->gen)) / this->chiN) < (1.4 + 2./(this->dim+1));
	for (i = 0; i < this->dim; ++i) {
		this->rgpc[i] = (1. - this->ccumcov) * this->rgpc[i] + hsig * sqrt(this->ccumcov * (2. - this->ccumcov)) * this->rgBDz[i];
	}
	/* stop initial phase */
	if (this->flgIniphase && this->gen > MIN(1/this->cs, 1+this->dim/this->mucov)) 
	{
		if (psxps / this->damps / (1.-pow((1. - this->cs), this->gen)) < this->dim * 1.05) 
			this->flgIniphase = 0;
	}
	this->Adapt_C2(hsig, popparent);
	/* update of sigma */
	this->sigma *= exp(((sqrt(psxps)/this->chiN)-1.)*this->cs/this->damps);
	this->gen++;
}

