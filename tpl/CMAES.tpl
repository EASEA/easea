\TEMPLATE_START/**
 This is program entry for CUDA template for EASEA

*/
\ANALYSE_PARAMETERS
using namespace std;
#include <iostream>
#include "EASEATools.hpp"
#include "EASEAIndividual.hpp"
#include <time.h>

RandomGenerator* globalRandomGenerator;
size_t *EZ_NB_GEN;

int main(int argc, char** argv){


  parseArguments("EASEA.prm",argc,argv);

  size_t parentPopulationSize = setVariable("popSize",\POP_SIZE);
  size_t offspringPopulationSize = setVariable("nbOffspring",\OFF_SIZE);
  float pCrossover = \XOVER_PROB;
  float pMutation = \MUT_PROB;
  float pMutationPerGene = 0.05;

  time_t seed = setVariable("seed",time(0));
  globalRandomGenerator = new RandomGenerator(seed);

  std::cout << "Seed is : " << seed << std::endl;

  SelectionOperator* selectionOperator = new \SELECTOR;
  SelectionOperator* replacementOperator = new \RED_FINAL;
  SelectionOperator* parentReductionOperator = new \RED_PAR;
  SelectionOperator* offspringReductionOperator = new \RED_OFF;
  float selectionPressure = \SELECT_PRM;
  float replacementPressure = \RED_FINAL_PRM;
  float parentReductionPressure = \RED_PAR_PRM;
  float offspringReductionPressure = \RED_OFF_PRM;

  string outputfile = setVariable("outputfile","");
  string inputfile = setVariable("inputfile","");

  EASEAInit(argc,argv);
    
  EvolutionaryAlgorithm ea(parentPopulationSize,offspringPopulationSize,selectionPressure,replacementPressure,parentReductionPressure,offspringReductionPressure,
			   selectionOperator,replacementOperator,parentReductionOperator, offspringReductionOperator, pCrossover, pMutation, pMutationPerGene,
			   outputfile,inputfile);

  StoppingCriterion* sc = new GenerationalCriterion(&ea,setVariable("nbGen",\NB_GEN));
  ea.addStoppingCriterion(sc);
  EZ_NB_GEN=((GenerationalCriterion*)ea.stoppingCriteria[0])->getGenerationalLimit();
  Population* pop = ea.getPopulation();


  ea.runEvolutionaryLoop();

  EASEAFinal(pop);

  delete pop;
  delete sc;
  delete selectionOperator;
  delete replacementOperator;
  delete globalRandomGenerator;


  return 0;
}


\START_CUDA_GENOME_CU_TPL
#include "EASEAIndividual.hpp"
#include "EASEAUserClasses.hpp"
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include <math.h>

#define CMAES_TPL

extern RandomGenerator* globalRandomGenerator;

//Déclarations nécéssaires pour cma-es
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
typedef struct 
{
	/* Variables for Uniform() */
	long int startseed;
	long int aktseed;
	long int aktalea;
	long int rgalea[32];
  
	/* Variables for Gauss() */
	short flgstored;
	double hold;
} aleatoire_t;

typedef struct{
	//random_t rand; /* random number generator */
		
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
	
	aleatoire_t alea; /* random number generator */
	int seed;
}CMA;

CMA cma;
int nbEnfants;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

//Functions for cma
long alea_Start( aleatoire_t *t, long unsigned inseed)
{
	long tmp;
	int i;

	t->flgstored = 0;
	t->startseed = inseed;
	if (inseed < 1)
		inseed = 1; 
	t->aktseed = inseed;
	for (i = 39; i >= 0; --i)
	{
		tmp = t->aktseed/127773;
		t->aktseed = 16807 * (t->aktseed - tmp * 127773)
				- 2836 * tmp;
		if (t->aktseed < 0) t->aktseed += 2147483647;
		if (i < 32)
			t->rgalea[i] = t->aktseed;
	}
	t->aktalea = t->rgalea[0];
	return inseed;
}

long alea_init(aleatoire_t *t, long unsigned inseed)
{
	clock_t cloc = clock();

	t->flgstored = 0;
	if (inseed < 1) {
		while ((long) (cloc - clock()) == 0)
			; /* TODO: remove this for time critical applications? */
		inseed = (long)abs(100*time(NULL)+clock());
	}
	return alea_Start(t, inseed);
}

double alea_Uniform( aleatoire_t *t)
{
	long tmp;

	tmp = t->aktseed/127773;
	t->aktseed = 16807 * (t->aktseed - tmp * 127773)
			- 2836 * tmp;
	if (t->aktseed < 0) 
		t->aktseed += 2147483647;
	tmp = t->aktalea / 67108865;
	t->aktalea = t->rgalea[tmp];
	t->rgalea[tmp] = t->aktseed;
	return (double)(t->aktalea)/(2.147483647e9);
}

double alea_Gauss(aleatoire_t *t)
{
	double x1, x2, rquad, fac;

	if (t->flgstored)
	{    
		t->flgstored = 0;
		return t->hold;
	}
	do 
	{
		x1 = 2.0 * alea_Uniform(t) - 1.0;
		x2 = 2.0 * alea_Uniform(t) - 1.0;
		rquad = x1*x1 + x2*x2;
	} while(rquad >= 1 || rquad <= 0);
	fac = sqrt(-2.0*log(rquad)/rquad);
	t->flgstored = 1;
	t->hold = fac * x1;
	return fac * x2;
}

void CMA_init_param(CMA *t, int lambda, int mu){
double s1, s2;
double t1, t2;
int i;
clock_t cloc = clock();

t->lambda = lambda;
t->mu = mu;
/*set weights*/
t->weights = (double*)malloc(t->mu*sizeof(double));
for (i=0; i<t->mu; ++i) 
      t->weights[i] = log(t->mu+1.)-log(i+1.);
/* normalize weights vector and set mueff */
s1=0., s2=0.;
for (i=0; i<t->mu; ++i) {
    s1 += t->weights[i];
    s2 += t->weights[i]*t->weights[i];
}
t->mueff = s1*s1/s2;
for (i=0; i<t->mu; ++i) 
    t->weights[i] /= s1;
if(t->mu < 1 || t->mu > t->lambda || (t->mu==t->lambda && t->weights[0]==t->weights[t->mu-1])){
    printf("readpara_SetWeights(): invalid setting of mu or lambda\n");
	exit(0);
}

/*supplement defaults*/
t->cs = (t->mueff + 2.) / (\PROBLEM_DIM + t->mueff + 3.);
t->ccumcov = 4. / (\PROBLEM_DIM + 4);
t->mucov = t->mueff;

t1 = 2. / ((\PROBLEM_DIM+1.4142)*(\PROBLEM_DIM+1.4142));
t2 = (2.*t->mueff-1.) / ((\PROBLEM_DIM+2.)*(\PROBLEM_DIM+2.)+t->mueff);
t2 = (t2 > 1) ? 1 : t2;
t2 = (1./t->mucov) * t1 + (1.-1./t->mucov) * t2;

t->ccov = t2;

//t->stopMaxIter = ceil((double)(t->stopMaxFunEvals / t->lambda));

t->damps = 1;

t->damps = t->damps * (1 + 2*MAX(0., sqrt((t->mueff-1.)/(\PROBLEM_DIM+1.)) - 1)) * 0.3 + t->cs;

t->updateCmode.modulo = 1./t->ccov/(double)(\PROBLEM_DIM)/10.;
t->updateCmode.modulo *= t->facupdateCmode;

while ((int) (cloc - clock()) == 0)
; /* TODO: remove this for time critical applications!? */
	t->seed = (unsigned int)abs(100*time(NULL)+clock());
}

static void TestMinStdDevs(CMA *t)
/* increases sigma */
{
	int i; 
	if (t->rgDiffMinChange == NULL)
		return;

	for (i = 0; i < \PROBLEM_DIM; ++i)
		while (t->sigma * sqrt(t->C[i][i]) < t->rgDiffMinChange[i]) 
			t->sigma *= exp(0.05+t->cs/t->damps);

} /* cmaes_TestMinStdDevs() */

int Check_Eigen(int SIZE,  double **C, double *diag, double **Q) 
{
	/* compute Q diag Q^T and Q Q^T to check */
	int i, j, k, res = 0;
	double cc, dd; 

	for (i=0; i < SIZE; ++i)
		for (j=0; j < SIZE; ++j) {
		for (cc=0.,dd=0., k=0; k < SIZE; ++k) {
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

void Eigen( int SIZE,  double **C, double *diag, double **Q, double *rgtmp)
{
	int i, j;
	if (rgtmp == NULL) /* was OK in former versions */
		printf("cmaes_t:Eigen(): input parameter double *rgtmp must be non-NULL\n");

	/* copy C to Q */
	if (C != Q) {
		for (i=0; i < SIZE; ++i)
			for (j = 0; j <= i; ++j)
				Q[i][j] = Q[j][i] = C[i][j];
	}
	Householder2( SIZE, Q, diag, rgtmp);
	QLalgo2( SIZE, diag, rgtmp, Q);
}

void cmaes_UpdateEigensystem(CMA *t, int flgforce)
{
  int i;

  if(flgforce == 0) {
    if (t->flgEigensysIsUptodate == 1)
      return; 

    /* return on modulo generation number */ 
    if (t->gen < t->genOfEigensysUpdate + t->updateCmode.modulo)
      return;
  }

  Eigen( \PROBLEM_DIM, t->C, t->rgD, t->B, t->rgdTmp);

  if (t->flgCheckEigen)
    /* needs O(n^3)! writes, in case, error message in error file */ 
    i = Check_Eigen( \PROBLEM_DIM, t->C, t->rgD, t->B);
  
  for (i = 0; i < \PROBLEM_DIM; ++i)
    t->rgD[i] = sqrt(t->rgD[i]);
  
  t->flgEigensysIsUptodate = 1;
  t->genOfEigensysUpdate = t->gen; 
  
  return;
} /* cmaes_UpdateEigensystem() */

/*Function for CMA*/
void CMA_init(CMA *t, int lambda, int mu){
	int i, j;
	double trace;
	/*read param*/
	t->xstart = NULL; 
	t->typicalX = NULL;
	t->typicalXcase = 0;
	t->rgInitialStds = NULL; 
	t->rgDiffMinChange = NULL;
	t->lambda = lambda;
	t->mu = -1;
	t->mucov = -1;
	t->weights = NULL;
	t->cs = -1;
	t->ccumcov = -1;
	t->damps = -1;
	t->ccov = -1;
	t->updateCmode.modulo = -1;  
	t->updateCmode.maxtime = -1;
	t->updateCmode.flgalways = 0;
	t->facupdateCmode = 1;
	t->flgIniphase = 0;

	t->xstart = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	
	t->typicalXcase = 1;
	for (i=0; i<\PROBLEM_DIM; ++i)
		t->xstart[i] = 0.5;

	t->rgInitialStds = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	for (i=0; i<\PROBLEM_DIM; ++i)
	      t->rgInitialStds[i] = 0.3;

	CMA_init_param(t,lambda,mu);

	t->seed = alea_init( &t->alea, (unsigned) t->seed);

	/* initialization  */
	for (i = 0, trace = 0.; i < \PROBLEM_DIM; ++i)
		trace += t->rgInitialStds[i]*t->rgInitialStds[i];
	t->sigma = sqrt(trace/\PROBLEM_DIM); /* t->sp.mueff/(0.2*t->mueff+sqrt(\PROBLEM_DIM)) * sqrt(trace/\PROBLEM_DIM); */

	t->chiN = sqrt((double) \PROBLEM_DIM) * (1. - 1./(4.*\PROBLEM_DIM) + 1./(21.*\PROBLEM_DIM*\PROBLEM_DIM));
	t->flgEigensysIsUptodate = 1;
	t->flgCheckEigen = 0;
	t->genOfEigensysUpdate = 0;

	t->rgpc = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	t->rgps = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	t->rgdTmp = (double*)malloc((\PROBLEM_DIM+1)*sizeof(double));
	t->rgBDz = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	t->rgxmean = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	t->rgxold = (double*)malloc(\PROBLEM_DIM*sizeof(double));  
	t->rgD = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	t->C = (double**)malloc(\PROBLEM_DIM*sizeof(double*));
	t->B = (double**)malloc(\PROBLEM_DIM*sizeof(double*));

	for (i = 0; i < \PROBLEM_DIM; ++i) {
		t->C[i] = (double*)malloc((i+1)*sizeof(double));;
		t->B[i] = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	}
	/* Initialize newed space  */

	for (i = 0; i < \PROBLEM_DIM; ++i)
		for (j = 0; j < i; ++j){
			t->C[i][j] = t->B[i][j] = t->B[j][i] = 0.;
		}
	for (i = 0; i < \PROBLEM_DIM; ++i)
	{
		t->B[i][i] = 1.;
		t->C[i][i] = t->rgD[i] = t->rgInitialStds[i] * sqrt(\PROBLEM_DIM / trace);
		t->C[i][i] *= t->C[i][i];
		t->rgpc[i] = t->rgps[i] = 0.;
	}
}

void Adapt_C2(CMA *t, int hsig, Individual **parents)
{
	int i, j, k;
	if (t->ccov != 0. && t->flgIniphase == 0) {

		/* definitions for speeding up inner-most loop */
		double ccovmu = t->ccov * (1-1./t->mucov); 
		double sigmasquare = t->sigma * t->sigma; 

		t->flgEigensysIsUptodate = 0;

		/* update covariance matrix */
		for (i = 0; i < \PROBLEM_DIM; ++i)
			for (j = 0; j <=i; ++j) {
				t->C[i][j] = (1 - t->ccov) * t->C[i][j] + t->ccov * (1./t->mucov) * (t->rgpc[i] * t->rgpc[j] + (1-hsig)*t->ccumcov*(2.-t->ccumcov) * t->C[i][j]);
			for (k = 0; k < t->mu; ++k) { /* additional rank mu update */
				t->C[i][j] += ccovmu * t->weights[k] * (parents[k]->\GENOME_NAME[i] - t->rgxold[i]) * (parents[k]->\GENOME_NAME[j] - t->rgxold[j]) / sigmasquare;
			}
		}

	}
}

void cmaes_exit(CMA *t)
{
	int i;
	free( t->rgpc);
	free( t->rgps);
	free( t->rgdTmp);
	free( t->rgBDz);
	free( t->rgxmean);
	free( t->rgxold); 
	free( t->rgD);
	for (i = 0; i < \PROBLEM_DIM; ++i) {
		free( t->C[i]);
		free( t->B[i]);
	}
	free( t->C);
	free( t->B);
} /* cmaes_exit() */

\INSERT_USER_FUNCTIONS

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION
\INSERT_GENERATION_FUNCTION
\INSERT_BEGIN_GENERATION_FUNCTION
\INSERT_END_GENERATION_FUNCTION
\INSERT_BOUND_CHECKING

void EASEAFinal(Population* pop){
  \INSERT_FINALIZATION_FCT_CALL
}

void EASEAInit(int argc, char** argv){
  \INSERT_INIT_FCT_CALL
}


using namespace std;

RandomGenerator* Individual::rg;

Individual::Individual(){
  \GENOME_CTOR 
  for(int i=0; i<\PROBLEM_DIM; i++ ) {
	this->\GENOME_NAME[i] = 0.5 + (cma.sigma * cma.rgD[i] * alea_Gauss(&cma.alea));
  }
  valid = false;
}


Individual::~Individual(){
  \GENOME_DTOR
}


float Individual::evaluate(){
  if(valid)
    return fitness;
  else{
    valid = true;
    \INSERT_EVALUATOR
  } 
}

Individual::Individual(const Individual& genome){

  // ********************
  // Problem specific part
  \COPY_CTOR
  
  // ********************
  // Generic part
  this->valid = genome.valid;
  this->fitness = genome.fitness;
}


Individual* Individual::crossover(Individual** ps){
  // ********************
  // Generic part  
  Individual parent1(*this);
  Individual parent2(*ps[0]);
  Individual child1(*this);

  //DEBUG_PRT("Xover");
/*   cout << "p1 : " << parent1 << endl; */
/*   cout << "p2 : " << parent2 << endl; */

  // ********************
  // Problem specific part
  for (int i = 0; i < \PROBLEM_DIM; ++i)
	cma.rgdTmp[i] = cma.rgD[i] * alea_Gauss(&cma.alea);

    child1.valid = false;
/*   cout << "child1 : " << child1 << endl; */
  return new Individual(child1);
}


void Individual::printOn(std::ostream& os) const{
  \INSERT_DISPLAY
}

std::ostream& operator << (std::ostream& O, const Individual& B) 
{ 
  // ********************
  // Problem specific part
  O << "\nIndividual : "<< std::endl;
  O << "\t\t\t";
  B.printOn(O);
    
  if( B.valid ) O << "\t\t\tfitness : " << B.fitness;
  else O << "fitness is not yet computed" << std::endl;
  return O; 
} 


size_t Individual::mutate( float pMutationPerGene ){
  this->valid=false;


  // ********************
  // Problem specific part
  double sum;
  for (int i = 0; i < \PROBLEM_DIM; ++i) {
	sum = 0.;
	for (int j = 0; j < \PROBLEM_DIM; ++j)
		sum += cma.B[i][j] * cma.rgdTmp[j];
	this->\GENOME_NAME[i] = cma.rgxmean[i] + cma.sigma * sum;
  }
  nbEnfants++;
  return 0;  
}

/* ****************************************
   EvolutionaryAlgorithm class
****************************************/

/* /\** */
/*    @DEPRECATED This contructor will be deleted. It was for test only, because it */
/*    is too much constrained (default selection/replacement operator) */
/*  *\/ */
/* EvolutionaryAlgorithm::EvolutionaryAlgorithm( size_t parentPopulationSize, */
/* 					      size_t offspringPopulationSize, */
/* 					      float selectionPressure, float replacementPressure, */
/* 					      float pCrossover, float pMutation,  */
/* 					      float pMutationPerGene){ */
/*   RandomGenerator* rg = globalRandomGenerator; */


/*   SelectionOperator* so = new MaxTournament(rg); */
/*   SelectionOperator* ro = new MaxTournament(rg); */
  
/*   Individual::initRandomGenerator(rg); */
/*   Population::initPopulation(so,ro,selectionPressure,replacementPressure); */
  
/*   this->population = new Population(parentPopulationSize,offspringPopulationSize, */
/* 				    pCrossover,pMutation,pMutationPerGene,rg); */

/*   this->currentGeneration = 0; */

/*   this->reduceParents = 0; */
/*   this->reduceOffsprings = 0; */


/* } */

EvolutionaryAlgorithm::EvolutionaryAlgorithm( size_t parentPopulationSize,
					      size_t offspringPopulationSize,
					      float selectionPressure, float replacementPressure, float parentReductionPressure, float offspringReductionPressure,
					      SelectionOperator* selectionOperator, SelectionOperator* replacementOperator,
					      SelectionOperator* parentReductionOperator, SelectionOperator* offspringReductionOperator,
					      float pCrossover, float pMutation, 
					      float pMutationPerGene, string& outputfile, string& inputfile){

  RandomGenerator* rg = globalRandomGenerator;

  SelectionOperator* so = selectionOperator;
  SelectionOperator* ro = replacementOperator;
  
  Individual::initRandomGenerator(rg);
  Population::initPopulation(so,ro,parentReductionOperator,offspringReductionOperator,selectionPressure,replacementPressure,parentReductionPressure,offspringReductionPressure);
  
  this->population = new Population(parentPopulationSize,offspringPopulationSize,
				    pCrossover,pMutation,pMutationPerGene,rg);

  this->currentGeneration = 0;

  this->reduceParents = 0;
  this->reduceOffsprings = 0;

  if( outputfile.length() )
    this->outputfile = new string(outputfile);
  else
    this->outputfile = NULL;

  if( inputfile.length() )
    this->inputfile = new std::string(inputfile);
  else
    this->inputfile = NULL;
  


}

void EvolutionaryAlgorithm::addStoppingCriterion(StoppingCriterion* sc){
  this->stoppingCriteria.push_back(sc);
}

void EvolutionaryAlgorithm::runEvolutionaryLoop(){
  std::vector<Individual*> tmpVect;

  nbEnfants=0;
  cma.gen=0;
  CMA_init(&cma, setVariable("nbOffspring",\OFF_SIZE), setVariable("popSize",\POP_SIZE));

  std::cout << "Parent's population initializing "<< std::endl;
  this->population->initializeParentPopulation();  
  std::cout << *population << std::endl;

  struct timeval begin;
  gettimeofday(&begin,NULL);

  //initialise mean;
  for (int i = 0; i < \PROBLEM_DIM; ++i){
			cma.rgxmean[i] = 0.5 + (cma.sigma * cma.rgD[i] * alea_Gauss(&cma.alea));
    		cma.rgxold[i] = cma.rgxmean[i];
  }
  
  while( this->allCriteria() == false ){    
    cmaes_UpdateEigensystem(&cma, 0);
    TestMinStdDevs(&cma);
    \INSERT_BEGINNING_GEN_FCT_CALL

    population->produceOffspringPopulation();
    \INSERT_BOUND_CHECKING_FCT_CALL
    population->evaluateOffspringPopulation();

     \INSERT_END_GEN_FCT_CALL

#if \IS_PARENT_REDUCTION
      population->reduceParentPopulation(\SURV_PAR_SIZE);
#endif
    

#if \IS_OFFSPRING_REDUCTION
      population->reduceOffspringPopulation(\SURV_OFF_SIZE);
#endif
    
    population->reduceTotalPopulation();
     
    \INSERT_GEN_FCT_CALL    

    int i, j, iNk, hsig;
	double sum; 
	double psxps;
	population->sortParentPopulation();
	Individual **popparents = population->parents;
	if (popparents[0]->fitness == popparents[(int)cma.mu/2]->fitness){
		cma.sigma *= exp(0.2+cma.cs/cma.damps);
		printf("Warning: sigma increased due to equal function values\n");
		printf("   Reconsider the formulation of the objective function\n");
	}
	for (i = 0; i < \PROBLEM_DIM; ++i) {
		cma.rgxold[i] = cma.rgxmean[i]; 
		cma.rgxmean[i] = 0.;
		for (iNk = 0; iNk < cma.mu; ++iNk) 
			cma.rgxmean[i] += cma.weights[iNk] * popparents[iNk]->\GENOME_NAME[i];
		cma.rgBDz[i] = sqrt(cma.mueff)*(cma.rgxmean[i] - cma.rgxold[i])/cma.sigma; 
	}
	/* calculate z := D^(-1) * B^(-1) * rgBDz into rgdTmp */
	for (i = 0; i < \PROBLEM_DIM; ++i) {
		sum = 0.;
		for (j = 0; j < \PROBLEM_DIM; ++j)
			sum += cma.B[j][i] * cma.rgBDz[j];
		cma.rgdTmp[i] = sum / cma.rgD[i];
	}
	/* cumulation for sigma (ps) using B*z */
	for (i = 0; i < \PROBLEM_DIM; ++i) {
		sum = 0.;
		for (j = 0; j < \PROBLEM_DIM; ++j)
			sum += cma.B[i][j] * cma.rgdTmp[j];
		cma.rgps[i] = (1. - cma.cs) * cma.rgps[i] + 
				sqrt(cma.cs * (2. - cma.cs)) * sum;
	}
	/* calculate norm(ps)^2 */
	psxps = 0.;
	for (i = 0; i < \PROBLEM_DIM; ++i)
		psxps += cma.rgps[i] * cma.rgps[i];
	/* cumulation for covariance matrix (pc) using B*D*z~N(0,C) */
	hsig = (sqrt(psxps) / sqrt(1. - pow(1.-cma.cs, 2*cma.gen)) / cma.chiN) < (1.4 + 2./(\PROBLEM_DIM+1));
	for (i = 0; i < \PROBLEM_DIM; ++i) {
		cma.rgpc[i] = (1. - cma.ccumcov) * cma.rgpc[i] + hsig * sqrt(cma.ccumcov * (2. - cma.ccumcov)) * cma.rgBDz[i];
	}
	/* stop initial phase */
	if (cma.flgIniphase && cma.gen > MIN(1/cma.cs, 1+\PROBLEM_DIM/cma.mucov)) 
	{
		if (psxps / cma.damps / (1.-pow((1. - cma.cs), cma.gen)) < \PROBLEM_DIM * 1.05) 
			cma.flgIniphase = 0;
	}
	Adapt_C2(&cma, hsig, popparents);
	/* update of sigma */
	cma.sigma *= exp(((sqrt(psxps)/cma.chiN)-1.)*cma.cs/cma.damps);
	cma.gen++;
	nbEnfants=0;    

    showPopulationStats(begin);
    currentGeneration += 1;
  }  
  population->sortParentPopulation();
  //std::cout << *population << std::endl;
  std::cout << "Generation : " << currentGeneration << std::endl;
}


void EvolutionaryAlgorithm::showPopulationStats(struct timeval beginTime){

  float currentAverageFitness=0.0;
  float currentSTDEV=0.0;

  //Calcul de la moyenne et de l'ecart type
  population->Best=population->parents[0];

  for(size_t i=0; i<population->parentPopulationSize; i++){
    currentAverageFitness+=population->parents[i]->getFitness();
#if \MINIMAXI
    if(population->parents[i]->getFitness()>population->Best->getFitness())
#else
    if(population->parents[i]->getFitness()<population->Best->getFitness())
#endif
      population->Best=population->parents[i];
  }

  currentAverageFitness/=population->parentPopulationSize;

  for(size_t i=0; i<population->parentPopulationSize; i++){
    currentSTDEV+=(population->parents[i]->getFitness()-currentAverageFitness)*(population->parents[i]->getFitness()-currentAverageFitness);
  }
  currentSTDEV/=population->parentPopulationSize;
  currentSTDEV=sqrt(currentSTDEV);
  
  //Affichage
  if(currentGeneration==0)
    printf("GEN\tTIME\t\tEVAL\tBEST\t\tAVG\t\tSTDEV\n\n");

    //  assert( currentSTDEV == currentSTDEV );
  
  struct timeval end, res;
  gettimeofday(&end,0);
  timersub(&end,&beginTime,&res);
  printf("%lu\t%d.%06d\t%lu\t%.15e\t%.15e\t%.15e\n",currentGeneration,res.tv_sec,res.tv_usec,population->currentEvaluationNb,
	 population->Best->getFitness(),currentAverageFitness,currentSTDEV);
}

bool EvolutionaryAlgorithm::allCriteria(){

  for( size_t i=0 ; i<stoppingCriteria.size(); i++ ){
    if( stoppingCriteria.at(i)->reached() ){
      std::cout << "Stopping criterion reached : " << i << std::endl;
      return true;
    }
  }
  return false;
}



\START_CUDA_USER_CLASSES_H_TPL
#include <iostream>
#include <ostream>
#include <sstream>
using namespace std;
\INSERT_USER_CLASSES

\START_CUDA_GENOME_H_TPL
#ifndef __INDIVIDUAL
#define __INDIVIDUAL
#include "EASEATools.hpp"
#include <iostream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


\INSERT_USER_CLASSES_DEFINITIONS

void EASEAInit(int argc, char *argv[]);
void EASEAFinal(Population* population);
void EASEAFinalization(Population* population);

class Individual{

 public: // in AESAE the genome is public (for user functions,...)
  \INSERT_GENOME
  bool valid;
  float fitness;
  static RandomGenerator* rg;

 public:
  Individual();
  Individual(const Individual& indiv);
  virtual ~Individual();
  float evaluate();
  static size_t getCrossoverArrity(){ return 2; }
  float getFitness(){ return this->fitness; }
  Individual* crossover(Individual** p2);
  void printOn(std::ostream& O) const;
  
  size_t mutate(float pMutationPerGene);

  friend std::ostream& operator << (std::ostream& O, const Individual& B) ;
  static void initRandomGenerator(RandomGenerator* rg){ Individual::rg = rg;}

 private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive& ar, const unsigned int version){

    ar & fitness;
    DEBUG_PRT("(de)serialization of %f fitness",fitness);
    ar & valid;
    DEBUG_PRT("(de)serialization of %d valid",valid);
    \GENOME_SERIAL
  }

  
};


/* ****************************************
   EvolutionaryAlgorithm class
****************************************/
class EvolutionaryAlgorithm{
public:
/*   EvolutionaryAlgorithm(  size_t parentPopulationSize, size_t offspringPopulationSize, */
/* 			  float selectionPressure, float replacementPressure,  */
/* 			  float pCrossover, float pMutation, float pMutationPerGene); */
  
  EvolutionaryAlgorithm( size_t parentPopulationSize,
			 size_t offspringPopulationSize,
			 float selectionPressure, float replacementPressure, float parentReductionPressure, float offspringReductionPressure,
			 SelectionOperator* selectionOperator, SelectionOperator* replacementOperator,
			 SelectionOperator* parentReductionOperator, SelectionOperator* offspringReductionOperator,
			 float pCrossover, float pMutation, 
			 float pMutationPerGene, std::string& outputfile, std::string& inputfile);

  size_t* getCurrentGenerationPtr(){ return &currentGeneration;}
  void addStoppingCriterion(StoppingCriterion* sc);
  void runEvolutionaryLoop();
  bool allCriteria();
  Population* getPopulation(){ return population;}
  size_t getCurrentGeneration() { return currentGeneration;}
public:
  size_t currentGeneration;
  Population* population;
  size_t reduceParents;
  size_t reduceOffsprings;
  //void showPopulationStats();
  void showPopulationStats(struct timeval beginTime);
  

  std::vector<StoppingCriterion*> stoppingCriteria;

  std::string* outputfile;
  std::string* inputfile;
};


#endif


\START_CUDA_TOOLS_CPP_TPL/* ****************************************
			    
   RandomGenerator class

****************************************/
#include "EASEATools.hpp"
#include "EASEAIndividual.hpp"
#include <stdio.h>
#include <iostream>
#include <values.h>
#include <string.h>
#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>


RandomGenerator::RandomGenerator(unsigned int seed){
  srand(seed);
}

int RandomGenerator::randInt(){
  return rand();
}

bool RandomGenerator::tossCoin(){

  int rVal = rand();
  if( rVal >=(RAND_MAX/2))
    return true;
  else return false;
}


bool RandomGenerator::tossCoin(float bias){

  int rVal = rand();
  if( rVal <=(RAND_MAX*bias) )
    return true;
  else return false;
}



int RandomGenerator::randInt(int min, int max){

  int rValue = (((float)rand()/RAND_MAX))*(max-min);
  //DEBUG_PRT("Int Random Value : %d",min+rValue);
  return rValue+min;

}

int RandomGenerator::random(int min, int max){
  return randInt(min,max);
}

float RandomGenerator::randFloat(float min, float max){
  float rValue = (((float)rand()/RAND_MAX))*(max-min);
  //DEBUG_PRT("Float Random Value : %f",min+rValue);
  return rValue+min;
}

float RandomGenerator::random(float min, float max){
  return randFloat(min,max);
}

double RandomGenerator::random(double min, double max){
  return randFloat(min,max);
}


int RandomGenerator::getRandomIntMax(int max){
  double r = rand();
  r = r / RAND_MAX;
  r = r * max;
  return r;
}


/* ****************************************
   Tournament class (min and max)
****************************************/
void MaxTournament::initialize(Individual** population, float selectionPressure, size_t populationSize) {
  SelectionOperator::initialize(population,selectionPressure,populationSize);
}


float MaxTournament::getExtremum(){
  return -FLT_MAX;
}

size_t MaxTournament::selectNext(size_t populationSize){
  size_t bestIndex = 0;
  float bestFitness = -FLT_MAX;

  //std::cout << "MaxTournament selection " ;
  if( currentSelectionPressure >= 2 ){
    for( size_t i = 0 ; i<currentSelectionPressure ; i++ ){
      size_t selectedIndex = rg->getRandomIntMax(populationSize);
      //std::cout << selectedIndex << " ";
      float currentFitness = population[selectedIndex]->getFitness();
      
      if( bestFitness < currentFitness ){
	bestIndex = selectedIndex;
	bestFitness = currentFitness;
      }

    }
  }
  else if( currentSelectionPressure <= 1 && currentSelectionPressure > 0 ){
    size_t i1 = rg->getRandomIntMax(populationSize);
    size_t i2 = rg->getRandomIntMax(populationSize);

    if( rg->tossCoin(currentSelectionPressure) ){
      if( population[i1]->getFitness() > population[i2]->getFitness() ){
	bestIndex = i1;
      }
    }
    else{
      if( population[i1]->getFitness() > population[i2]->getFitness() ){
	bestIndex = i2;
      }
    }
  }
  else{
    std::cerr << " MaxTournament selection operator doesn't handle selection pressure : " 
	      << currentSelectionPressure << std::endl;
  }
  //std::cout << std::endl;
  return bestIndex;
}


void MinTournament::initialize(Individual** population, float selectionPressure, size_t populationSize) {
  SelectionOperator::initialize(population,selectionPressure,populationSize);
}

float MinTournament::getExtremum(){
  return FLT_MAX;
}


size_t MinTournament::selectNext(size_t populationSize){
  size_t bestIndex = 0;
  float bestFitness = FLT_MAX;

  //std::cout << "MinTournament selection " ;
  if( currentSelectionPressure >= 2 ){
    for( size_t i = 0 ; i<currentSelectionPressure ; i++ ){
      size_t selectedIndex = rg->getRandomIntMax(populationSize);
      //std::cout << selectedIndex << " ";
      float currentFitness = population[selectedIndex]->getFitness();
      
      if( bestFitness > currentFitness ){
	bestIndex = selectedIndex;
	bestFitness = currentFitness;
      }

    }
  }
  else if( currentSelectionPressure <= 1 && currentSelectionPressure > 0 ){
    size_t i1 = rg->getRandomIntMax(populationSize);
    size_t i2 = rg->getRandomIntMax(populationSize);

    if( rg->tossCoin(currentSelectionPressure) ){
      if( population[i1]->getFitness() < population[i2]->getFitness() ){
	bestIndex = i1;
      }
    }
    else{
      if( population[i1]->getFitness() < population[i2]->getFitness() ){
	bestIndex = i2;
      }
    }
  }
  else{
    std::cerr << " MinTournament selection operator doesn't handle selection pressure : " 
	      << currentSelectionPressure << std::endl;
  }

  //std::cout << std::endl;
  return bestIndex;
}


/* ****************************************
   SelectionOperator class
****************************************/
void SelectionOperator::initialize(Individual** population, float selectionPressure, size_t populationSize){
  this->population = population;
  this->currentSelectionPressure = selectionPressure;
}

size_t SelectionOperator::selectNext(size_t populationSize){ return 0; }


/* ****************************************
   GenerationalCriterion class
****************************************/
GenerationalCriterion::GenerationalCriterion(EvolutionaryAlgorithm* ea, size_t generationalLimit){
  this->currentGenerationPtr = ea->getCurrentGenerationPtr();
  this->generationalLimit = generationalLimit;
}

bool GenerationalCriterion::reached(){
  if( generationalLimit <= *currentGenerationPtr ){
    std::cout << "Current generation " << *currentGenerationPtr << " Generational limit : " <<
      generationalLimit << std::endl;
    return true;
  }
  else return false;
}

size_t* GenerationalCriterion::getGenerationalLimit(){
	return &this->generationalLimit;
}


/* ****************************************
   Population class
****************************************/
SelectionOperator* Population::selectionOperator;
SelectionOperator* Population::replacementOperator;
SelectionOperator* Population::parentReductionOperator;
SelectionOperator* Population::offspringReductionOperator;


float Population::selectionPressure;
float Population::replacementPressure;
float Population::parentReductionPressure;
float Population::offspringReductionPressure;



Population::Population(){
}

Population::Population(size_t parentPopulationSize, size_t offspringPopulationSize,
		       float pCrossover, float pMutation, float pMutationPerGene,
		       RandomGenerator* rg){
  
  this->parents     = new Individual*[parentPopulationSize];
  this->offsprings  = new Individual*[offspringPopulationSize];
  
  this->parentPopulationSize     = parentPopulationSize;
  this->offspringPopulationSize  = offspringPopulationSize;
    
  this->actualParentPopulationSize    = 0;
  this->actualOffspringPopulationSize = 0;

  this->pCrossover       = pCrossover;
  this->pMutation        = pMutation;
  this->pMutationPerGene = pMutationPerGene;

  this->rg = rg;

  this->currentEvaluationNb = 0;
}

void Population::syncInVector(){
  for( size_t i = 0 ; i<actualParentPopulationSize ; i++ ){
    parents[i] = pop_vect.at(i);
  }
}

void Population::syncOutVector(){
  pop_vect.clear();
  for( size_t i = 0 ; i<actualParentPopulationSize ; i++ ){
    pop_vect.push_back(parents[i]);
  }
  DEBUG_PRT("Size of outVector",pop_vect.size());
}

Population::~Population(){
  for( size_t i=0 ; i<actualOffspringPopulationSize ; i++ ) delete(offsprings[i]);
  for( size_t i=0 ; i<actualParentPopulationSize ; i++ )    delete(parents[i]);

  delete[](this->parents);
  delete[](this->offsprings);
}

void Population::initPopulation(SelectionOperator* selectionOperator, 
				SelectionOperator* replacementOperator,
				SelectionOperator* parentReductionOperator,
				SelectionOperator* offspringReductionOperator,
				float selectionPressure, float replacementPressure,
				float parentReductionPressure, float offspringReductionPressure){
  Population::selectionOperator   = selectionOperator;
  Population::replacementOperator = replacementOperator;
  Population::parentReductionOperator = parentReductionOperator;
  Population::offspringReductionOperator = offspringReductionOperator;

  Population::selectionPressure   = selectionPressure;
  Population::replacementPressure = replacementPressure;
  Population::parentReductionPressure = parentReductionPressure;
  Population::offspringReductionPressure = offspringReductionPressure;

}


void Population::initializeParentPopulation(){

  DEBUG_PRT("Creation of %d/%d parents (other could have been loaded from input file)",parentPopulationSize-actualParentPopulationSize,parentPopulationSize);
  for( size_t i=actualParentPopulationSize ; i<parentPopulationSize ; i++ )
    parents[i] = new Individual();

  actualParentPopulationSize = parentPopulationSize;
  actualOffspringPopulationSize = 0;
  
  evaluateParentPopulation();
}


void Population::evaluatePopulation(Individual** population, size_t populationSize){
  for( size_t i=0 ; i < populationSize ; i++ )
    population[i]->evaluate();
  currentEvaluationNb += populationSize;
}


void Population::evaluateParentPopulation(){
  evaluatePopulation(parents,parentPopulationSize);
}


void Population::evaluateOffspringPopulation(){
  evaluatePopulation(offsprings,offspringPopulationSize);
}


/**
   Reduit la population population de taille populationSize 
   a une population reducedPopulation de taille obSize.
   reducedPopulation doit etre alloue a obSize.

   Ici on pourrait avoir le best fitness de la prochaine population de parents.
   

 */
void Population::reducePopulation(Individual** population, size_t populationSize,
					  Individual** reducedPopulation, size_t obSize,
					  SelectionOperator* replacementOperator){
  

  replacementOperator->initialize(population,replacementPressure,populationSize);

  for( size_t i=0 ; i<obSize ; i++ ){
    
    // select an individual and add it to the reduced population
    size_t selectedIndex = replacementOperator->selectNext(populationSize - i);
    // std::cout << "Selected " << selectedIndex << "/" << populationSize
    // 	      << " replaced by : " << populationSize-(i+1)<< std::endl;
    reducedPopulation[i] = population[selectedIndex];
    
    // erase it to the std population by swapping last individual end current
    population[selectedIndex] = population[populationSize-(i+1)];
    //population[populationSize-(i+1)] = NULL;
  }

  //return reducedPopulation;
}


Individual** Population::reduceParentPopulation(size_t obSize){
  Individual** nextGeneration = new Individual*[obSize];

  reducePopulation(parents,actualParentPopulationSize,nextGeneration,obSize,
		   Population::replacementOperator);

  // free no longer needed individuals
  for( size_t i=0 ; i<actualParentPopulationSize-obSize ; i++ )
    delete(parents[i]);
  delete[](parents);

  this->actualParentPopulationSize = obSize;
  parents = nextGeneration;
  

  return nextGeneration;
}



Individual** Population::reduceOffspringPopulation(size_t obSize){
  // this array has offspringPopulationSize because it will be used as offspring population in
  // the next generation
  Individual** nextGeneration = new Individual*[offspringPopulationSize]; 
  

  reducePopulation(offsprings,actualOffspringPopulationSize,nextGeneration,obSize,
		   Population::replacementOperator);

  // free no longer needed individuals
  for( size_t i=0 ; i<actualOffspringPopulationSize-obSize ; i++ )
    delete(offsprings[i]);
  delete[](offsprings);

  this->actualOffspringPopulationSize = obSize;
  offsprings = nextGeneration;
  return nextGeneration;
}


static int individualCompare(const void* p1, const void* p2){
  Individual** p1_i = (Individual**)p1;
  Individual** p2_i = (Individual**)p2;

  return p1_i[0]->getFitness() > p2_i[0]->getFitness();
}

static int individualRCompare(const void* p1, const void* p2){
  Individual** p1_i = (Individual**)p1;
  Individual** p2_i = (Individual**)p2;

  return p1_i[0]->getFitness() < p2_i[0]->getFitness();
}


void Population::sortPopulation(Individual** population, size_t populationSize){
  qsort(population,populationSize,sizeof(Individual*),individualCompare);
}

void Population::sortRPopulation(Individual** population, size_t populationSize){
  qsort(population,populationSize,sizeof(Individual*),individualRCompare);
}


/**
   Reduit les populations en faisant l'operation de remplacement.

   @TODO : on aurait voulu eviter la recopie des deux populations en une seule
   mais cela semble incompatible avec SelectionOperator (notamment l'operation 
   d'initialisation.
*/
void Population::reduceTotalPopulation(){

  Individual** nextGeneration = new Individual*[parentPopulationSize];

#if ((\ELITE_SIZE!=0) && (\ELITISM==true))                     // If there is elitism and it is strong
  Population::elitism(\ELITE_SIZE,parents,actualParentPopulationSize,
		      nextGeneration,parentPopulationSize); // do the elitism on the parent population only
  actualParentPopulationSize -= \ELITE_SIZE;                // decrement the parent population size
#endif

  size_t actualGlobalSize = actualParentPopulationSize+actualOffspringPopulationSize;
  Individual** globalPopulation = new Individual*[actualGlobalSize]();


  memcpy(globalPopulation,parents,sizeof(Individual*)*actualParentPopulationSize);
  memcpy(globalPopulation+actualParentPopulationSize,offsprings,
   	 sizeof(Individual*)*actualOffspringPopulationSize);
  replacementOperator->initialize(globalPopulation, replacementPressure,actualGlobalSize);

#if ((\ELITE_SIZE!=0) && (\ELITISM==false))                    // If there is elitism and it is weak
  Population::elitism(\ELITE_SIZE,globalPopulation,actualGlobalSize,
		      nextGeneration,parentPopulationSize); // do the elitism on the global (already merged) population
  actualGlobalSize -= \ELITE_SIZE;                // decrement the parent population size
#endif

    
  Population::reducePopulation(globalPopulation,actualGlobalSize,\ELITE_SIZE+nextGeneration,
			       parentPopulationSize-\ELITE_SIZE,replacementOperator);

  for( size_t i=0 ; i<((int)actualGlobalSize+\ELITE_SIZE)-(int)parentPopulationSize ; i++ )
    delete(globalPopulation[i]);
    
  delete[](parents);
  delete[](globalPopulation);

  actualParentPopulationSize = parentPopulationSize;
  actualOffspringPopulationSize = 0;
  parents = nextGeneration;
  
}


void Population::produceOffspringPopulation(){

  size_t crossoverArrity = Individual::getCrossoverArrity();
  Individual* p1;
  Individual** ps = new Individual*[crossoverArrity]();
  Individual* child;

  selectionOperator->initialize(parents,selectionPressure,actualParentPopulationSize);

  for( size_t i=0 ; i<offspringPopulationSize ; i++ ){
    size_t index = selectionOperator->selectNext(parentPopulationSize);
    p1 = parents[index];
    
    if( rg->tossCoin(pCrossover) ){
      for( size_t j=0 ; j<crossoverArrity-1 ; j++ ){
	index = selectionOperator->selectNext(parentPopulationSize);
	ps[j] = parents[index];
      }
      child = p1->crossover(ps);
    }
    else child = new Individual(*parents[index]);

    if( rg->tossCoin(pMutation) ){
      child->mutate(pMutationPerGene);
    }
    
    offsprings[actualOffspringPopulationSize++] = child;
  }
  delete[](ps);
  }




/**
   Here we save elit individuals to the replacement
   
   @ARG elitismSize the number of individuals save by elitism
   @ARG population the population where the individuals are save
   @ARG populationSize the size of the population
   @ARG outPopulation the output population, this must be allocated with size greather than elitism
   @ARG outPopulationSize the size of the output population
   
*/
void Population::elitism(size_t elitismSize, Individual** population, size_t populationSize, 
			 Individual** outPopulation, size_t outPopulationSize){
  
  float bestFitness = population[0]->getFitness();
  size_t bestIndividual = 0;
  
  if( elitismSize >= 5 )DEBUG_PRT("Warning, elitism has O(n) complexity, elitismSize is maybe too big (%d)",elitismSize);
  
  
  for(size_t i = 0 ; i<elitismSize ; i++ ){
    bestFitness = replacementOperator->getExtremum();
    bestIndividual = 0;
    for( size_t j=0 ; j<populationSize-i ; j++ ){
#if \MINIMAXI
      if( bestFitness < population[j]->getFitness() ){
#else
      if( bestFitness > population[j]->getFitness() ){
#endif
	bestFitness = population[j]->getFitness();
	bestIndividual = j;
      }
    }
    outPopulation[i] = population[bestIndividual];
    population[bestIndividual] = population[populationSize-(i+1)];
    population[populationSize-(i+1)] = NULL;
  }
}
 



std::ostream& operator << (std::ostream& O, const Population& B) 
{ 
  
  size_t offspringPopulationSize = B.offspringPopulationSize;
  size_t realOffspringPopulationSize = B.actualOffspringPopulationSize;

  size_t parentPopulationSize = B.parentPopulationSize;
  size_t realParentPopulationSize = B.actualParentPopulationSize;


  O << "Population : "<< std::endl;
  O << "\t Parents size : "<< realParentPopulationSize << "/" << 
    parentPopulationSize << std::endl;
  
  for( size_t i=0 ; i<realParentPopulationSize ; i++){
    O << "\t\t" << *B.parents[i] ;
  } 

  O << "\t Offspring size : "<< realOffspringPopulationSize << "/" << 
    offspringPopulationSize << std::endl;
  for( size_t i=0 ; i<realOffspringPopulationSize ; i++){
    O << "\t\t" << *B.offsprings[i] << std::endl;
  }  
  return O; 
} 



void MaxDeterministic::initialize(Individual** population, float selectionPressure,size_t populationSize){
  SelectionOperator::initialize(population,selectionPressure,populationSize);
  Population::sortPopulation(population,populationSize);
  populationSize = populationSize;
}


size_t MaxDeterministic::selectNext(size_t populationSize){
  return populationSize-1;
}

float MaxDeterministic::getExtremum(){
  return -FLT_MAX;
}



void MinDeterministic::initialize(Individual** population, float selectionPressure,size_t populationSize){
  SelectionOperator::initialize(population,selectionPressure,populationSize);
  Population::sortRPopulation(population,populationSize);
  populationSize = populationSize;
}


size_t MinDeterministic::selectNext(size_t populationSize){
  return populationSize-1;
}

float MinDeterministic::getExtremum(){
  return FLT_MAX;
}

MaxRandom::MaxRandom(RandomGenerator* globalRandomGenerator){
  rg = globalRandomGenerator;
}

void MaxRandom::initialize(Individual** population, float selectionPressure, size_t populationSize){
  SelectionOperator::initialize(population,selectionPressure,populationSize);
}

size_t MaxRandom::selectNext(size_t populationSize){
  return rg->random(0,populationSize-1);
}

float MaxRandom::getExtremum(){
  return -FLT_MAX;
}

MinRandom::MinRandom(RandomGenerator* globalRandomGenerator){
  rg = globalRandomGenerator;
}

void MinRandom::initialize(Individual** population, float selectionPressure, size_t populationSize){
  SelectionOperator::initialize(population,selectionPressure,populationSize);
}

size_t MinRandom::selectNext(size_t populationSize){
  return rg->random(0,populationSize-1);
}

float MinRandom::getExtremum(){
  return -FLT_MAX;
}

namespace po = boost::program_options;


po::variables_map vm;
po::variables_map vm_file;

using namespace std;

string setVariable(string argumentName, string defaultValue, po::variables_map vm, po::variables_map vm_file){
  string ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<string>();
    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<string>();
    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
    cout << argumentName << " is not declared, default value is "<< ret<< endl;
  }
  return ret;
}

int setVariable(string argumentName, int defaultValue, po::variables_map vm, po::variables_map vm_file ){
  int ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<int>();
    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<int>();
    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
    cout << argumentName << " is not declared, default value is "<< ret<< endl;
  }
  return ret;
}


int loadParametersFile(const string& filename, char*** outputContainer){

  FILE* paramFile = fopen(filename.c_str(),"r");
  char buffer[512];
  vector<char*> tmpContainer;
  
  char* padding = (char*)malloc(sizeof(char));
  padding[0] = 0;

  tmpContainer.push_back(padding);
  
  while( fgets(buffer,512,paramFile)){
    for( size_t i=0 ; i<512 ; i++ )
      if( buffer[i] == '#' || buffer[i] == '\n' || buffer[i] == '\0' || buffer[i]==' '){
	buffer[i] = '\0';
	break;
      } 
    int str_len;
    if( (str_len = strlen(buffer)) ){
      cout << "line : " <<buffer << endl;
      char* nLine = (char*)malloc(sizeof(char)*(str_len+1));
      strcpy(nLine,buffer);
      tmpContainer.push_back(nLine);
    }    
  }

  (*outputContainer) = (char**)malloc(sizeof(char*)*tmpContainer.size());
 
  for ( size_t i=0 ; i<tmpContainer.size(); i++)
    (*outputContainer)[i] = tmpContainer.at(i);

  fclose(paramFile);
  return tmpContainer.size();
}


void parseArguments(const char* parametersFileName, int ac, char** av, 
		    po::variables_map& vm, po::variables_map& vm_file){

  char** argv;
  int argc = loadParametersFile(parametersFileName,&argv);
  
  po::options_description desc("Allowed options ");
  desc.add_options()
    ("help", "produce help message")
    ("compression", po::value<int>(), "set compression level")
    ("seed", po::value<int>(), "set the global seed of the pseudo random generator")
    ("popSize",po::value<int>(),"set the population size")
    ("nbOffspring",po::value<int>(),"set the offspring population size")
    ("elite",po::value<int>(),"Nb of elite parents (absolute)")
    ("eliteType",po::value<int>(),"Strong (1) or weak (1)")
    ("nbGen",po::value<int>(),"Set the number of generation")
    ("surviveParents",po::value<int>()," Nb of surviving parents (absolute)")
    ("surviveOffsprings",po::value<int>()," Nb of surviving offsprings (absolute)")
    ("outputfile",po::value<string>(),"Set an output file for the final population (default : none)")
    ("inputfile",po::value<string>(),"Set an input file for the initial population (default : none)")
    ("u1",po::value<string>(),"User defined parameter 1")
    ("u2",po::value<string>(),"User defined parameter 2")
    ("u3",po::value<string>(),"User defined parameter 3")
    ("u4",po::value<string>(),"User defined parameter 4")
    ;
    
  try{
    po::store(po::parse_command_line(ac, av, desc,0), vm);
    po::store(po::parse_command_line(argc, argv, desc,0), vm_file);
  }
  catch(po::unknown_option& e){
    cerr << "Unknown option  : " << e.what() << endl;    
    cout << desc << endl;
    exit(1);
  }
  
  po::notify(vm);    
  po::notify(vm_file);    

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }

  for( int i = 0 ; i<argc ; i++ )
    free(argv[i]);
  free(argv);
 
}

void parseArguments(const char* parametersFileName, int ac, char** av){
  parseArguments(parametersFileName,ac,av,vm,vm_file);
}


int setVariable(const string optionName, int defaultValue){
  return setVariable(optionName,defaultValue,vm,vm_file);
}

string setVariable(const string optionName, string defaultValue){
  return setVariable(optionName,defaultValue,vm,vm_file);
}








\START_CUDA_TOOLS_H_TPL/* ****************************************
   Some tools classes for algorithm
****************************************/
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <boost/archive/text_oarchive.hpp> //for serialization (dumping)
#include <boost/archive/text_iarchive.hpp> //for serialization (loading)
#include <boost/serialization/vector.hpp>

class EvolutionaryAlgorithm;
class Individual;
class Population;

extern size_t *EZ_NB_GEN;

#define EZ_MINIMIZE \MINIMAXI
#define EZ_MINIMISE \MINIMAXI
#define EZ_MAXIMIZE !\MINIMAXI
#define EZ_MAXIMISE !\MINIMAXI

#ifdef DEBUG
#define DEBUG_PRT(format, args...) fprintf (stdout,"***DBG***  %s-%d: "format"\n",__FILE__,__LINE__,##args)
#define DEBUG_YACC(format, args...) fprintf (stdout,"***DBG_YACC***  %s-%d: "format"\n",__FILE__,__LINE__,##args)
#else
#define DEBUG_PRT(format, args...) 
#define DEBUG_YACC(format, args...)
#endif


/* ****************************************
   StoppingCriterion class
****************************************/
#ifndef __EASEATOOLS
#define __EASEATOOLS
class StoppingCriterion {

public:
  virtual bool reached() = 0;

};


/* ****************************************
   GenerationalCriterion class
****************************************/
class GenerationalCriterion : public StoppingCriterion {
 private:
  size_t* currentGenerationPtr;
  size_t generationalLimit;
 public:
  virtual bool reached();
  GenerationalCriterion(EvolutionaryAlgorithm* ea, size_t generationalLimit);
  size_t* getGenerationalLimit(); 
};


/* ****************************************
   RandomGenerator class
****************************************/
class RandomGenerator{
public:
  RandomGenerator(unsigned int seed);
  int randInt();
  bool tossCoin();
  bool tossCoin(float bias);
  int randInt(int min, int max);
  int getRandomIntMax(int max);
  float randFloat(float min, float max);
  int random(int min, int max);
  float random(float min, float max);
  double random(double min, double max);

};



/* ****************************************
   Selection Operator class
****************************************/
class SelectionOperator{
public:
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  virtual float getExtremum() = 0 ;
protected:
  Individual** population;
  float currentSelectionPressure;
};


/* ****************************************
   Tournament classes (min and max)
****************************************/
class MaxTournament : public SelectionOperator{
public:
  MaxTournament(RandomGenerator* rg){ this->rg = rg; }
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  RandomGenerator* rg;
  
};



class MinTournament : public SelectionOperator{
public:
  MinTournament(RandomGenerator* rg){ this->rg = rg; }
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  RandomGenerator* rg;
  
};


class MaxDeterministic : public SelectionOperator{
 public:
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
};

class MinDeterministic : public SelectionOperator{
 public:
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;

};


class MaxRandom : public SelectionOperator{
 public:
  MaxRandom(RandomGenerator* globalRandomGenerator);
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
  RandomGenerator* rg;

};

class MinRandom : public SelectionOperator{
 public:
  MinRandom(RandomGenerator* globalRandomGenerator);
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
  RandomGenerator* rg;
};



class Population {
  
 public:
  
  float pCrossover;
  float pMutation;
  float pMutationPerGene;

  Individual* Best;
  
  Individual** parents;
  Individual** offsprings;

  size_t parentPopulationSize;
  size_t offspringPopulationSize;

  size_t actualParentPopulationSize;
  size_t actualOffspringPopulationSize;

  static SelectionOperator* selectionOperator;
  static SelectionOperator* replacementOperator;
  static SelectionOperator* parentReductionOperator;
  static SelectionOperator* offspringReductionOperator;

  size_t currentEvaluationNb;
  RandomGenerator* rg;
  std::vector<Individual*> pop_vect;

 public:
  Population();
  Population(size_t parentPopulationSize, size_t offspringPopulationSize, 
	     float pCrossover, float pMutation, float pMutationPerGene, RandomGenerator* rg);
  virtual ~Population();

  void initializeParentPopulation();  
  void evaluatePopulation(Individual** population, size_t populationSize);
  void evaluateParentPopulation();

  static void elitism(size_t elitismSize, Individual** population, size_t populationSize, Individual** outPopulation,
		      size_t outPopulationSize);

  void evaluateOffspringPopulation();
  Individual** reducePopulations(Individual** population, size_t populationSize,
			       Individual** reducedPopulation, size_t obSize);
  Individual** reduceParentPopulation(size_t obSize);
  Individual** reduceOffspringPopulation(size_t obSize);
  void reduceTotalPopulation();
  void evolve();

  static float selectionPressure;
  static float replacementPressure;
  static float parentReductionPressure;
  static float offspringReductionPressure;

  static void initPopulation(SelectionOperator* selectionOperator, 
			     SelectionOperator* replacementOperator,
			     SelectionOperator* parentReductionOperator,
			     SelectionOperator* offspringReductionOperator,
			     float selectionPressure, float replacementPressure,
			     float parentReductionPressure, float offspringReductionPressure);

  static void sortPopulation(Individual** population, size_t populationSize);

  static void sortRPopulation(Individual** population, size_t populationSize);


  void sortParentPopulation(){ Population::sortPopulation(parents,actualParentPopulationSize);}

  void produceOffspringPopulation();

  friend std::ostream& operator << (std::ostream& O, const Population& B);


  void setParentPopulation(Individual** population, size_t actualParentPopulationSize){ 
    this->parents = population;
    this->actualParentPopulationSize = actualParentPopulationSize;
  }

  static void reducePopulation(Individual** population, size_t populationSize,
				       Individual** reducedPopulation, size_t obSize,
				       SelectionOperator* replacementOperator);
  void syncOutVector();
  void syncInVector();

 private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive& ar, const unsigned int version){

    ar & actualParentPopulationSize;
    DEBUG_PRT("(de)serialization of %d parents",actualParentPopulationSize);
    ar & pop_vect;
    DEBUG_PRT("(de)serialization of %d offspring",actualOffspringPopulationSize);
  }
};

/* namespace boost{ */
/*   namespace serialization{ */
/*     template<class Archive> */
/*       void serialize(Archive & ar,std::vector<Individual*> population, const unsigned int version){ */
/*       ar & population; */
/*     } */
/*   } */
/* } */

void parseArguments(const char* parametersFileName, int ac, char** av);
int setVariable(const std::string optionName, int defaultValue);
std::string setVariable(const std::string optionName, std::string defaultValue);



#endif



\START_CUDA_MAKEFILE_TPL

NVCC= nvcc
CPPC= g++
CXXFLAGS+=-g -Wall 
LDFLAGS=-lboost_program_options -lboost_serialization

#USER MAKEFILE OPTIONS :
\INSERT_MAKEFILE_OPTION#END OF USER MAKEFILE OPTIONS

EASEA_SRC= EASEATools.cpp EASEAIndividual.cpp
EASEA_MAIN_HDR= EASEA.cpp
EASEA_UC_HDR= EASEAUserClasses.hpp

EASEA_HDR= $(EASEA_SRC:.cpp=.hpp) 

SRC= $(EASEA_SRC) $(EASEA_MAIN_HDR)
HDR= $(EASEA_HDR) $(EASEA_UC_HDR)
OBJ= $(EASEA_SRC:.cpp=.o) $(EASEA_MAIN_HDR:.cpp=.o)

BIN= EASEA
  
all:$(BIN)

$(BIN):$(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

easeaclean: clean
	rm -f Makefile $(SRC) $(HDR) EASEA.mak
clean:
	rm -f $(OBJ) $(BIN)

\START_EO_PARAM_TPL#****************************************
#                                         
#  EASEA.prm
#                                         
#  Parameter file generated by AESAE-EO v0.7b
#                                         
#***************************************
# --seed=0   # -S : Random number seed. It is possible to give a specific seed.

######    Evolution Engine    ######
--popSize=\POP_SIZE # -P : Population Size
--nbOffspring=\OFF_SIZE # -O : Nb of offspring (percentage or absolute)

######    Evolution Engine / Replacement    ######
--elite=\ELITE_SIZE  # Nb of elite parents (percentage or absolute)
--eliteType=\ELITISM # Strong (true) or weak (false) elitism (set elite to 0 for none)
--surviveParents=\SURV_PAR_SIZE # Nb of surviving parents (percentage or absolute)
# --reduceParents=Ranking # Parents reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform
--surviveOffspring=\SURV_OFF_SIZE  # Nb of surviving offspring (percentage or absolute)
# --reduceOffspring=Roulette # Offspring reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform
# --reduceFinal=DetTour(2) # Final reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform


\TEMPLATE_END
