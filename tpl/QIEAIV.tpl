\TEMPLATE_START
/***********************************************************************
| QIEAIII   Single Objective Quantum Inspired Algorithm Template                            |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2019-10                                                         |
|                                                                       |
 ***********************************************************************/


\ANALYSE_PARAMETERS
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "COptionParser.h"
#include "CRandomGenerator.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "EASEAIndividual.hpp"

using namespace std;

/** Global variables for the whole algorithm */

CIndividual** pPopulation = NULL;
CIndividual*  bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
unsigned *EZ_NB_GEN;
unsigned *EZ_current_generation;
int EZ_POP_SIZE;
int OFFSPRING_SIZE;

CEvolutionaryAlgorithm* EA;

int main(int argc, char** argv){


	ParametersImpl p("EASEA.prm", argc, argv);
	CEvolutionaryAlgorithm* ea = p.newEvolutionaryAlgorithm();
	pPopulation = ea->population->parents;

	EA = ea;

	EASEAInit(argc, argv, p);

	CPopulation* pop = ea->getPopulation();

	ea->runEvolutionaryLoop();
	
	EASEAFinal(pop);

	delete pop;


	return 0;
}

\START_CUDA_GENOME_CU_TPL

#include <fstream>
#include <time.h>
#include <cstring>
#include <sstream>
#include <chrono>
#include <vector>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"
#include <CLogger.h>
#include <problems/CProblem.h>

using namespace std;
#define EASENA
bool bReevaluate = false;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

int SAMPLES = 1000;
CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
#define STD_TPL
typedef std::mt19937 TRandom;
typedef double TT;
typedef easea::problem::CProblem<TT> TP;
typedef TP::TV TV;
typedef TP::TV TO;
TRandom m_generator;
size_t limitGen;
bool reset;
int num = 0;

typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES


\INSERT_USER_FUNCTIONS

\INSERT_FINALIZATION_FUNCTION

size_t m_quantDomain = 1;
size_t m_classPopSize = -1;
//std::vector<std::vector<TT>> quantPop;
random_device rd;
mt19937 mt(rd());
uniform_real_distribution<double> point(0.0,1.0);
double B_MAX;
double B_MIN;
double Vref;
double Vave;
double Vsum;
double Esum;
int N;  // number of particle 
int N0; // initial number of particle
double ds; // monte carlo step width
double dt; //time width
int mcs;
double b;
//double v;
double a; //diffusion adjustment parameter
double size;
double width;
int PSI_SIZE;
/*
 * \brief Function for find maximal value
 *        
 */

/*double V(double x, double b, double v, double F){
    if (x <-b || x > b)
            return 0.0;
    return v + F;
}*/
/*
double V(double const &x, double const &b, double f){
    if (x <-b || x > b)
	return 0.;
    return 0.5*x*x + f;

}
*/
double V(double const &r, double f){
	return -1/r + f;
}
double rnd()
{
    return point(mt);
}
double R(std::vector<double> x){
    double sum = 0.;
    for (size_t i = 0; i < NB_VARIABLES; i++)
	    sum += x[i]*x[i];
    return sqrt(sum);
}
 /*
  in[1]: x coordinate of the i-th particle
  in[2]: Vref reference potential
  in[3]: size - width of box for sorting corresponding to array psi
  in[4]: dt time interval
  in[5]: I total energy
  in[6]: nequil
  in[7]: a diffusion adjustment parameter
  in[8]: b
  int9]   : v
*/

double getMax(std::vector<double> a, const size_t nbVar)
{
        double c=a[0];

        for(int i=1;i<nbVar;i++)
        {
                while(a[i]>c)
                {
                        c=a[i];
                }
        }
        return c;
}

int switchPop(const size_t nbVar, const size_t szPop, std::vector<std::vector<double>>&Pop, std::vector<std::vector<double>>&tempPop, std::vector<double> F, std::vector<double>&tempF )
{
        int i,j;
        for(i=0; i < nbVar; i++)
        {
                for(j=0; j < szPop; j++)
                {
                        tempPop[j][i]=Pop[j][i];
                        tempF[j]=F[j];
                }
        }
        return 1;
}
double AverageRandom(double min,double max)
{
        long minInteger =(long) (min*1000000);
    long maxInteger =(long) (max*1000000);
        long randInteger =rand();
        long diffInteger =maxInteger - minInteger;
        return (randInteger%diffInteger+minInteger)/1000000.0;
}
double Norm(double miu,double sigma)
{

        double x;
        double u1=rand()*1.0/RAND_MAX;
    double u2=rand()*1.0/RAND_MAX;
        x=/*(2/sigma)*powf(sin((PI*miu*u1)/sigma),2);*/ miu+sigma*sqrt(-2.0*(log(u1)))*cos(2.0*PI*u2);
        return x;

}
double NormalRandom(double miu,double sigma)
{
        double norm_x;
        do{
        norm_x=Norm(miu,sigma);
        }while(norm_x<B_MIN && norm_x>B_MAX);
    return norm_x;
}

std::vector<double> getVariance(const size_t nbVar, const size_t szPop, std::vector<std::vector<TT>> tempPop)
{
        double sum[nbVar]={0.0}; // save the intermediate operation value when calculating the mean and variance
        double aver[nbVar]={0.0};// save the intermediate operation value when calculating the mean and variance
        double sum1[nbVar]={0.0};
	std::vector<double>accuracy(nbVar);

        for(int i=0; i < nbVar; i++)
        {
                for(int j=0; j < szPop; j++)
                {
                        sum[i] += tempPop[j][i];//Summing each dimension coordinate
                }
                aver[i]=sum[i]/szPop;   // Calculate the average of each dimension of coordinates
                for(int j=0; j < szPop; j++)
                {
                        sum1[i] +=(tempPop[j][i]-aver[i])*(tempPop[j][i]-aver[i]);
                }
                accuracy[i]=sqrt(sum1[i]/szPop);
        }
return accuracy;
}

/*
 * Init population
 */


std::vector<double> initPopulation(const std::vector<std::pair<TT,TT>>&boundary, const size_t szPop, std::vector<std::vector<double>>&Pop, std::vector<std::vector<double>>&tempPop, std::vector<double>&F)
{
	size_t szDimension = boundary.size(); /* Size of search space dimension */
	
	
	size = 2*ds; // twice the stride

        int i,j;
        std::vector<double> tmp(szDimension); 
	/* Define number of particles */
	N0 = szPop; // initial number of particles
	N = N0;
	ds = 0.1;
	mcs = 1000;
	dt = ds * ds; //time step 
	width = 1.;  //B_MAX-B_MIN; // width of partition box for psi array
	Vref = 0.;/*define reference potential for every individual (particle)*/
	Vsum = 0.;/* total energy */

        for(i=0;i < szPop; i++)
        {
                for(j=0; j < szDimension; j++)
                {
                        tempPop[i][j]=AverageRandom(boundary[j].first, boundary[j].second);
                /*        double pulse = (boundary[j].second - boundary[j].first)/(double)szPop;
                        if (i == 0)
                            tempPop[i][j] = -boundary[j].second + pulse/2.;
                        else
                            tempPop[i][j] = tempPop[i-1][j] + pulse;
*/
		        Pop[i][j]=tempPop[i][j];
                        tmp[j]=tempPop[i][j];
	//		Pop[i][j] = (2*rnd() - 1)*width;
	//		Vref[i] = Vref+V(Pop[i][j], 0);
			
                }

		pPopulation[i] = (new IndividualImpl(tmp));
		F[i] = pPopulation[i]->evaluate();
		Vref = Vref + V(R(tmp), F[i]);
	//	pPopulation[i]->Vref = Vref;		
        }
	Vref = Vref/(double)N; //The initial reference energy is the average of the potential energy of all particles
	size = 2*ds;
	/* Calculate the variance for each individual at initialization */
        std::vector<double> accuracy(szDimension);
	accuracy = getVariance(szDimension, szPop, tempPop);

        return accuracy;
}
int sampling(std::vector<double> mu,double sigma, const size_t nbVar, const size_t szPop, std::vector<std::vector<double>>&Pop, std::vector<double>&F)
{
        double datatemp[nbVar]; //save sample points for each dimension
        double rtemp;
        double stemp;
        int i=0,p=0,s=0;

        while( i < SAMPLES)
        {
		/* Generate high-dimensional sample point locations according to a normal distribution */
                for(int j=0; j < nbVar; j++)
                {
                        datatemp[j]=NormalRandom(mu[j],sigma);
			((IndividualImpl *)pPopulation[0])->x[j] = datatemp[j];

                }
                stemp=F[0];
                s=0;
                p=1;
		/* Find the maximum from szPop values */
                while(p < szPop)
                {
                        if(stemp < F[p])
                        {
                        stemp=F[p];
                        s=p;
                        }
                        p++;
                }
		rtemp = pPopulation[0]->evaluate();
//		double y[2];
//		coco_eval(datatemp, y);
//		rtemp = y[0];
                if(rtemp<stemp)
                {
                        for(int k=0; k < nbVar; k++)
                        {
                                Pop[s][k]=datatemp[k];
                        }
                        F[s]=rtemp;
                }
                i++;
        }
        return 1;
}
/*
  in[1]: x coordinate of i-th particle
  in[2]: N numbre of particle
  in[3]: N0 initial numbre of particle
  in[4]: Vref reference potential
  in[5]: Vave average potential of all particle
  in[6]: ds Monte Carlo step 
  in[7]: dt time step
  in[8]: a  Diffusion adjustment
  in[9]: b
  in     v
*/
double gauss(double dt){
        //Box Muller method
        return sqrt(dt) * sqrt( -2.0 * log(rnd()) )*cos(2.0*M_PI*rnd());
}
void walk(std::vector<std::vector<double>>&x, int nbVar, int &N, int &N0, double &Vref,
          double &Vave, double const ds, double const dt,
          double const &a, double const &b, std::vector<double> &F){
  
    int i, j, k, w;
    int Nin = N; // current population size

//    double x_old[Nin];

    double Vsum = 0.0;
    double Vold, Vnew, dV;
    double Gbranch;

    // for all individuals (particles) in population
  //  For all particles

    for( i = Nin-1; i >= 0; i-- ){

    /* Diffusion process */
	for (j = 0; j < nbVar; j++)
	    ((IndividualImpl*)pPopulation[0])->x[j]=x[i][j];
	double f = ((IndividualImpl*)pPopulation[0])->evaluate();
	Vold = V(R(x[i]),  f);

	for (j = 0; j < nbVar; j++){
	    x[i][j] += gauss(dt); //transition according to Gaussian distribution
	    ((IndividualImpl*)pPopulation[0])->x[j]=x[i][j];
	}
	f = ((IndividualImpl*)pPopulation[0])->evaluate();
	Vnew = V(R(x[i]),  f);


	dV = (Vnew + Vold)/2 - (Vref);
	Gbranch = exp(-dV*dt); // Branch Green function

	w = (int)(Gbranch + rnd()); // Gbranch + rnd() Integer part of Branch Green

	/* Branching process */
	if( w > 1 ){ // split the particle w times

         for( j = 0; j < w-1; j++ ){ // Since one already exists, increase w-1
	    (N)++;           // Add one particle
	    std::vector<double> tmp(nbVar);
	    for (k = 0; k < nbVar; k++)
		tmp[k] = x[i][k];
	    x.insert(x.end(),tmp);
//    		x[(N-1)][k] = x[i][k];   // New particle position is current
        }

        Vsum += w*Vnew;     // There are w particles at the same position so the potential is w*Vnew

	}else if( w == 1 ){ //The number of particles is same
    	Vsum += Vnew;

    }else{ // particle disappears
	for(k = 0; k < nbVar; k++)
    	    x[i] = x[(N-1)];  // The position of the i-th particle is the last particle
    	    (N)--;          // we reduce the number of perticles - it will disappear formally speaking
	    x.erase(x.end());
      // Since the particle has disappeared, there is no potential calculation
    } // end

  } // end
  (Vave) = Vsum/(N); //avarage of all particles potential
  (Vref) = (Vave) - a*((N)-N0)/(N0*dt); // new reference potential

}
/*
  in[1]: x coord of the i-th particle
  in[2]: psi
  in[3]: N number of particle
  in[4]: Vref
  in[5]: Vave
  in[6]: Esum
  in[7]: imcs
  in[8]: size
*/

void data(std::vector<std::vector<double>>&x, double *psi, int N, double Vref, double Vave,
          double &Esum, int imcs, double size){

  double Ebas, position;
  int bin;
  // accumulate data
  Esum = Esum + Vave;

  // Disribute particle into boxes with size spaceing
  for( int i = 0; i < N; i++ ){

    position = R(x[i]);
    // Since position is negative, split the array psi so, that the origin is PSI_SIZE/2
    bin = floor(position/size+0.5);//(int)(NB_VARIABLES/2) + floor(position/size + 0.5); // psi is divided equally into N_SIZE Think of a box
    psi[bin]++;                        // increase psi corresponding to particle position by 1
  }

  Ebas = Esum/imcs; //ground state energy
  if( imcs%10 == 0 ){
    printf("imc  = %d\n", imcs);
    printf("N    = %d\n", N);
    printf("Ebas = %f\n", Ebas);
    printf("Vref = %f\n", Vref);
    printf("\n");
  }

}

 // psi[] Particle postiion from psi
void output_psi(double *psi, int N, double size, FILE *fp, std::vector<double>f){
       int i, mi, pi;
       double norm, Psum;
       Psum = 0.0;

    
      for( i = 0; i < PSI_SIZE; i++ ){
         Psum += psi[i]*psi[i];
   }
     
       norm = sqrt(Psum*size); // normalisation factor
     
       for( i = 0; i < PSI_SIZE/2; i++ ){
    
        // if( i == 0 ){
           // printf("x = %f\t psi = %f\n",i*size, psi[PSI_SIZE/2]/norm);
           fprintf(fp,"%f\t%f\n",i*size, psi[/*PSI_SIZE/2*/i]/norm);
        /* }else{
     
          pi = i + PSI_SIZE/2;
	  mi = -i + PSI_SIZE/2;
           //
           // printf("x = %f\t psi = %f\n",i*size, psi[pi]/norm);
           // printf("x = %f\t psi = %f\n",-i*size, psi[mi]/norm);
     
           fprintf(fp,"%f\t%f\n",i*size, psi[pi]/norm);
           fprintf(fp,"%f\t%f\n",-i*size, psi[mi]/norm);
         }*/
       }
}
/* 
 *  Quantum Harmonic Oscillator Algorithm
 */
/*int qhoa(double sigma, const size_t nbVar, const size_t szPop, std::vector<std::vector<double>>&Pop, std::vector<std::vector<double>>&tempPop,std::vector<double>&F,std::vector<double>&tempF, std::vector<double>&accuracy)
{
        int p=0;
        while(p < szPop)
        {
                std::vector<double> data_c(nbVar);
		double x[nbVar];
		double psi[nbVar];
		double Vref = 0.;((IndividualImpl*)pPopulation[p])->Vref;
		double Vave = 0.;
		double Esum = 0.;

		// Export each dimension coordinate to data_c[] 
                for(int j=0; j < nbVar; j++)
                {
                        data_c[j]=tempPop[p][j]; 
			x[j]=tempPop[p][j];
			psi[j]=0.;
                }
		// Sampling region iteration 
            //    sampling(data_c,sigma, nbVar, szPop, Pop, F);
	    for( int imcs = 1; imcs <= mcs; imcs++){
		walk(x, N, N0, Vref, Vave, ds, dt, a, b, F[p]);
    		data(x, psi, N, Vref, Vave, Esum, imcs, size);
		FILE *fp = fopen("psi.data","w");
	        output_psi(psi, N, size, fp, F);
	        fclose(fp);
    	    }
	       for(int j=0; j < nbVar; j++)
	        {
		   tempPop[p][j]=x[j];
		    ((IndividualImpl *)pPopulation[p])->x[j] = x[j];
    
		   }
		    F[p]= pPopulation[p]->evaluate();
		    ((IndividualImpl*)pPopulation[p])->Esum = Esum/mcs;


                p++;
	//	FILE *fp = fopen("psi.data","w");
	//        output_psi(psi, N, size, fp, F);
	//        fclose(fp);

        }
        switchPop(nbVar, szPop, Pop, tempPop, F, tempF);
        accuracy = getVariance(nbVar, szPop, tempPop);
        if(num%50==0)
                srand((int)time(0));
        num++;
        return 1;
}

*/

void evale_pop_chunk([[maybe_unused]] CIndividual** population, [[maybe_unused]] int popSize) {
  \INSTEAD_EVAL_FUNCTION
}

void EASEAInit(int argc, char* argv[], ParametersImpl& p){
	(void)argc;(void)argv;(void)p;
	auto setVariable = [&](std::string const& arg, auto def) {
		return p.setVariable(arg, std::forward<decltype(def)>(def));
	}; // for compatibility
	(void)setVariable;

	\INSERT_INITIALISATION_FUNCTION
	if (m_classPopSize <= 0) LOG_ERROR(errorCode::value, "Wrong size of parent population");
	if ((m_quantDomain <= 0) || (m_quantDomain > m_classPopSize/2)) LOG_ERROR(errorCode::value, "Wrong size of quantum domain number");
	//quantPop = initQuantumPopulation(m_problem.getBoundary(), m_quantDomain);
}

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;
}

void AESAEBeginningGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_BEGIN_GENERATION_FUNCTION
 
}
void EASEABeginningGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_END_GENERATION_FUNCTION
}
void EASEAEndGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_END_GENERATION_FUNCTION
}
void EASEAGenerationFunctionBeforeReplace(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}

void AESAEGenerationFunctionBeforeReplacement([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}


IndividualImpl::IndividualImpl() : CIndividual() {
  \GENOME_CTOR 
  \INSERT_EO_INITIALISER
  valid = false;
  isImmigrant = false;
 
}
IndividualImpl::IndividualImpl(std::vector<double> ind)
{
	for (size_t i = 0; i < m_problem.getBoundary().size(); i++)
	    x[i] = ind[i];
}

CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  \GENOME_DTOR
}


float IndividualImpl::evaluate(){
 /* if(valid)
    return fitness;
  else{
    valid = true;*/
    \INSERT_EVALUATOR
  //}
}

void IndividualImpl::boundChecking(){
	\INSERT_BOUND_CHECKING
}

string IndividualImpl::serialize(){
    ostringstream AESAE_Line(ios_base::app);
    \GENOME_SERIAL
    AESAE_Line << this->fitness;
    return AESAE_Line.str();
}

void IndividualImpl::deserialize(string Line){
    istringstream AESAE_Line(Line);
    string line;
    \GENOME_DESERIAL
    AESAE_Line >> this->fitness;
    this->valid=true;
    this->isImmigrant = false;
}


IndividualImpl::IndividualImpl(const IndividualImpl& genome){

  // ********************
  // Problem specific part
  \COPY_CTOR


  // ********************
  // Generic part
  this->valid = genome.valid;
  this->fitness = genome.fitness;
  this->isImmigrant = false;
}


CIndividual* IndividualImpl::crossover(CIndividual** ps){
	// ********************
	// Generic part
	IndividualImpl** tmp = (IndividualImpl**)ps;
	IndividualImpl parent1(*this);
	IndividualImpl parent2(*tmp[0]);
	IndividualImpl child(*this);

	//DEBUG_PRT("Xover");
	/*   cout << "p1 : " << parent1 << endl; */
	/*   cout << "p2 : " << parent2 << endl; */

	// ********************
	// Problem specific part
  	\INSERT_CROSSOVER


	child.valid = false;
	/*   cout << "child : " << child << endl; */
	return new IndividualImpl(child);
}


void IndividualImpl::printOn([[maybe_unused]] std::ostream& os) const {
	\INSERT_DISPLAY
}

std::ostream& operator << (std::ostream& O, const IndividualImpl& B)
{
  // ********************
  // Problem specific part
  O << "\nIndividualImpl : "<< std::endl;
  O << "\t\t\t";
  B.printOn(O);

  if( B.valid ) O << "\t\t\tfitness : " << B.fitness;
  else O << "fitness is not yet computed" << std::endl;
  return O;
}


unsigned IndividualImpl::mutate([[maybe_unused]] float pMutationPerGene ) {
  this->valid=false;


  // ********************
  // Problem specific part
  \INSERT_MUTATOR

  return 0;
}

ParametersImpl::ParametersImpl(std::string const& file, int argc, char* argv[]) : Parameters(file, argc, argv) {

	this->minimizing = \MINIMAXI;
	this->nbGen = setVariable("nbGen", (int)\NB_GEN);

	globalRandomGenerator = new CRandomGenerator(seed);
	this->randomGenerator = globalRandomGenerator;


	selectionOperator = getSelectionOperator(setVariable("selectionOperator", "\SELECTOR_OPERATOR"), this->minimizing, globalRandomGenerator);
	replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator", "\RED_FINAL_OPERATOR"),this->minimizing, globalRandomGenerator);
	parentReductionOperator = getSelectionOperator(setVariable("reduceParentsOperator", "\RED_PAR_OPERATOR"),this->minimizing, globalRandomGenerator);
	offspringReductionOperator = getSelectionOperator(setVariable("reduceOffspringOperator", "\RED_OFF_OPERATOR"),this->minimizing, globalRandomGenerator);
	selectionPressure = setVariable("selectionPressure", (float)\SELECT_PRM);
	replacementPressure = setVariable("reduceFinalPressure", (float)\RED_FINAL_PRM);
	parentReductionPressure = setVariable("reduceParentsPressure", (float)\RED_PAR_PRM);
	offspringReductionPressure = setVariable("reduceOffspringPressure", (float)\RED_OFF_PRM);
	pCrossover = \XOVER_PROB;
	pMutation = \MUT_PROB;
	pMutationPerGene = 0.05;

	parentPopulationSize = setVariable("popSize", (int)\POP_SIZE);
	offspringPopulationSize = getOffspringSize((int)\OFF_SIZE, \POP_SIZE);
	m_classPopSize = parentPopulationSize;

	parentReductionSize = setReductionSizes(parentPopulationSize, setVariable("survivingParents", (float)\SURV_PAR_SIZE));
	offspringReductionSize = setReductionSizes(offspringPopulationSize, setVariable("survivingOffspring", (float)\SURV_OFF_SIZE));

	this->elitSize = setVariable("elite", (int)\ELITE_SIZE);
	this->strongElitism = setVariable("eliteType", (int)\ELITISM);

	if((this->parentReductionSize + this->offspringReductionSize) < this->parentPopulationSize){
		printf("*WARNING* parentReductionSize + offspringReductionSize < parentPopulationSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	if((this->parentPopulationSize-this->parentReductionSize)>this->parentPopulationSize-this->elitSize){
		printf("*WARNING* parentPopulationSize - parentReductionSize > parentPopulationSize - elitSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	if(!this->strongElitism && ((this->offspringPopulationSize - this->offspringReductionSize)>this->offspringPopulationSize-this->elitSize)){
		printf("*WARNING* offspringPopulationSize - offspringReductionSize > offspringPopulationSize - elitSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	

	/*
	 * The reduction is set to true if reductionSize (parent or offspring) is set to a size less than the
	 * populationSize. The reduction size is set to populationSize by default
	 */
	if(offspringReductionSize<offspringPopulationSize) offspringReduction = true;
	else offspringReduction = false;

	if(parentReductionSize<parentPopulationSize) parentReduction = true;
	else parentReduction = false;

	generationalCriterion = new CGenerationalCriterion(setVariable("nbGen", (int)\NB_GEN));
	controlCStopingCriterion = new CControlCStopingCriterion();
	timeCriterion = new CTimeCriterion(setVariable("timeLimit", \TIME_LIMIT));


	this->printStats = setVariable("printStats", \PRINT_STATS);
	this->generateCSVFile = setVariable("generateCSVFile", \GENERATE_CSV_FILE);
	this->generatePlotScript = setVariable("generatePlotScript", \GENERATE_GNUPLOT_SCRIPT);
	this->generateRScript = setVariable("generateRScript", \GENERATE_R_SCRIPT);
	this->plotStats = setVariable("plotStats", \PLOT_STATS);
	this->savePopulation = setVariable("savePopulation", \SAVE_POPULATION);
	this->startFromFile = setVariable("startFromFile", \START_FROM_FILE);

	this->outputFilename = (char*)"EASEA";
	this->plotOutputFilename = (char*)"EASEA.png";

	this->remoteIslandModel = setVariable("remoteIslandModel", \REMOTE_ISLAND_MODEL);
	this->ipFile = setVariable("ipFile", "\IP_FILE");
	this->migrationProbability = setVariable("migrationProbability", (float)\MIGRATION_PROBABILITY);
    this->serverPort = setVariable("serverPort", \SERVER_PORT);
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	//EZ_NB_GEN = (unsigned*)setVariable("nbGen", \NB_GEN);
	EZ_current_generation=0;
  EZ_POP_SIZE = parentPopulationSize;
  OFFSPRING_SIZE = offspringPopulationSize;

	CEvolutionaryAlgorithm* ea = new EvolutionaryAlgorithmImpl(this);
	generationalCriterion->setCounterEa(ea->getCurrentGenerationPtr());
	ea->addStoppingCriterion(generationalCriterion);
	ea->addStoppingCriterion(controlCStopingCriterion);
	ea->addStoppingCriterion(timeCriterion);	

	EZ_NB_GEN=((CGenerationalCriterion*)ea->stoppingCriteria[0])->getGenerationalLimit();
	EZ_current_generation=&(ea->currentGeneration);

	 return ea;
}

void EvolutionaryAlgorithmImpl::initializeParentPopulation(){
        this->population->actualParentPopulationSize = this->params->parentPopulationSize;
}

/*
 * \brief Main Evolutionary Loop
 *  F(x) - objective function = Potential well V(x)
 *  best - global optimal solution = Wavefunction at the ground state
 *  sample - for Gaussina sampling - Wavefunction converges to the ground state
 *  local optimum - it is metastable state at high energy level
 *  replacement of optimal solution - transition from a metastable state to the ground state 
 */

void EvolutionaryAlgorithmImpl::runEvolutionaryLoop(){

	/* Get the size of serach space dimension from problem difinition by user in ez file*/
	const size_t nbVar = m_problem.getBoundary().size();
	
	/* Get population size from prm file - defined by user
	 * (in this version the same for quantum and classical population 
	 * 
	 */ 
	const size_t szPop = this->params->parentPopulationSize;

	/* Define quantum population */
	std::vector<std::vector<TT>>Pop(szPop, std::vector<TT>(nbVar));
	std::vector<std::vector<TT>>tempPop(szPop, std::vector<TT>(nbVar));
	/* Define reference potential */
	
	/* Objective function f(x) is the potential well V(x) in Schrödinger equation */
	std::vector<TT> F(szPop);
	std::vector<TT> tempF(szPop);

	/* array to hold the standard deviation of each dimension */
	std::vector<TT> accuracy(nbVar);

	/* wavefunction of harmonic oscillator potential is adopted as a sampling probability density function */
	/* Sample point matrix */
	//std::vector<std::vector<TT>> samplePos(szPop, std::vector<TT>(nbVar));
	
	/* Sigma - current standard deviation of optimal solution */
	B_MAX = m_problem.getBoundary()[0].second;
	B_MIN = m_problem.getBoundary()[0].first;
	b = B_MAX;
	double sigma = B_MAX-B_MIN;
	Vave = 0;
	PSI_SIZE = szPop;

	/* Define calculation accuracy - minimal value of  sigma */ 
	const double ACCURACY = 0.001; 

	/* Start logging */
	LOG_MSG(msgType::INFO, "QIEAII starting....");
	auto tmStart = std::chrono::system_clock::now();
	
	limitGen = EZ_NB_GEN[0];  /* Get maximal number of geneation from the prm file, defined by user */

        double best;		/* Local best value */
	reset = false;		/* Set "reset" in true by user in ez file in case of chain of test functions */ 
	currentGeneration = 0;  /* Counter of generation */
	size_t currentEval; 	/* Counter of evaluation number */
	bool first_time;
	pair<double, int> ans;
	double psi[szPop];
	/* Check all cases of stop critaria */ 
	while( this->allCriteria() == false){
	    srand(int(time(0)));
	    clock_t starttime, endtime;
	    double totaltime;
	    starttime = clock();
	    int p =0, q= 0;
	    double smin =0.0;
	    
	    /* Launch user settings from section "befor everything else" in ez file */
	    EASEABeginningGeneration(this);
	    
	    if (currentGeneration == 0){
		currentEval = 0;
		for (size_t i = 0; i < szPop; i++){
		    F[i] = 0; tempF[i] = 0; psi[i] = 0.;
		}
		for (size_t i = 0; i < nbVar; i++)
		    accuracy[i] = 0;
		/* Initialization of  population if the first generation */
		this->initializeParentPopulation();
		smin = 0; p = 0; q = 0;
		accuracy = initPopulation(m_problem.getBoundary(), szPop, Pop, tempPop, F);
		sigma = B_MAX - B_MIN;
		/* Set the initial best value */
		best = powf(2,64) - 1;
		reset = false;
		first_time = true;
	    }

	    /* Logging current generation number */
	    ostringstream ss;
	    ss << "Generation: " << currentGeneration << std::endl;
	    LOG_MSG(msgType::INFO, ss.str());
	    for( int imcs = 1; imcs <= mcs; imcs++){
		walk(Pop, nbVar, N, N0, Vref, Vave, ds, dt, a, b,  F);
		if (N <= 1){ printf("All prticles is disapperes\n"); break;}
		data(Pop, psi, N, Vref, Vave, Esum, imcs, size);
	    }    
	    FILE *fp = fopen("psi.data","w");
            output_psi(psi, N, size, fp, F);
            fclose(fp);

    
/*	    int cnt1 = 0;
	    do{	int cnt = 0;
		do{	
		    qhoa(sigma, nbVar, szPop, Pop, tempPop, F, tempF, accuracy);
		    cnt++;
		    if (cnt > 5) break;
		}while(getMax(accuracy, nbVar) >= sigma);
		//if (cnt < 50)
		    sigma=sigma/2.0;
		if(cnt > 5) {  cnt1++;}
		if (cnt1 == 2) break;
                smin=F[0];
                p=1;
                while(p < szPop)
                {
                        if(smin>F[p])
                        {
                                smin=F[p];
                                q=p;
                        }
                        p++;
                }
		best = smin;
printf("BEST: %f\n",best);
}while(sigma > ACCURACY);

		currentGeneration++;
		EASEAEndGeneration(this);
		if (reset == true)
		{
		    currentGeneration = 0;
		    reset = false;
		}
*/
	}

	std::chrono::duration<double> tmDur = std::chrono::system_clock::now() - tmStart;
	ostringstream ss;
        ss << "Total execution time (in sec.): " << tmDur.count() << std::endl;
	LOG_MSG(msgType::INFO, ss.str());

}

EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
	;
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include <cstring>
#include <sstream>

using namespace std;

class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;

extern int EZ_POP_SIZE;
extern int OFFSPRING_SIZE;

\INSERT_USER_CLASSES

class IndividualImpl : public CIndividual {

public: // in EASEA the genome is public (for user functions,...)
	double Vref;
	double Esum;
	// Class members
  	\INSERT_GENOME

public:
	IndividualImpl();
	IndividualImpl(const IndividualImpl& indiv);
        IndividualImpl(std::vector<double> ind);
	virtual ~IndividualImpl();
	float evaluate();
	static unsigned getCrossoverArrity(){ return 2; }
	float getFitness(){ return this->fitness; }
	CIndividual* crossover(CIndividual** p2);
	void printOn(std::ostream& O) const;
	CIndividual* clone();

	unsigned mutate(float pMutationPerGene);

	void boundChecking();      

	string serialize();
	void deserialize(string AESAE_Line);

	friend std::ostream& operator << (std::ostream& O, const IndividualImpl& B) ;
	void initRandomGenerator(CRandomGenerator* rg){ IndividualImpl::rg = rg;}

};


class ParametersImpl : public Parameters {
public:
	ParametersImpl(std::string const& file, int argc, char* argv[]);
	CEvolutionaryAlgorithm* newEvolutionaryAlgorithm();
};

/**
 * @TODO ces functions devraient s'appeler weierstrassInit, weierstrassFinal etc... (en gros EASEAFinal dans le tpl).
 *
 */

void EASEAInit(int argc, char* argv[], ParametersImpl& p);
void EASEAFinal(CPopulation* pop);
void EASEABeginningGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAEndGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAGenerationFunctionBeforeReplace(CEvolutionaryAlgorithm* evolutionaryAlgorithm);


class EvolutionaryAlgorithmImpl: public CEvolutionaryAlgorithm {
public:
	EvolutionaryAlgorithmImpl(Parameters* params);
	virtual ~EvolutionaryAlgorithmImpl();
	void initializeParentPopulation();
	void runEvolutionaryLoop();

};

#endif /* PROBLEM_DEP_H */

\START_CUDA_MAKEFILE_TPL

UNAME := $(shell uname)

ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

ifneq ("$(OS)","")
	EZ_PATH=../../
endif

EASEALIB_PATH=$(EZ_PATH)/libeasea/

CXXFLAGS =  -std=c++14 -fopenmp -O2 -g -Wall -fmessage-length=0 -I$(EASEALIB_PATH)include

OBJS = EASEA.o EASEAIndividual.o 

LIBS = -lpthread -fopenmp 
ifneq ("$(OS)","")
	LIBS += -lws2_32 -lwinmm -L"C:\MinGW\lib"
endif

#USER MAKEFILE OPTIONS :
\INSERT_MAKEFILE_OPTION
#END OF USER MAKEFILE OPTIONS

TARGET =	EASEA

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) -g $(EASEALIB_PATH)/libeasea.a $(LIBS)

	
#%.o:%.cpp
#	$(CXX) -c $(CXXFLAGS) $^

all:	$(TARGET)
clean:
ifneq ("$(OS)","")
	-del $(OBJS) $(TARGET).exe
else
	rm -f $(OBJS) $(TARGET)
endif
easeaclean:
ifneq ("$(OS)","")
	-del $(TARGET).exe *.o *.cpp *.hpp EASEA.png EASEA.dat EASEA.prm EASEA.mak Makefile EASEA.vcproj EASEA.csv EASEA.r EASEA.plot EASEA.pop
else
	rm -f $(TARGET) *.o *.cpp *.hpp EASEA.png EASEA.dat EASEA.prm EASEA.mak Makefile EASEA.vcproj EASEA.csv EASEA.r EASEA.plot EASEA.pop
endif
\START_CMAKELISTS
cmake_minimum_required(VERSION 3.9) # 3.9: OpenMP improved support
set(EZ_ROOT $ENV{EZ_PATH})

project(EASEA)
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

file(GLOB EASEA_src ${CMAKE_SOURCE_DIR}/*.cpp ${CMAKE_SOURCE_DIR}/*.c)
add_executable(EASEA ${EASEA_src})

target_compile_features(EASEA PUBLIC cxx_std_17)
target_compile_options(EASEA PUBLIC
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /W3>
	$<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -march=native -mtune=native -Wall -Wextra -pedantic>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/W4 /DEBUG:FULL>
	$<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -Wall -Wextra -pedantic>
	)

find_library(libeasea_LIB
	NAMES libeasea easea
	HINTS ${EZ_ROOT} ${CMAKE_INSTALL_PREFIX}/easena ${CMAKE_INSTALL_PREFIX}/EASENA
	PATH_SUFFIXES lib libeasea easea easena)
find_path(libeasea_INCLUDE
	NAMES CLogger.h
	HINTS ${EZ_ROOT}/libeasea ${CMAKE_INSTALL_PREFIX}/*/libeasea
	PATH_SUFFIXES include easena libeasea)
find_package(Boost)
find_package(OpenMP)

target_include_directories(EASEA PUBLIC ${Boost_INCLUDE_DIRS} ${libeasea_INCLUDE})
target_link_libraries(EASEA PUBLIC ${libeasea_LIB} $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX> $<$<CXX_COMPILER_ID:MSVC>:winmm>)

if (SANITIZE)
        target_compile_options(EASEA PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr> $<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>
)
        target_link_options(EASEA PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr> $<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>)
endif()

\INSERT_USER_CMAKE

\START_EO_PARAM_TPL#****************************************
#                                         
#  EASEA.prm
#                                         
#  Parameter file generated by STD.tpl AESAE v1.0
#                                         
#***************************************
# --seed=0   # -S : Random number seed. It is possible to give a specific seed.

######    Evolution Engine    ######
--popSize=\POP_SIZE # -P : Population Size
--nbOffspring=\OFF_SIZE # -O : Nb of offspring (percentage or absolute)

######	  Stopping Criterions    #####
--nbGen=\NB_GEN #Nb of generations
--timeLimit=\TIME_LIMIT # Time Limit: desactivate with (0) (in Seconds)

######    Evolution Engine / Replacement    ######
--elite=\ELITE_SIZE  # Nb of elite parents (absolute)
--eliteType=\ELITISM # Strong (1) or weak (0) elitism (set elite to 0 for none)
--survivingParents=\SURV_PAR_SIZE # Nb of surviving parents (percentage or absolute)
--survivingOffspring=\SURV_OFF_SIZE  # Nb of surviving offspring (percentage or absolute)
--selectionOperator=\SELECTOR_OPERATOR # Selector: Deterministic, Tournament, Random, Roulette 
--selectionPressure=\SELECT_PRM
--reduceParentsOperator=\RED_PAR_OPERATOR 
--reduceParentsPressure=\RED_PAR_PRM
--reduceOffspringOperator=\RED_OFF_OPERATOR 
--reduceOffspringPressure=\RED_OFF_PRM
--reduceFinalOperator=\RED_FINAL_OPERATOR
--reduceFinalPressure=\RED_FINAL_PRM

#####	Stats Ouput 	#####
--printStats=\PRINT_STATS #print Stats to screen
--plotStats=\PLOT_STATS #plot Stats
--printInitialPopulation=0 #Print initial population
--printFinalPopulation=0 #Print final population
--generateCSVFile=\GENERATE_CSV_FILE
--generatePlotScript=\GENERATE_GNUPLOT_SCRIPT
--generateRScript=\GENERATE_R_SCRIPT

#### Population save	####
--savePopulation=\SAVE_POPULATION #save population to EASEA.pop file
--startFromFile=\START_FROM_FILE #start optimisation from EASEA.pop file

#### Remote Island Model ####
--remoteIslandModel=\REMOTE_ISLAND_MODEL #To initialize communications with remote AESAE's
--ipFile=\IP_FILE
--migrationProbability=\MIGRATION_PROBABILITY #Probability to send an individual every generation
--serverPort=\SERVER_PORT
\TEMPLATE_END
