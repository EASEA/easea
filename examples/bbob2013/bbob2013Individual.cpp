

#include <fstream>
#include <time.h>
#include <string>
#include <sstream>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"

using namespace std;

#include "bbob2013Individual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
#define STD_TPL

// User declarations
#line 8 "bbob2013.ez"

#define SIZE 100
#define X_MIN -5.
#define X_MAX 5.
#define ITER 120      
#define Abs(x) ((x) < 0 ? -(x) : (x))
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define SIGMA  1.                     /*  mutation parameter */
#define PI 3.141592654

 
float pMutPerGene=0.1;
int DIM = 2;
int trialid = 3;
int funcId = 1; //passed by parameter
int instanceId = 3;

double * peaks;
double * Xopt; /*Initialized in benchmarkhelper.c*/
double Fopt;
unsigned int isInitDone=0;



// User classes


// User functions

#line 33 "bbob2013.ez"

#include <math.h>
extern "C" {
#include "bbobStructures.h"
}

  inline double bbob_eval( double *x)
{
   return fgeneric_evaluate(x);
} 

float gauss()
/* Generates a normally distributed globalRandomGenerator->random value with variance 1 and 0 mean.
    Algorithm based on "gasdev" from Numerical recipes' pg. 203. */
{
  static int iset = 0;
  float gset = 0.0;
  float v1 = 0.0, v2 = 0.0, r = 0.0;
  float factor = 0.0;

  if (iset) {
        iset = 0;
        return gset;
      	}
  else {    
        do {
            v1 = (float)globalRandomGenerator->random(0.,1.) * 2.0 - 1.0;
            v2 = (float)globalRandomGenerator->random(0.,1.) * 2.0 - 1.0;
            r = v1 * v1 + v2 * v2;
	                }
        while (r > 1.0);
        factor = sqrt (-2.0 * log (r) / r);
        gset = v1 * factor;
        iset = 1;
        return (v2 * factor);
    	}
}




// Initialisation function
void EASEAInitFunction(int argc, char *argv[]){
#line 76 "bbob2013.ez"

  fgeneric_initialize();
}

// Finalization function
void EASEAFinalization(CPopulation* population){
#line 80 "bbob2013.ez"

  //cout << "After everything else function called" << endl;
  //fgeneric_finalize();
}



void evale_pop_chunk(CIndividual** population, int popSize){
  
// No Instead evaluation step function.

}

void bbob2013Init(int argc, char** argv){
	
  EASEAInitFunction(argc, argv);

}

void bbob2013Final(CPopulation* pop){
	
  EASEAFinalization(pop);
;
}

void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	#line 192 "bbob2013.ez"
{
#line 85 "bbob2013.ez"

}
}

void EASEAEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	{

  //cout << "At the end of each generation function called" << endl;
}
}

void EASEAGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	{

  //cout << "At each generation before replacement function called" << endl;
}
}


IndividualImpl::IndividualImpl() : CIndividual() {
      sigma=NULL;
    x=NULL;
 
  // Genome Initialiser
#line 111 "bbob2013.ez"
 // "initializer" is also accepted
  (*this).x = new double[DIM];
  (*this).sigma = new double[DIM];
  for(int i=0; i<DIM; i++ ) {
     	(*this).x[i] = (double)globalRandomGenerator->random(X_MIN,X_MAX);
	(*this).sigma[i]=(double)globalRandomGenerator->random(0.,0.5);
	}

  valid = false;
  isImmigrant = false;
}

CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  // Destructing pointers
  if (sigma) delete []sigma;
  sigma=NULL;
  if (x) delete []x;
  x=NULL;

}


float IndividualImpl::evaluate(){
  if(valid)
    return fitness;
  else{
    valid = true;
    #line 145 "bbob2013.ez"
 // Returns the score
  double Score= 0.0;
  Score= bbob_eval((*this).x);         
  //Score= rosenbrock(Genome.x);         
  return fitness =  Score;

  }
}

void IndividualImpl::boundChecking(){
	
// No Bound checking function.
}

string IndividualImpl::serialize(){
    ostringstream EASEA_Line(ios_base::app);
    // Memberwise serialization

    EASEA_Line << this->fitness;
    return EASEA_Line.str();
}

void IndividualImpl::deserialize(string Line){
    istringstream EASEA_Line(Line);
    string line;
    // Memberwise deserialization

    EASEA_Line >> this->fitness;
    this->valid=true;
    this->isImmigrant = false;
}

IndividualImpl::IndividualImpl(const IndividualImpl& genome){

  
  
  // ********************
  // Problem specific part
  // Memberwise copy
   //  sigma=(genome.sigma ? new double(*(genome.sigma)) : NULL);
   // x=(genome.x ? new double(*(genome.x)) : NULL);
   sigma = new double[DIM];
   x = new double[DIM];
   
   for(int i=0; i<DIM; i++)
   {
       x[i] =  genome.x[i];
       sigma[i] = genome.sigma[i];
   }    


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
  	#line 120 "bbob2013.ez"

  for (int i=0; i<DIM; i++)
  {
    float alpha = (float)globalRandomGenerator->random(0.,1.); // barycentric crossover
     child.x[i] = alpha*parent1.x[i] + (1.-alpha)*parent2.x[i];
  }



	child.valid = false;
	/*   cout << "child : " << child << endl; */
	return new IndividualImpl(child);
}


void IndividualImpl::printOn(std::ostream& os) const{
	
/* 	 for( size_t i=0 ; i<SIZE ; i++){ */
/* 	      //     cout << Genome.x[i] << ":" << Genome.sigma[i] << "|"; */
/* 	      printf("0.00:0.00|",Genome.x[i],Genome.sigma[i]); */
/* 	 }	       */

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


unsigned IndividualImpl::mutate( float pMutationPerGene ){
  this->valid=false;


  // ********************
  // Problem specific part
  #line 128 "bbob2013.ez"
 // Must return the number of mutations
  int NbMut=0;
  float pond = 1./sqrt((float)DIM);

    for (int i=0; i<DIM; i++)
    if (globalRandomGenerator->tossCoin(pMutPerGene)){
    	NbMut++;
       	(*this).sigma[i] = (*this).sigma[i] * exp(SIGMA*pond*(float)gauss());
       	(*this).sigma[i] = MIN(0.5,(*this).sigma[i]);              
       	(*this).sigma[i] = MAX(0.,(*this).sigma[i]);
       	(*this).x[i] += (*this).sigma[i]*(float)gauss();
       	(*this).x[i] = MIN(X_MAX,(*this).x[i]);              // pour eviter les depassements
       	(*this).x[i] = MAX(X_MIN,(*this).x[i]);
    	}
return  NbMut>0?true:false;

}

void ParametersImpl::setDefaultParameters(int argc, char** argv){

	this->minimizing = true;
	this->nbGen = setVariable("nbGen",(int)100);

	seed = setVariable("seed",(int)time(0));
	globalRandomGenerator = new CRandomGenerator(seed);
	this->randomGenerator = globalRandomGenerator;


	selectionOperator = getSelectionOperator(setVariable("selectionOperator","Tournament"), this->minimizing, globalRandomGenerator);
	replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator","Tournament"),this->minimizing, globalRandomGenerator);
	parentReductionOperator = getSelectionOperator(setVariable("reduceParentsOperator","Tournament"),this->minimizing, globalRandomGenerator);
	offspringReductionOperator = getSelectionOperator(setVariable("reduceOffspringOperator","Tournament"),this->minimizing, globalRandomGenerator);
	selectionPressure = setVariable("selectionPressure",(float)2.000000);
	replacementPressure = setVariable("reduceFinalPressure",(float)2.000000);
	parentReductionPressure = setVariable("reduceParentsPressure",(float)2.000000);
	offspringReductionPressure = setVariable("reduceOffspringPressure",(float)2.000000);
	pCrossover = 1.000000;
	pMutation = 1.000000;
	pMutationPerGene = 0.05;

	parentPopulationSize = setVariable("popSize",(int)2048);
	offspringPopulationSize = setVariable("nbOffspring",(int)2048);


	parentReductionSize = setReductionSizes(parentPopulationSize, setVariable("survivingParents",(float)1.000000));
	offspringReductionSize = setReductionSizes(offspringPopulationSize, setVariable("survivingOffspring",(float)1.000000));

	this->elitSize = setVariable("elite",(int)1);
	this->strongElitism = setVariable("eliteType",(int)1);

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

	generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)100));
	controlCStopingCriterion = new CControlCStopingCriterion();
	timeCriterion = new CTimeCriterion(setVariable("timeLimit",0));

	this->optimise = 0;

	this->printStats = setVariable("printStats",1);
	this->generateCSVFile = setVariable("generateCSVFile",0);
	this->generatePlotScript = setVariable("generatePlotScript",0);
	this->generateRScript = setVariable("generateRScript",0);
	this->plotStats = setVariable("plotStats",1);
	this->printInitialPopulation = setVariable("printInitialPopulation",0);
	this->printFinalPopulation = setVariable("printFinalPopulation",0);
	this->savePopulation = setVariable("savePopulation",0);
	this->startFromFile = setVariable("startFromFile",0);

	this->outputFilename = (char*)"bbob2013";
	this->plotOutputFilename = (char*)"bbob2013.png";

	this->remoteIslandModel = setVariable("remoteIslandModel",0);
	this->ipFile = (char*)setVariable("ipFile","ip.txt").c_str();
	this->expId = (char*)setVariable("expId","weierstrass").c_str();
	this->working_path = (char*)setVariable("working_path","lfn:/grid/vo.complex-systems.eu/easea/experiments/").c_str();
	this->migrationProbability = setVariable("migrationProbability",(float)0.330000);
        this->serverPort = setVariable("serverPort",2929);
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	EZ_NB_GEN = (unsigned*)setVariable("nbGen",100);
	EZ_current_generation=0;

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
	if(this->params->startFromFile){
	  ifstream EASEA_File("bbob2013.pop");
	  string EASEA_Line;
  	  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
	  	  getline(EASEA_File, EASEA_Line);
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
		  ((IndividualImpl*)this->population->parents[i])->deserialize(EASEA_Line);
	  }
	  
	}
	else{
  	  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
	  }
	}
        this->population->actualParentPopulationSize = this->params->parentPopulationSize;
}


EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
	;
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

