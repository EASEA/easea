

#include <fstream>
#include <time.h>
#include <cstring>
#include <sstream>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"

using namespace std;

#include "memetic_weierstrassIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
#define STD_TPL

// User declarations
#line 8 "memetic_weierstrass.ez"

#define SIZE 100
#define X_MIN -1.
#define X_MAX 1.
#define ITER 120      
#define Abs(x) ((x) < 0 ? -(x) : (x))
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define SIGMA  1.                     /*  mutation parameter */
#define PI 3.141592654

 
float pMutPerGene=0.1;






// User functions

#line 26 "memetic_weierstrass.ez"

//fitness function
#include <math.h>

  inline float Weierstrass(float x[SIZE], int n)  // Weierstrass multimidmensionnel h = 0.25
{
   float res = 0.;
   float val[SIZE];
   float b=2.;
   float h = 0.25;

   for (int i = 0;i<n; i++) {
	val[i] = 0.;
    	for (int k=0;k<ITER;k++)
		val[i] += pow(b,-(float)k*h) * sin(pow(b,(float)k)*x[i]);
	res += Abs(val[i]);
	}
   return (res);
} 

float gauss()
/* Generates a normally distributed globalRandomGenerator->random value with variance 1 and 0 mean.
    Algorithm based on "gasdev" from Numerical recipes' pg. 203. */
{
  int iset = 0;
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
#line 75 "memetic_weierstrass.ez"

  //cout<<"Before everything else function called "<<endl;
}

// Finalization function
void EASEAFinalization(CPopulation* population){
#line 79 "memetic_weierstrass.ez"

  //cout << "After everything else function called" << endl;
}



void evale_pop_chunk(CIndividual** population, int popSize){
  
// No Instead evaluation step function.

}

void memetic_weierstrassInit(int argc, char** argv){
	
  EASEAInitFunction(argc, argv);

}

void memetic_weierstrassFinal(CPopulation* pop){
	
  EASEAFinalization(pop);
;
}

void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	#line 190 "memetic_weierstrass.ez"
{
#line 83 "memetic_weierstrass.ez"

  //cout << "At the beginning of each generation function called" << endl;
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
   
  // Genome Initialiser
#line 110 "memetic_weierstrass.ez"
 // "initializer" is also accepted
  for(int i=0; i<SIZE; i++ ) {
     	(*this).x[i] = (float)globalRandomGenerator->random(X_MIN,X_MAX);
	(*this).sigma[i]=(float)globalRandomGenerator->random(0.,0.5);
	}

  valid = false;
  isImmigrant = false;
}

CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  // Destructing pointers

}


float IndividualImpl::evaluate(){
  if(valid)
    return fitness;
  else{
    valid = true;
    #line 142 "memetic_weierstrass.ez"
 // Returns the score
  float Score= 0.0;
  Score= Weierstrass((*this).x, SIZE);         
  return fitness =  Score;

  }
}

void IndividualImpl::boundChecking(){
	
// No Bound checking function.

}

string IndividualImpl::serialize(){
    ostringstream EASEA_Line(ios_base::app);
    // Memberwise serialization
	for(int EASEA_Ndx=0; EASEA_Ndx<100; EASEA_Ndx++)
		EASEA_Line << this->sigma[EASEA_Ndx] <<" ";
	for(int EASEA_Ndx=0; EASEA_Ndx<100; EASEA_Ndx++)
		EASEA_Line << this->x[EASEA_Ndx] <<" ";

    EASEA_Line << this->fitness;
    return EASEA_Line.str();
}

void IndividualImpl::deserialize(string Line){
    istringstream EASEA_Line(Line);
    string line;
    // Memberwise deserialization
	for(int EASEA_Ndx=0; EASEA_Ndx<100; EASEA_Ndx++)
		EASEA_Line >> this->sigma[EASEA_Ndx];
	for(int EASEA_Ndx=0; EASEA_Ndx<100; EASEA_Ndx++)
		EASEA_Line >> this->x[EASEA_Ndx];

    EASEA_Line >> this->fitness;
    this->valid=true;
    this->isImmigrant = false;
}

IndividualImpl::IndividualImpl(const IndividualImpl& genome){

  // ********************
  // Problem specific part
  // Memberwise copy
    {for(int EASEA_Ndx=0; EASEA_Ndx<100; EASEA_Ndx++)
       sigma[EASEA_Ndx]=genome.sigma[EASEA_Ndx];}
    {for(int EASEA_Ndx=0; EASEA_Ndx<100; EASEA_Ndx++)
       x[EASEA_Ndx]=genome.x[EASEA_Ndx];}



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
  	#line 117 "memetic_weierstrass.ez"

  for (int i=0; i<SIZE; i++)
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
  #line 125 "memetic_weierstrass.ez"
 // Must return the number of mutations
  int NbMut=0;
  float pond = 1./sqrt((float)SIZE);

    for (int i=0; i<SIZE; i++)
    if (globalRandomGenerator->tossCoin(pMutPerGene)){
    	NbMut++;
       	(*this).sigma[i] = (*this).sigma[i] * exp(SIGMA*pond*(float)gauss());
       	(*this).sigma[i] = MIN(0.5,(*this).sigma[0]);              
       	(*this).sigma[i] = MAX(0.,(*this).sigma[0]);
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

	parentPopulationSize = setVariable("popSize",(int)10);
	offspringPopulationSize = setVariable("nbOffspring",(int)10);


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
	this->plotStats = setVariable("plotStats",0);
	this->printInitialPopulation = setVariable("printInitialPopulation",0);
	this->printFinalPopulation = setVariable("printFinalPopulation",0);
	this->savePopulation = setVariable("savePopulation",0);
	this->startFromFile = setVariable("startFromFile",0);

	this->outputFilename = (char*)"memetic_weierstrass";
	this->plotOutputFilename = (char*)"memetic_weierstrass.png";

	this->remoteIslandModel = setVariable("remoteIslandModel",0);
	std::string* ipFilename=new std::string();
	*ipFilename=setVariable("ipFile","NULL");

	this->ipFile =(char*)ipFilename->c_str();
	this->migrationProbability = setVariable("migrationProbability",(float)0.000000);
    this->serverPort = setVariable("serverPort",0);
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	//EZ_NB_GEN = (unsigned*)setVariable("nbGen",100);
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
	if(this->params->startFromFile){
	  ifstream EASEA_File("memetic_weierstrass.pop");
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

