\TEMPLATE_START/**
 This is program entry for CUDA template for EASEA

*/
\ANALYSE_PARAMETERS
using namespace std;
#include <iostream>
#include "EASEATools.hpp"
#include "EASEAIndividual.hpp"

RandomGenerator* globalRandomGenerator;

int main(int argc, char** argv){

  size_t parentPopulationSize = \POP_SIZE;
  size_t offspringPopulationSize = \OFF_SIZE;
  float pCrossover = \XOVER_PROB;
  float pMutation = \MUT_PROB;
  float pMutationPerGene = 0.05;

  globalRandomGenerator = new RandomGenerator(0);

  SelectionOperator* selectionOperator = new \SELECTOR;
  SelectionOperator* replacementOperator = new \RED_FINAL;
  float selectionPressure = \SELECT_PRM;
  float replacementPressure = \RED_FINAL_PRM;


  \INSERT_INIT_FCT_CALL
    
  EvolutionaryAlgorithm ea(parentPopulationSize,offspringPopulationSize,selectionPressure,replacementPressure,
			   selectionOperator,replacementOperator,pCrossover, pMutation, pMutationPerGene);

  StoppingCriterion* sc = new GenerationalCriterion(&ea,\NB_GEN);
  ea.addStoppingCriterion(sc);
  Population* pop = ea.getPopulation();

  //pop->initializeParentPopulation();
  //pop->evaluateParentPopulation();
  
  //cout << *pop;

  ea.runEvolutionaryLoop();

  EASEAFinal(pop);

  delete pop;
  delete sc;

  return 0;
}


\START_CUDA_GENOME_CU_TPL
#include "EASEAIndividual.hpp"
#include "EASEAUserClasses.hpp"

extern RandomGenerator* globalRandomGenerator;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_USER_FUNCTIONS

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION


void EASEAFinal(Population* pop){
  \INSERT_FINALIZATION_FCT_CALL
}

using namespace std;

RandomGenerator* Individual::rg;

Individual::Individual(){
  \GENOME_CTOR 
  \INSERT_EO_INITIALISER
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

  DEBUG_PRT("Xover");
/*   cout << "p1 : " << parent1 << endl; */
/*   cout << "p2 : " << parent2 << endl; */

  // ********************
  // Problem specific part
  \INSERT_CROSSOVER

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
  \INSERT_MUTATOR  
}

/* ****************************************
   EvolutionaryAlgorithm class
****************************************/

/**
   @DEPRECATED This contructor will be deleted. It was for test only, because it
   is too much constrained (default selection/replacement operator)
 */
EvolutionaryAlgorithm::EvolutionaryAlgorithm( size_t parentPopulationSize,
					      size_t offspringPopulationSize,
					      float selectionPressure, float replacementPressure,
					      float pCrossover, float pMutation, 
					      float pMutationPerGene){
  RandomGenerator* rg = globalRandomGenerator;


  SelectionOperator* so = new MaxTournament(rg);
  SelectionOperator* ro = new MaxTournament(rg);
  
  Individual::initRandomGenerator(rg);
  Population::initPopulation(so,ro,selectionPressure,replacementPressure);
  
  this->population = new Population(parentPopulationSize,offspringPopulationSize,
				    pCrossover,pMutation,pMutationPerGene,rg);

  this->currentGeneration = 0;

  this->reduceParents = 0;
  this->reduceOffsprings = 0;


}

EvolutionaryAlgorithm::EvolutionaryAlgorithm( size_t parentPopulationSize,
					      size_t offspringPopulationSize,
					      float selectionPressure, float replacementPressure,
					      SelectionOperator* selectionOperator, SelectionOperator* replacementOperator,
					      float pCrossover, float pMutation, 
					      float pMutationPerGene){

  RandomGenerator* rg = globalRandomGenerator;

  SelectionOperator* so = selectionOperator;
  SelectionOperator* ro = replacementOperator;
  
  Individual::initRandomGenerator(rg);
  Population::initPopulation(so,ro,selectionPressure,replacementPressure);
  
  this->population = new Population(parentPopulationSize,offspringPopulationSize,
				    pCrossover,pMutation,pMutationPerGene,rg);

  this->currentGeneration = 0;

  this->reduceParents = 0;
  this->reduceOffsprings = 0;


}

void EvolutionaryAlgorithm::addStoppingCriterion(StoppingCriterion* sc){
  this->stoppingCriteria.push_back(sc);
}

void EvolutionaryAlgorithm::runEvolutionaryLoop(){

  std::cout << "Parent's population initializing "<< std::endl;
  this->population->initializeParentPopulation();  
  std::cout << *population << std::endl;
  
  while( this->allCriteria() == false ){    


    population->produceOffspringPopulation();
    population->evaluateOffspringPopulation();
    
    if(reduceParents)
      population->reduceParentPopulation(reduceParents);
    
    if(reduceOffsprings)
      population->reduceOffspringPopulation(reduceOffsprings);
    
    population->reduceTotalPopulation();
    
    currentGeneration += 1;
  }  
  population->sortParentPopulation();
  std::cout << *population << std::endl;
  std::cout << "Generation : " << currentGeneration << std::endl;
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

\INSERT_USER_CLASSES_DEFINITIONS

void EASEAInitFunction(int argc, char *argv[]);
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

  
};


/* ****************************************
   EvolutionaryAlgorithm class
****************************************/
class EvolutionaryAlgorithm{
public:
  EvolutionaryAlgorithm(  size_t parentPopulationSize, size_t offspringPopulationSize,
			  float selectionPressure, float replacementPressure, 
			  float pCrossover, float pMutation, float pMutationPerGene);
  EvolutionaryAlgorithm( size_t parentPopulationSize,size_t offspringPopulationSize,
			 float selectionPressure, float replacementPressure,
			 SelectionOperator* selectionOperator, SelectionOperator* replacementOperator,
			 float pCrossover, float pMutation, 
			 float pMutationPerGene);

  size_t* getCurrentGenerationPtr(){ return &currentGeneration;}


  void addStoppingCriterion(StoppingCriterion* sc);
  void runEvolutionaryLoop();
  bool allCriteria();
  Population* getPopulation(){ return population;}

private:
  size_t currentGeneration;
  Population* population;
  size_t reduceParents;
  size_t reduceOffsprings;
  std::vector<StoppingCriterion*> stoppingCriteria;
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
  DEBUG_PRT("Int Random Value : %d",min+rValue);
  return rValue+min;

}

int RandomGenerator::random(int min, int max){
  return randInt(min,max);
}

float RandomGenerator::randFloat(float min, float max){
  float rValue = (((float)rand()/RAND_MAX))*(max-min);
  DEBUG_PRT("Float Random Value : %f",min+rValue);
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
void MaxTournament::initialize(Individual** population, float selectionPressure) {
  SelectionOperator::initialize(population,selectionPressure);
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


void MinTournament::initialize(Individual** population, float selectionPressure) {
  SelectionOperator::initialize(population,selectionPressure);
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
void SelectionOperator::initialize(Individual** population, float selectionPressure){
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


/* ****************************************
   Population class
****************************************/
SelectionOperator* Population::selectionOperator;
SelectionOperator* Population::replacementOperator;

float Population::selectionPressure;
float Population::replacementPressure;



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


}

Population::~Population(){
  for( size_t i=0 ; i<actualOffspringPopulationSize ; i++ ) delete(offsprings[i]);
  for( size_t i=0 ; i<actualParentPopulationSize ; i++ )    delete(parents[i]);

  delete[](this->parents);
  delete[](this->offsprings);
}

void Population::initPopulation(SelectionOperator* selectionOperator, 
				SelectionOperator* replacementOperator,
				float selectionPressure, float replacementPressure){
  Population::selectionOperator   = selectionOperator;
  Population::replacementOperator = replacementOperator;
  Population::selectionPressure   = selectionPressure;
  Population::replacementPressure = replacementPressure;
}


void Population::initializeParentPopulation(){

  for( size_t i=0 ; i<parentPopulationSize ; i++ )
    parents[i] = new Individual();

  actualParentPopulationSize = parentPopulationSize;
  actualOffspringPopulationSize = 0;
  
  evaluateParentPopulation();
}


void Population::evaluatePopulation(Individual** population, size_t populationSize){
  for( size_t i=0 ; i < populationSize ; i++ )
    population[i]->evaluate();
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
Individual** Population::reducePopulation(Individual** population, size_t populationSize,
					  Individual** reducedPopulation, size_t obSize,
					  SelectionOperator* replacementOperator){
  

  replacementOperator->initialize(population,replacementPressure);

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
  Individual** nextGeneration = new Individual*[obSize];

  reducePopulation(offsprings,actualOffspringPopulationSize,nextGeneration,obSize,
		   Population::replacementOperator);

  // free no longer needed individuals
  for( size_t i=0 ; i<actualOffspringPopulationSize-obSize ; i++ )
    delete(parents[i]);
  delete[](parents);

  this->actualParentPopulationSize = obSize;
  parents = nextGeneration;
  return nextGeneration;
}


static int individualCompare(const void* p1, const void* p2){
  Individual** p1_i = (Individual**)p1;
  Individual** p2_i = (Individual**)p2;

  return p1_i[0]->getFitness() > p2_i[0]->getFitness();
}

void Population::sortPopulation(Individual** population, size_t populationSize){
  qsort(population,populationSize,sizeof(Individual*),individualCompare);
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

  replacementOperator->initialize(globalPopulation, replacementPressure);
  memcpy(globalPopulation,parents,sizeof(Individual*)*actualParentPopulationSize);
  memcpy(globalPopulation+actualParentPopulationSize,offsprings,
   	 sizeof(Individual*)*actualOffspringPopulationSize);


#if ((\ELITE_SIZE!=0) && (\ELITISM==false))                    // If there is elitism and it is weak
  Population::elitism(\ELITE_SIZE,globalPopulation,actualGlobalSize,
		      nextGeneration,parentPopulationSize); // do the elitism on the global (already merged) population
  actualGlobalSize -= \ELITE_SIZE;                // decrement the parent population size
#endif

    
  Population::reducePopulation(globalPopulation,actualGlobalSize,\ELITE_SIZE+nextGeneration,
			       parentPopulationSize-\ELITE_SIZE,replacementOperator);

  for( size_t i=0 ; i<offspringPopulationSize ; i++ )
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

  selectionOperator->initialize(parents,selectionPressure);

  for( size_t i=0 ; i<offspringPopulationSize ; i++ ){
    size_t index = selectionOperator->selectNext(offspringPopulationSize);
    p1 = parents[index];
    
    if( rg->tossCoin(pCrossover) ){
      for( size_t j=0 ; j<crossoverArrity-1 ; j++ ){
	index = selectionOperator->selectNext(offspringPopulationSize);
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
  
  float bestFitness;
  size_t bestIndividual;
  
  if( elitismSize >= 5 )DEBUG_PRT("Warning, elitism has O(n) complexity, elitismSize is maybe too big (%d)",elitismSize);
  
  
  for(size_t i = 0 ; i<elitismSize ; i++ ){
    bestFitness = replacementOperator->getExtremum();
    bestIndividual = 0;
    for( size_t j=0 ; j<populationSize-i ; j++ ){
      if( bestFitness < population[j]->getFitness() ){
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
    O << "\t\t" << *B.parents[i] << std::endl;
  } 

  O << "\t Offspring size : "<< realOffspringPopulationSize << "/" << 
    offspringPopulationSize << std::endl;
  for( size_t i=0 ; i<realOffspringPopulationSize ; i++){
    O << "\t\t" << *B.offsprings[i] << std::endl;
  }  
  return O; 
} 




\START_CUDA_TOOLS_H_TPL/* ****************************************
   Some tools classes for algorithm
****************************************/
#include <stdlib.h>
#include <vector>
#include <iostream>
class EvolutionaryAlgorithm;
class Individual;
class Population;

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
  virtual void initialize(Individual** population, float selectionPressure);
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
  virtual void initialize(Individual** population, float selectionPressure);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  RandomGenerator* rg;
  
};



class MinTournament : public SelectionOperator{
public:
  MinTournament(RandomGenerator* rg){ this->rg = rg; }
  virtual void initialize(Individual** population, float selectionPressure);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  RandomGenerator* rg;
  
};

class Population {
  
 private:
  
  float pCrossover;
  float pMutation;
  float pMutationPerGene;
  
  Individual** parents;
  Individual** offsprings;

  size_t parentPopulationSize;
  size_t offspringPopulationSize;

  size_t actualParentPopulationSize;
  size_t actualOffspringPopulationSize;

  static SelectionOperator* selectionOperator;
  static SelectionOperator* replacementOperator;

  RandomGenerator* rg;

 public:

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
  static void initPopulation(SelectionOperator* selectionOperator, 
			     SelectionOperator* replacementOperator,
			     float selectionPressure, float replacementPressure);

  static void sortPopulation(Individual** population, size_t populationSize);

  void sortParentPopulation(){ Population::sortPopulation(parents,actualParentPopulationSize);}

  void produceOffspringPopulation();

  friend std::ostream& operator << (std::ostream& O, const Population& B);


  void setParentPopulation(Individual** population, size_t actualParentPopulationSize){ 
    this->parents = population;
    this->actualParentPopulationSize = actualParentPopulationSize;
  }

  static Individual** reducePopulation(Individual** population, size_t populationSize,
				       Individual** reducedPopulation, size_t obSize,
				       SelectionOperator* replacementOperator);
};


#endif


\START_CUDA_MAKEFILE_TPL

NVCC= nvcc
CPPC= g++
CXXFLAGS+=-g
NVCCFLAGS=$(CXXFLAGS)

EASEA_CU_SRC= EASEAIndividual.cu 
EASEA_SRC= EASEATools.cpp
EASEA_MAIN_HDR= EASEA.cpp
EASEA_UC_HDR= EASEAUserClasses.hpp

EASEA_CU_HDR= $(EASEA_CU_SRC:.cu=.hpp)
EASEA_HDR= $(EASEA_SRC:.cpp=.hpp) 

SRC= $(EASEA_SRC) $(EASEA_CU_SRC) $(EASEA_MAIN_HDR)
HDR= $(EASEA_HDR) $(EASEA_CU_HDR) $(EASEA_UC_HDR)
OBJ= $(EASEA_SRC:.cpp=.o) $(EASEA_CU_SRC:.cu=.o) $(EASEA_MAIN_HDR:.cpp=.o)

BIN= EASEA
  
all:$(BIN)

%.o:%.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

$(BIN):$(OBJ)
	$(NVCC) $^ -o $@

easeaclean: clean
	rm -f Makefile $(SRC) $(HDR) EASEA.mak
clean:
	rm -f $(OBJ) $(BIN)

\TEMPLATE_END
