/*
 * CPopulation.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CPopulation.h"
#include <iostream>
#include <fstream>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "include/CRandomGenerator.h"
#include "include/CIndividual.h"
#include "include/Parameters.h"
#include "include/CStats.h"

using namespace std;

CSelectionOperator* CPopulation::selectionOperator;
CSelectionOperator* CPopulation::replacementOperator;
CSelectionOperator* CPopulation::parentReductionOperator;
CSelectionOperator* CPopulation::offspringReductionOperator;


float CPopulation::selectionPressure;
float CPopulation::replacementPressure;
float CPopulation::parentReductionPressure;
float CPopulation::offspringReductionPressure;


extern float* pEZ_MUT_PROB;
extern float* pEZ_XOVER_PROB;
extern CIndividual** pPopulation;
extern CIndividual* bBest;

CPopulation::CPopulation(){
}

CPopulation::CPopulation(unsigned parentPopulationSize, unsigned offspringPopulationSize,
           float pCrossover, float pMutation, float pMutationPerGene,
           CRandomGenerator* rg, Parameters* params, CStats* cstats){

  this->parents     = new CIndividual*[parentPopulationSize];
  this->offsprings  = new CIndividual*[offspringPopulationSize];

  this->parentPopulationSize     = parentPopulationSize;
  this->offspringPopulationSize  = offspringPopulationSize;

  this->actualParentPopulationSize    = 0;
  this->actualOffspringPopulationSize = 0;

  this->pCrossover       = pCrossover;
  this->pMutation        = pMutation;
  pEZ_MUT_PROB = &this->pMutation;
  pEZ_XOVER_PROB = &this->pCrossover;
  this->pMutationPerGene = pMutationPerGene;
  pPopulation = parents;
  bBest = Best;

  this->rg = rg;

  this->currentEvaluationNb = 0;
  this->params = params;
  this->cstats = cstats;
}

void CPopulation::syncInVector(){
  for( unsigned i = 0 ; i<actualParentPopulationSize ; i++ ){
    parents[i] = pop_vect.at(i);
  }
}

void CPopulation::syncOutVector(){
  pop_vect.clear();
  for( unsigned i = 0 ; i<actualParentPopulationSize ; i++ ){
    pop_vect.push_back(parents[i]);
  }
#ifndef WIN32
  DEBUG_PRT("Size of outVector",pop_vect.size());
#endif
}

CPopulation::~CPopulation(){
  for( unsigned i=0 ; i<actualOffspringPopulationSize ; i++ ) delete(offsprings[i]);
  for( unsigned i=0 ; i<actualParentPopulationSize ; i++ )    delete(parents[i]);

  delete[](this->parents);
  delete[](this->offsprings);
}

void CPopulation::initPopulation(CSelectionOperator* selectionOperator,
        CSelectionOperator* replacementOperator,
        CSelectionOperator* parentReductionOperator,
        CSelectionOperator* offspringReductionOperator,
        float selectionPressure, float replacementPressure,
        float parentReductionPressure, float offspringReductionPressure){
  CPopulation::selectionOperator   = selectionOperator;
  CPopulation::replacementOperator = replacementOperator;
  CPopulation::parentReductionOperator = parentReductionOperator;
  CPopulation::offspringReductionOperator = offspringReductionOperator;

  CPopulation::selectionPressure   = selectionPressure;
  CPopulation::replacementPressure = replacementPressure;
  CPopulation::parentReductionPressure = parentReductionPressure;
  CPopulation::offspringReductionPressure = offspringReductionPressure;

}




void CPopulation::evaluatePopulation(CIndividual** population, unsigned populationSize){
  #pragma omp parallel for schedule(runtime)
  for( unsigned i=0 ; i < populationSize ; i++ )
    population[i]->evaluate();
}

void CPopulation::optimisePopulation(CIndividual** population, unsigned populationSize){
}

void CPopulation::evaluateParentPopulation(){
  evaluatePopulation(parents,parentPopulationSize);
}

void CPopulation::optimiseParentPopulation(){
  optimisePopulation(parents,parentPopulationSize);
}

void CPopulation::evaluateOffspringPopulation(){
  evaluatePopulation(offsprings,offspringPopulationSize);
}

void CPopulation::optimiseOffspringPopulation(){
  optimisePopulation(offsprings,offspringPopulationSize);
}


/**
   Reduit la population population de taille populationSize
   a une population reducedPopulation de taille obSize.
   reducedPopulation doit etre alloue a obSize.

   Ici on pourrait avoir le best fitness de la prochaine population de parents.


 */
void CPopulation::reducePopulation(CIndividual** population, unsigned populationSize,
            CIndividual** reducedPopulation, unsigned obSize,
            CSelectionOperator* replacementOperator,int pressure){
  
  int nbThread=omp_get_max_threads();
  int popSizeThread=(populationSize)/nbThread;
  int popSizeLastThread=populationSize/nbThread+populationSize%nbThread;
  int tid;
  unsigned int i;

  #pragma omp parallel private(tid,i)
  {
    tid=omp_get_thread_num();
    int actualPopSize=popSizeThread;
    int last=actualPopSize-1;
    int j=0;
    printf("%d nbThread %d popSize %d \n", actualPopSize,nbThread, populationSize );
    unsigned int selectedIndex;
    CRandomGenerator safeRG(CRandomGenerator(selectionOperator->rg->get_seed()+tid));
    CSelectionOperator* safeSelector=replacementOperator->copy(actualPopSize,&safeRG);
    CIndividual** popThread=population+(tid*(popSizeThread));
    printf(" %d pop %d \n",tid,popThread);
    
    safeSelector->initialize(population+((tid*(popSizeThread))),pressure,actualPopSize);
    
    #pragma omp for schedule(runtime)    
    for( i=0 ; i<obSize ; i++ ){

      // select an CIndividual and add it to the reduced population
      selectedIndex = safeSelector->selectNext((popSizeThread) - j);
      //printf(" %d index %d %d \n",tid,selectedIndex,i);
      
      printf("%d %p",i,&(popThread[selectedIndex]));
      reducedPopulation[i] = popThread[selectedIndex];
      
      //printf("%d move to %d \n",last,selectedIndex);
      popThread[selectedIndex] = popThread[last];
      last--;
      j++;
    }
  }
 printf("finito\n");
}


CIndividual** CPopulation::reduceParentPopulation(unsigned obSize){
  CIndividual** nextGeneration;
  if(obSize==0){
    nextGeneration = new CIndividual*[1];
  }
  else
    nextGeneration = new CIndividual*[obSize];

  reducePopulation(parents,actualParentPopulationSize,nextGeneration,obSize,
       CPopulation::parentReductionOperator,parentReductionPressure);

  // free no longer needed CIndividuals
  for( unsigned i=0 ; i<actualParentPopulationSize-obSize ; i++ )
    delete(parents[i]);
  delete[](parents);

  this->actualParentPopulationSize = obSize;

  parents = nextGeneration;

  return nextGeneration;
}



CIndividual** CPopulation::reduceOffspringPopulation(unsigned obSize){
  // this array has offspringPopulationSize because it will be used as offspring population in
  // the next generation
  CIndividual** nextGeneration = new CIndividual*[offspringPopulationSize];

  reducePopulation(offsprings,actualOffspringPopulationSize,nextGeneration,obSize,
       CPopulation::offspringReductionOperator,offspringReductionPressure);

  //printf("POPULATION SIZE %d\n",actualOffspringPopulationSize-obSize);
  // free no longer needed CIndividuals
  for( unsigned i=0 ; i<actualOffspringPopulationSize-obSize ; i++ )
    delete(offsprings[i]);
  delete[](offsprings);
  //printf("DANS LA FONCTION DE REMPLACEMENT\n");
  /*for(int i=0; i<parentPopulationSize; i++)
  printf("Indiv %d %f | ",i, parents[i]->fitness);
  printf("\n");*/

  this->actualOffspringPopulationSize = obSize;
  offsprings = nextGeneration;
  return nextGeneration;
}


static int CIndividualCompare(const void* p1, const void* p2){
  CIndividual** p1_i = (CIndividual**)p1;
  CIndividual** p2_i = (CIndividual**)p2;

  return p1_i[0]->getFitness() > p2_i[0]->getFitness();
}

static int CIndividualRCompare(const void* p1, const void* p2){
  CIndividual** p1_i = (CIndividual**)p1;
  CIndividual** p2_i = (CIndividual**)p2;

  return p1_i[0]->getFitness() < p2_i[0]->getFitness();
}


void CPopulation::sortPopulation(CIndividual** population, unsigned populationSize){
  qsort(population,populationSize,sizeof(CIndividual*),CIndividualCompare);
}

void CPopulation::sortRPopulation(CIndividual** population, unsigned populationSize){
  qsort(population,populationSize,sizeof(CIndividual*),CIndividualRCompare);
}

/* Fonction qui va serializer la population */
void CPopulation::serializePopulation(){
  ofstream EASEA_File;
  std::string fichier = params->outputFilename;
  fichier.append(".pop");
  EASEA_File.open(fichier.c_str(), ios::app); 
  for(int i=0; (unsigned)i<parentPopulationSize; i++){
  EASEA_File << parents[i]->serialize() << endl;
  
  }
  EASEA_File.close();
}

int CPopulation::getWorstIndividualIndex(CIndividual** population){
  int index=0;
  for(int i=1; i<(signed)this->parentPopulationSize; i++){
    if((params->minimizing && (population[i]->fitness > population[index]->fitness)) || (!params->minimizing && (population[i]->fitness < population[index]->fitness)))
      index=i;
  } 
  return index;
}


/**
   Reduit les populations en faisant l'operation de remplacement.

   @TODO : on aurait voulu eviter la recopie des deux populations en une seule
   mais cela semble incompatible avec CSelectionOperator (notamment l'operation
   d'initialisation.
*/
void CPopulation::reduceTotalPopulation(CIndividual** elitPop){

  CIndividual** nextGeneration = new CIndividual*[parentPopulationSize];

  if(params->elitSize)
  memcpy(nextGeneration,elitPop, sizeof(CIndividual*)*params->elitSize);

  unsigned actualGlobalSize = actualParentPopulationSize+actualOffspringPopulationSize;
  
  CIndividual** globalPopulation = new CIndividual*[actualGlobalSize]();

  if(actualParentPopulationSize==0){
    memcpy(globalPopulation,offsprings,sizeof(CIndividual*)*actualOffspringPopulationSize);
  }
  else if(actualOffspringPopulationSize==0){
    memcpy(globalPopulation,parents,sizeof(CIndividual*)*actualParentPopulationSize);
  }
  else{
    memcpy(globalPopulation,parents,sizeof(CIndividual*)*actualParentPopulationSize);
        memcpy(globalPopulation+actualParentPopulationSize,offsprings,sizeof(CIndividual*)*actualOffspringPopulationSize);
  }


  replacementOperator->initialize(globalPopulation, replacementPressure,actualGlobalSize);

  CPopulation::reducePopulation(globalPopulation,actualGlobalSize,params->elitSize+nextGeneration,
             parentPopulationSize-params->elitSize,replacementOperator,replacementPressure);


  for( unsigned int i=0 ; i<((int)actualGlobalSize+params->elitSize)-(int)parentPopulationSize ; i++ )
    delete(globalPopulation[i]);

  delete[](parents);
  delete[](globalPopulation);

  actualParentPopulationSize = parentPopulationSize;
  actualOffspringPopulationSize = 0;
  parents = nextGeneration;

}


void CPopulation::produceOffspringPopulation(){

  unsigned crossoverArrity = CIndividual::getCrossoverArrity();
  /*
  CIndividual** ps = new CIndividual*[crossoverArrity]();
  */
  CIndividual* p1;
  CIndividual* child;
  int tid;
  int i;
  int startIndex=actualOffspringPopulationSize;
  /*
  CIndividual** p1= new CIndividual*[numThreads]();
  CIndividual* child = new CIndividual*[numThreads]();
  */

  
  selectionOperator->initialize(parents,selectionPressure,actualParentPopulationSize);
  
  #pragma omp parallel private(tid,i,p1,child)
  {
    tid=omp_get_thread_num();
    CRandomGenerator safeRG(CRandomGenerator(selectionOperator->rg->get_seed()+tid));
    CSelectionOperator* safeSelector=selectionOperator->copy(actualParentPopulationSize,&safeRG);
    CIndividual** ps = new CIndividual*[crossoverArrity]();
  
    #pragma omp for schedule(runtime) 
    for(i=0 ; i<offspringPopulationSize ; i++ ){
      unsigned index = safeSelector->selectNext(parentPopulationSize);
      p1 = parents[index];
      //Check if Any Immigrants will reproduce
      if( this->params->remoteIslandModel && parents[index]->isImmigrant ){
          this->cstats->currentNumberOfImmigrantReproductions++;
      }

      if( safeRG.tossCoin(pCrossover) ){
        for( unsigned j=0 ; j<crossoverArrity-1 ; j++ ){
        index = safeSelector->selectNext(parentPopulationSize);
        ps[j] = parents[index];
          if( this->params->remoteIslandModel && parents[index]->isImmigrant ){
              this->cstats->currentNumberOfImmigrantReproductions++;
          }
        }
        child = p1->crossover(ps,&safeRG);
      }
      else child = parents[index]->clone();//new CIndividual(*parents[index]);
      
      
      if( safeRG.tossCoin(pMutation) ){
        child->mutate(pMutationPerGene,&safeRG);
      }
      
      child->boundChecking();

      offsprings[startIndex+i] = child;
    }
    delete[](ps);
    delete safeSelector;
  }
  actualOffspringPopulationSize+=offspringPopulationSize;
}




/**
   Here we save elit CIndividuals to the replacement

   @ARG elitismSize the number of CIndividuals save by elitism
   @ARG population the population where the CIndividuals are save
   @ARG populationSize the size of the population
   @ARG outPopulation the output population, this must be allocated with size greather than elitism
   @ARG outPopulationSize the size of the output population

*/
void CPopulation::strongElitism(unsigned elitismSize, CIndividual** population, unsigned populationSize,
       CIndividual** outPopulation, unsigned outPopulationSize){

  float bestFitness = population[0]->getFitness();
  unsigned bestCIndividual = 0;

#ifndef WIN32
  if( elitismSize >= 5 )DEBUG_PRT("Warning, elitism has O(n) complexity, elitismSize is maybe too big (%d)",elitismSize);
#endif

  //printf("MINIMIZING ? %d\n",params->minimizing);
  for(unsigned i = 0 ; i<elitismSize ; i++ ){
    //bestFitness = replacementOperator->getExtremum();
    bestFitness = population[0]->getFitness();
    bestCIndividual = 0;
    for( unsigned j=0 ; j<populationSize-i ; j++ ){

      if( (params->minimizing && bestFitness > population[j]->getFitness() ) ||
          ( !params->minimizing && bestFitness < population[j]->getFitness() )){
        bestFitness = population[j]->getFitness();
        bestCIndividual = j;
      }
    }
    outPopulation[i] = population[bestCIndividual];
    population[bestCIndividual] = population[populationSize-(i+1)];
    population[populationSize-(i+1)] = NULL;
  }
}

void CPopulation::weakElitism(unsigned elitismSize, CIndividual** parentsPopulation, CIndividual** offspringPopulation, unsigned* parentPopSize, unsigned* offPopSize, CIndividual** outPopulation, unsigned outPopulationSize){

  float bestParentFitness = parentsPopulation[0]->getFitness();
  float bestOffspringFitness = offspringPopulation[0]->getFitness();
  int bestParentIndiv = 0;
  int bestOffspringIndiv = 0;

  for(int i=1; (unsigned)i<(*parentPopSize); i++){
        if( (params->minimizing && bestParentFitness > parentsPopulation[i]->getFitness() ) ||
                        ( !params->minimizing && bestParentFitness < parentsPopulation[i]->getFitness() )){
                bestParentFitness = parentsPopulation[i]->getFitness();
                bestParentIndiv = i;
        }
  }
 
  for(int i=1; (unsigned)i<(*offPopSize); i++){
        if( (params->minimizing && bestOffspringFitness > offspringPopulation[i]->getFitness() ) ||
                        ( !params->minimizing && bestOffspringFitness < offspringPopulation[i]->getFitness() )){
                bestOffspringFitness = offspringPopulation[i]->getFitness();
                bestOffspringIndiv = i;
        }
  }
 
  for(int i = 0 ; (unsigned)i<elitismSize ; i++ ){
  if(((!params->minimizing && bestParentFitness > bestOffspringFitness) || (params->minimizing && bestParentFitness<bestOffspringFitness) || (*offPopSize)==0) && (*parentPopSize)>0){
    outPopulation[i] = parentsPopulation[bestParentIndiv];
    parentsPopulation[bestParentIndiv] = parentsPopulation[(*parentPopSize)-1];
    parentsPopulation[(*parentPopSize)-1] = NULL;
    (*parentPopSize)-=1; 
    if((*parentPopSize)>0){
        bestParentFitness = parentsPopulation[0]->getFitness();
      bestParentIndiv=0;
      for(int j=1; (unsigned)j<(*parentPopSize); j++){
              if( (params->minimizing && bestParentFitness > parentsPopulation[j]->getFitness() ) ||
                            ( !params->minimizing && bestParentFitness < parentsPopulation[j]->getFitness() )){
                      bestParentFitness = parentsPopulation[j]->getFitness();
                      bestParentIndiv = j;
              }
        }
    }
  }
  else{
    outPopulation[i] = offspringPopulation[bestOffspringIndiv];
    offspringPopulation[bestOffspringIndiv] = offspringPopulation[(*offPopSize)-1];
    offspringPopulation[(*offPopSize)-1] = NULL;
    (*offPopSize)-=1;
    if((*offPopSize)>0){
        bestOffspringFitness = offspringPopulation[0]->getFitness();
      bestOffspringIndiv = 0; 
      for(int j=1; (unsigned)j<(*offPopSize); j++){
              if( (params->minimizing && bestOffspringFitness > offspringPopulation[j]->getFitness() ) ||
                            ( !params->minimizing && bestOffspringFitness < offspringPopulation[j]->getFitness() )){
                      bestOffspringFitness = offspringPopulation[j]->getFitness();
                      bestOffspringIndiv = j;
              }
        }
    } 
  }
  }
}


void CPopulation::addIndividualParentPopulation(CIndividual* indiv, unsigned id){
  parents[id] = indiv;
}
void CPopulation::addIndividualParentPopulation(CIndividual* indiv){
  parents[actualParentPopulationSize++] = indiv;
}

std::ostream& operator << (std::ostream& O, const CPopulation& B)
{

  unsigned offspringPopulationSize = B.offspringPopulationSize;
  unsigned realOffspringPopulationSize = B.actualOffspringPopulationSize;

  unsigned parentPopulationSize = B.parentPopulationSize;
  unsigned realParentPopulationSize = B.actualParentPopulationSize;


  O << "CPopulation : "<< std::endl;
  O << "\t Parents size : "<< realParentPopulationSize << "/" <<
    parentPopulationSize << std::endl;

  O << "\t Offspring size : "<< realOffspringPopulationSize << "/" <<
    offspringPopulationSize << std::endl;
  for( unsigned i=0 ; i<realParentPopulationSize ; i++){
  B.parents[i]->printOn(O);
   O << "\n";

  }
  return O;
}

