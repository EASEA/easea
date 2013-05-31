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


/*CONSTRUCTEUR DE BASE, ON PASSE LES ATTRIBUTS EN ARGUMENT */
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




/* Met les éléments du pop vect dans la pop parent */
void CPopulation::syncInVector(){
  for( unsigned i = 0 ; i<actualParentPopulationSize ; i++ ){
    parents[i] = pop_vect.at(i);
  }
}


/* Vide pop_vect et met les éléments de la pop parent dedans */
void CPopulation::syncOutVector(){
   pop_vect.clear();
  for( unsigned i = 0 ; i<actualParentPopulationSize ; i++ ){
    pop_vect.push_back(parents[i]);
  }
#ifndef WIN32
  DEBUG_PRT("Size of outVector",pop_vect.size());
#endif
}


/*   Destructeur   */
CPopulation::~CPopulation(){
  for( unsigned i=0 ; i<actualOffspringPopulationSize ; i++ ) delete(offsprings[i]);
  for( unsigned i=0 ; i<actualParentPopulationSize ; i++ )    delete(parents[i]);

  delete[](this->parents);
  delete[](this->offsprings);
}


/* Initialise les opérateurs et certains paramètres de l'EA */
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



/* évalue une population */
void CPopulation::evaluatePopulation(CIndividual** population, unsigned populationSize){
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
					  CSelectionOperator* replacementOperator){


  replacementOperator->initialize(population,replacementPressure,populationSize);

  for( unsigned i=0 ; i<obSize ; i++ ){

    // select an CIndividual and add it to the reduced population
    unsigned selectedIndex = replacementOperator->selectNext(populationSize - i);
    // std::cout << "Selected " << selectedIndex << "/" << populationSize
    // 	      << " replaced by : " << populationSize-(i+1)<< std::endl;
    reducedPopulation[i] = population[selectedIndex];
    //printf("TEST REMPLACEMENT %d %d %f %f\n", i, selectedIndex, reducedPopulation[i]->fitness, population[selectedIndex]->fitness);

    // erase it to the std population by swapping last CIndividual end current
    population[selectedIndex] = population[populationSize-(i+1)];
    //population[populationSize-(i+1)] = NULL;
  }

  //return reducedPopulation;
}


CIndividual** CPopulation::reduceParentPopulation(unsigned obSize){
  CIndividual** nextGeneration;
  if(obSize==0){
  	nextGeneration = new CIndividual*[1];
  }
  else
  	nextGeneration = new CIndividual*[obSize];

  reducePopulation(parents,actualParentPopulationSize,nextGeneration,obSize,
		   CPopulation::parentReductionOperator);

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
		   CPopulation::offspringReductionOperator);

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
			       parentPopulationSize-params->elitSize,replacementOperator);


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
  CIndividual* p1;
  CIndividual** ps = new CIndividual*[crossoverArrity]();
  CIndividual* child;

  selectionOperator->initialize(parents,selectionPressure,actualParentPopulationSize);

  for( unsigned i=0 ; i<offspringPopulationSize ; i++ ){
    unsigned index = selectionOperator->selectNext(parentPopulationSize);
    p1 = parents[index];

    //Check if Any Immigrants will reproduce
    if( this->params->remoteIslandModel && parents[index]->isImmigrant ){
        if(this->cstats!=NULL)this->cstats->currentNumberOfImmigrantReproductions++;
	else std::cout << "Cstats is null";
      
    }

    if( rg->tossCoin(pCrossover) ){
      for( unsigned j=0 ; j<crossoverArrity-1 ; j++ ){
	    index = selectionOperator->selectNext(parentPopulationSize);
	    ps[j] = parents[index];
        if( this->params->remoteIslandModel && parents[index]->isImmigrant ){
            if(this->cstats!=NULL)this->cstats->currentNumberOfImmigrantReproductions++;
	    else std::cout << "Cstats is null"; 
        }
      }
      child = p1->crossover(ps);
      /*-------------------*/
     child->origine = 'C';
     child->parent1 = (p1->ident);
     child->parent2 = (ps[0]->ident);
     child->survival = 0;
     child->gainFitness= -((p1->fitness) + (ps[0]->fitness))/(float)2;
      /*------------------*/
    }
    else  {child = parents[index]->clone();//new CIndividual(*parents[index]);
      /*-------------------*/
      child->parent1 = (parents[index]->ident);
      child->survival=0;
      child->gainFitness= -(p1->fitness);
      /*-------------------*/
    }

    if( rg->tossCoin(pMutation) ){
      child->mutate(pMutationPerGene);
       /*-------------------*/
      if(child->origine == 'C')child->origine = 'B'; 
      else child->origine = 'M';
      /*------------------*/
    }
    /*-------------------*/
    if (((child->origine) != 'C')&&((child->origine) != 'B')&&((child->origine) != 'M')){child->origine = 'N'; }
    /*------------------*/

    child->boundChecking();

    offsprings[actualOffspringPopulationSize++] = child;
  }
  delete[](ps);
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

