/*
 * CSelectionOperator.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CSelectionOperator.h"
#ifdef WIN32
#include <float.h>
#elif __APPLE__
#include <float.h>
#else
#include <values.h>
#endif
#include "include/CPopulation.h"
#include "include/CRandomGenerator.h"

float getSelectionPressure(std::string selectop){
  int pos1 = selectop.find('(',0)+1;
  int pos2 = selectop.find(')',0);
  std::string press = selectop.substr(pos1,pos2-pos1);
  return (float)atof(press.c_str());
}

CSelectionOperator* getSelectionOperator(std::string selectop, int minimizing, CRandomGenerator* globalRandomGenerator){
  if(minimizing){
	if(selectop.compare("Tournament")==0)
		return (new MinTournament(globalRandomGenerator));
	else if (selectop.compare("Random")==0)
		return (new MinRandom(globalRandomGenerator));
	else if (selectop.compare("Deterministic")==0)
		return (new MinDeterministic());
	else{
		std::cout << "Operateur n\'existe pas pour minimisation, utilise Tournament par defaut" << std::endl;
		return (new MinTournament(globalRandomGenerator));
	}
  }
  else{
	if(selectop.compare("Tournament")==0)
		return (new MaxTournament(globalRandomGenerator));
	else if (selectop.compare("Random")==0)
		return (new MaxRandom(globalRandomGenerator));
	else if (selectop.compare("Deterministic")==0)
		return (new MaxDeterministic());
	else if (selectop.compare("Roulette")==0)
		return (new MaxRoulette(globalRandomGenerator));
	else{
		std::cout << "Operateur n\'existe pas pour maximisation, utilise Tournament par defaut" << std::endl;
		return (new MaxTournament(globalRandomGenerator));
	}
  }
}


/* ****************************************
   SelectionOperator class
****************************************/
void CSelectionOperator::initialize(CIndividual** population, float selectionPressure, size_t populationSize){
  this->population = population;
  this->currentSelectionPressure = selectionPressure;
}

size_t CSelectionOperator::selectNext(size_t populationSize){ return 0; }

/* ****************************************
   MaxDeterministic class
****************************************/
void MaxDeterministic::initialize(CIndividual** population, float selectionPressure,size_t populationSize){
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
  CPopulation::sortPopulation(population,populationSize);
  populationSize = populationSize;
}

size_t MaxDeterministic::selectNext(size_t populationSize){
  return populationSize-1;
}

float MaxDeterministic::getExtremum(){
  return -FLT_MAX;
}

/* ****************************************
   MinDeterministic class
****************************************/
void MinDeterministic::initialize(CIndividual** population, float selectionPressure,size_t populationSize){
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
  CPopulation::sortRPopulation(population,populationSize);
  populationSize = populationSize;
}

size_t MinDeterministic::selectNext(size_t populationSize){
  return populationSize-1;
}

float MinDeterministic::getExtremum(){
  return FLT_MAX;
}

/* ****************************************
   MaxRandom class
****************************************/
MaxRandom::MaxRandom(CRandomGenerator* globalRandomGenerator){
  rg = globalRandomGenerator;
}

void MaxRandom::initialize(CIndividual** population, float selectionPressure, size_t populationSize){
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
}

size_t MaxRandom::selectNext(size_t populationSize){
  return rg->random(0,populationSize-1);
}

float MaxRandom::getExtremum(){
  return -FLT_MAX;
}

/* ****************************************
   MinRandom class
****************************************/
MinRandom::MinRandom(CRandomGenerator* globalRandomGenerator){
  rg = globalRandomGenerator;
}

void MinRandom::initialize(CIndividual** population, float selectionPressure, size_t populationSize){
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
}

size_t MinRandom::selectNext(size_t populationSize){
  return rg->random(0,populationSize-1);
}

float MinRandom::getExtremum(){
  return -FLT_MAX;
}


/* ****************************************
   MinTournament class
****************************************/
void MinTournament::initialize(CIndividual** population, float selectionPressure, size_t populationSize) {
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
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
   MaxTournament class
****************************************/
void MaxTournament::initialize(CIndividual** population, float selectionPressure, size_t populationSize) {
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
}

float MaxTournament::getExtremum(){
  return -FLT_MAX;
}

size_t MaxTournament::selectNext(size_t populationSize){
  size_t bestIndex = 0;
  float bestFitness = -FLT_MAX;

  //std::cout << "MinTournament selection " ;
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
    std::cerr << " MinTournament selection operator doesn't handle selection pressure : "
	      << currentSelectionPressure << std::endl;
  }

  //std::cout << std::endl;
  return bestIndex;
}

/*****************************************
 *    MaxRoulette class
 *****************************************/
void MaxRoulette::initialize(CIndividual** population, float selectionPressure, size_t populationSize) {
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
  CPopulation::sortRPopulation(population,populationSize);
  populationSize = populationSize;
}

size_t MaxRoulette::selectNext(size_t populationSize){
        size_t bestIndex = 0;
        float poidsTotal = 0.;
        float poidsCourant = 0.;
        float poidsSelectionne;
        float *poids = (float*)malloc(populationSize*sizeof(float));
        int i;

        for(i=0; (unsigned)i<populationSize; i++){
                poidsTotal += population[i]->getFitness();
        }
        for(i=0; (unsigned)i<populationSize; i++){
                poidsCourant += (population[i]->getFitness()/poidsTotal);
                poids[i] = poidsCourant;
        }
        poidsSelectionne = rg->randFloat(0.0,1.0);
        for(i=0; (unsigned)i<populationSize; i++){
                if(poidsSelectionne<poids[i]){
                        bestIndex = i;
                        break;
                }
        }
		free(poids);
        return bestIndex;
}

float MaxRoulette::getExtremum(){
  return -FLT_MAX;
}

