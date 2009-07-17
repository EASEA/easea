/*
 * CSelectionOperator.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CSelectionOperator.h"
#ifndef WIN32
#include <values.h>
#else
#include <float.h>
#endif
#include "include/CPopulation.h"
#include "include/CRandomGenerator.h"


/* ****************************************
   SelectionOperator class
****************************************/
void CSelectionOperator::initialize(CIndividual** population, float selectionPressure, size_t populationSize){
  this->population = population;
  this->currentSelectionPressure = selectionPressure;
}

size_t CSelectionOperator::selectNext(size_t populationSize){ return 0; }



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
