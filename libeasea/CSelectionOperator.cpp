/*
 * CSelectionOperator.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 *  ====
 *  Updated on 26 july 2022 by Léo Chéneau
 *
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

#include <string>

float getSelectionPressure(std::string const& selectop){
  auto pos1 = selectop.find('(',0)+1;
  auto pos2 = selectop.find(')',0);
  return std::stof(selectop.substr(pos1,pos2-pos1));
}

CSelectionOperator* getSelectionOperator(std::string const& selectop, int minimizing){
  if(minimizing){
  if(selectop.compare("Tournament")==0)
    return (new MinTournament(selectop));
  else if (selectop.compare("Random")==0)
    return (new MinRandom(selectop));
  else if (selectop.compare("Deterministic")==0)
    return (new MinDeterministic(selectop));
  else{
    std::cout << "Operateur n\'existe pas pour minimisation, utilise Tournament par defaut" << std::endl;
    return (new MinTournament("Tournament"));
  }
  }
  else{
  if(selectop.compare("Tournament")==0)
    return (new MaxTournament(selectop));
  else if (selectop.compare("Random")==0)
    return (new MaxRandom(selectop));
  else if (selectop.compare("Deterministic")==0)
    return (new MaxDeterministic(selectop));
  else if (selectop.compare("Roulette")==0)
    return (new MaxRoulette(selectop));
  else{
    std::cout << "Operateur n\'existe pas pour maximisation, utilise Tournament par defaut" << std::endl;
    return (new MaxTournament("Tournament"));
  }
  }
}


/* ****************************************
   SelectionOperator class
****************************************/
void CSelectionOperator::initialize(CIndividual** population, float selectionPressure, size_t){
  this->population = population;
  this->currentSelectionPressure = selectionPressure;
}

size_t CSelectionOperator::selectNext(size_t){ return 0; }

/* ****************************************
   MaxDeterministic class
****************************************/
void MaxDeterministic::initialize(CIndividual** population, float selectionPressure,size_t populationSize){
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
  CPopulation::sortPopulation(population,populationSize);
  this->populationSize = populationSize;
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
  this->populationSize = populationSize;
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
MaxRandom::MaxRandom(std::string const& name_) {
	this->name = name_; // bruh
}

void MaxRandom::initialize(CIndividual** population, float selectionPressure, size_t populationSize){
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
}

size_t MaxRandom::selectNext(size_t populationSize){
  return static_cast<size_t>(random(0,static_cast<int>(populationSize)));
}

float MaxRandom::getExtremum(){
  return -FLT_MAX;
}

/* ****************************************
   MinRandom class
****************************************/
MinRandom::MinRandom(std::string const& name_) {
	this->name = name_; // bruh
}

void MinRandom::initialize(CIndividual** population, float selectionPressure, size_t populationSize){
  CSelectionOperator::initialize(population,selectionPressure,populationSize);
}

size_t MinRandom::selectNext(size_t populationSize){
  return static_cast<size_t>(random(0,static_cast<int>(populationSize)));
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
  if( currentSelectionPressure >= 2.f ){
    for( size_t i = 0 ; static_cast<float>(i) < currentSelectionPressure ; i++ ){
      auto selectedIndex = random(0, static_cast<int>(populationSize));
      //std::cout << selectedIndex << " ";
      float currentFitness = population[selectedIndex]->getFitness();

      if( bestFitness > currentFitness ){
  bestIndex = selectedIndex;
  bestFitness = currentFitness;
      }

    }
  }
  else if( currentSelectionPressure <= 1.f && currentSelectionPressure > 0.f ){
    auto i1 = random(0, static_cast<int>(populationSize));
    auto i2 = random(0, static_cast<int>(populationSize));

    if( tossCoin(currentSelectionPressure) ){
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
  if( currentSelectionPressure >= 2.f ){
    for( size_t i = 0 ; static_cast<float>(i) < currentSelectionPressure ; i++ ){
      auto selectedIndex = random(0, static_cast<int>(populationSize));;
      //std::cout << selectedIndex << " ";
      float currentFitness = population[selectedIndex]->getFitness();

      if( bestFitness < currentFitness ){
  bestIndex = selectedIndex;
  bestFitness = currentFitness;
      }

    }
  }
  else if( currentSelectionPressure <= 1.f && currentSelectionPressure > 0.f ){
    auto i1 = random(0, static_cast<int>(populationSize));;
    auto i2 = random(0, static_cast<int>(populationSize));;

    if( tossCoin(currentSelectionPressure) ){
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
  this->populationSize = populationSize;
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
        poidsSelectionne = random(0.f, 1.f);
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

