/*
 * CEvolutionaryAlgorithm.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CEVOLUTIONARYALGORITHM_H_
#define CEVOLUTIONARYALGORITHM_H_
#include <stdlib.h>
#include <string>
#include <time.h>
#include "CEvolutionaryAlgorithm.h"
#include "CSelectionOperator.h"
#include "CPopulation.h"
#include "CStoppingCriterion.h"
#ifdef WIN32
#include <windows.h>
#endif

class Parameters;
class CGnuplot;

class CEvolutionaryAlgorithm {
public:

  CEvolutionaryAlgorithm( size_t parentPopulationSize,
			 size_t offspringPopulationSize,
			 float selectionPressure, float replacementPressure, float parentReductionPressure, float offspringReductionPressure,
			 CSelectionOperator* selectionOperator, CSelectionOperator* replacementOperator,
			 CSelectionOperator* parentReductionOperator, CSelectionOperator* offspringReductionOperator,
			 float pCrossover, float pMutation,
			 float pMutationPerGene);

  CEvolutionaryAlgorithm( Parameters* params );
  virtual void initializeParentPopulation() = 0;

  size_t* getCurrentGenerationPtr(){ return &currentGeneration;}
  void addStoppingCriterion(CStoppingCriterion* sc);
  void runEvolutionaryLoop();
  bool allCriteria();
  CPopulation* getPopulation(){ return population;}
  size_t getCurrentGeneration() { return currentGeneration;}
public:
  size_t currentGeneration;
  CPopulation* population;
  size_t reduceParents;
  size_t reduceOffsprings;

#ifdef WIN32
  void showPopulationStats(clock_t beginTime);
#else
  void showPopulationStats(struct timeval beginTime);
#endif
  void generateGnuplotScript();
  void generateRScript();

  void outputGraph();
  Parameters* params;

  CGnuplot* gnuplot;
  float currentAverageFitness;
  float currentSTDEV;


  std::vector<CStoppingCriterion*> stoppingCriteria;

  std::string* outputfile;
  std::string* inputfile;
};

#endif /* CEVOLUTIONARYALGORITHM_H_ */
