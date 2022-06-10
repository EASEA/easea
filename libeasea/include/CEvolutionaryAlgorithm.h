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
#include "CSelectionOperator.h"
#include "CPopulation.h"
#include "CStoppingCriterion.h"
#include "CComUDPLayer.h"
#include "CStats.h"
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

class Parameters;
class CGrapher;

class CEvolutionaryAlgorithm {
public:

 /* CEvolutionaryAlgorithm( size_t parentPopulationSize,
			 size_t offspringPopulationSize,
			 float selectionPressure, float replacementPressure, float parentReductionPressure, float offspringReductionPressure,
			 CSelectionOperator* selectionOperator, CSelectionOperator* replacementOperator,
			 CSelectionOperator* parentReductionOperator, CSelectionOperator* offspringReductionOperator,
			 float pCrossover, float pMutation,
			 float pMutationPerGene);*/

    void initLogger();
  CEvolutionaryAlgorithm( Parameters* params );
  virtual void initializeParentPopulation() = 0;

  unsigned int *getCurrentGenerationPtr(){ return &currentGeneration;}
  void addStoppingCriterion(CStoppingCriterion* sc);
  virtual void runEvolutionaryLoop();
  bool allCriteria();
  CPopulation* getPopulation(){ return population;}
  unsigned getCurrentGeneration() { return currentGeneration;}
public:
  unsigned currentGeneration;
  CPopulation* population;
  unsigned reduceParents;
  unsigned reduceOffsprings;

  //methods and variables for remote island model
  unsigned treatedIndividuals;
  unsigned numberOfClients;
  unsigned myClientNumber;
  CComUDPServer *server;
  std::vector<std::unique_ptr<CComUDPClient>> Clients;
  void initializeClients();
  void receiveIndividuals();
  void sendIndividual();
  void refreshClient();

#ifdef WIN32
  void showPopulationStats(clock_t beginTime);
#else
  void showPopulationStats(struct timeval beginTime);
#endif
  void generatePlotScript();
  void generateRScript();

  void outputGraph();
  Parameters* params;

  CGrapher* grapher;

  CStats* cstats;

  virtual ~CEvolutionaryAlgorithm();

  std::vector<CStoppingCriterion*> stoppingCriteria;

  std::string* outputfile;
  std::string* inputfile;
};

#endif /* CEVOLUTIONARYALGORITHM_H_ */
