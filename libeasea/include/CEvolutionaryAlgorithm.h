/*
 * CEvolutionaryAlgorithm.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CEVOLUTIONARYALGORITHM_H_
#define CEVOLUTIONARYALGORITHM_H_
#include <cstdlib>
#include <string>
#include <chrono>
#include "CSelectionOperator.h"
#include "CPopulation.h"
#include "CStoppingCriterion.h"
#include "CComUDPLayer.h"
#include "CStats.h"

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
  std::unique_ptr<CComUDPServer> server;
  std::vector<std::unique_ptr<CComUDPClient>> Clients;
  void initializeClients();
  void receiveIndividuals();
  void sendIndividual();
  void refreshClient();


  void showPopulationStats(std::chrono::time_point<std::chrono::system_clock> const& beginTime);
  void generatePlotScript();
  void generateRScript();

  void outputGraph();
  Parameters* params;

  CGrapher* grapher;

  CStats* cstats;

  virtual ~CEvolutionaryAlgorithm();

  std::vector<CStoppingCriterion*> stoppingCriteria;
};

#endif /* CEVOLUTIONARYALGORITHM_H_ */
