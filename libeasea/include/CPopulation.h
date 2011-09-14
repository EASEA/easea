/*
 * CPopulation.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CPOPULATION_H_
#define CPOPULATION_H_


#ifdef DEBUG
#define DEBUG_PRT(format, args...) fprintf (stdout,"***DBG***  %s-%d: "format"\n",__FILE__,__LINE__,##args)
#define DEBUG_YACC(format, args...) fprintf (stdout,"***DBG_YACC***  %s-%d: "format"\n",__FILE__,__LINE__,##args)
#else
#ifndef WIN32
#define DEBUG_PRT(format, args...)
#define DEBUG_YACC(format, args...)
#endif
#endif

#include <vector>
#include "CSelectionOperator.h"

class Parameters;
class CStats;

class CPopulation {

public:

  float pCrossover;
  float pMutation;
  float pMutationPerGene;

  CIndividual* Best;
  CIndividual* Worst;
  float currentAverageFitness;
  float currentSTDEV;

  CIndividual** parents;
  CIndividual** offsprings;

  unsigned parentPopulationSize;
  unsigned offspringPopulationSize;

  unsigned actualParentPopulationSize;
  unsigned actualOffspringPopulationSize;

  static CSelectionOperator* selectionOperator;
  static CSelectionOperator* replacementOperator;
  static CSelectionOperator* parentReductionOperator;
  static CSelectionOperator* offspringReductionOperator;

  unsigned currentEvaluationNb;
  CRandomGenerator* rg;
  std::vector<CIndividual*> pop_vect;

  Parameters* params;
  CStats* cstats;

 public:
  CPopulation();
  CPopulation(unsigned parentPopulationSize, unsigned offspringPopulationSize,
	     float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params, CStats* cstats);
  virtual ~CPopulation();

  //virtual void initializeParentPopulation() = 0;
  void addIndividualParentPopulation(CIndividual* indiv, unsigned id);
  void addIndividualParentPopulation(CIndividual* indiv);
  void evaluatePopulation(CIndividual** population, unsigned populationSize);
  virtual void optimisePopulation(CIndividual** population, unsigned populationSize);
  virtual void evaluateParentPopulation();
  virtual void optimiseParentPopulation();

  void strongElitism(unsigned elitismSize, CIndividual** population, unsigned populationSize, CIndividual** outPopulation, unsigned outPopulationSize);
  void weakElitism(unsigned elitismSize, CIndividual** parentsPopulation, CIndividual** offspringPopulation, unsigned* parentPopSize, unsigned* offPopSize, CIndividual** outPopulation, unsigned outPopulationSize);

  virtual void evaluateOffspringPopulation();
  virtual void optimiseOffspringPopulation();
  CIndividual** reducePopulations(CIndividual** population, unsigned populationSize,
			       CIndividual** reducedPopulation, unsigned obSize);
  CIndividual** reduceParentPopulation(unsigned obSize);
  CIndividual** reduceOffspringPopulation(unsigned obSize);
  void reduceTotalPopulation(CIndividual** elitPop);
  void evolve();

  static float selectionPressure;
  static float replacementPressure;
  static float parentReductionPressure;
  static float offspringReductionPressure;

  static void initPopulation(CSelectionOperator* selectionOperator,
			     CSelectionOperator* replacementOperator,
			     CSelectionOperator* parentReductionOperator,
			     CSelectionOperator* offspringReductionOperator,
			     float selectionPressure, float replacementPressure,
			     float parentReductionPressure, float offspringReductionPressure);

  static void sortPopulation(CIndividual** population, unsigned populationSize);

  static void sortRPopulation(CIndividual** population, unsigned populationSize);

  void serializePopulation();
  int getWorstIndividualIndex(CIndividual** population);

  void sortParentPopulation(){ CPopulation::sortPopulation(parents,actualParentPopulationSize);}

  virtual void produceOffspringPopulation();

  friend std::ostream& operator << (std::ostream& O, const CPopulation& B);


  void setParentPopulation(CIndividual** population, unsigned actualParentPopulationSize){
    this->parents = population;
    this->actualParentPopulationSize = actualParentPopulationSize;
  }

  static void reducePopulation(CIndividual** population, unsigned populationSize,
				       CIndividual** reducedPopulation, unsigned obSize,
				       CSelectionOperator* replacementOperator);
  void syncOutVector();
  void syncInVector();

};

#endif /* CPOPULATION_H_ */
