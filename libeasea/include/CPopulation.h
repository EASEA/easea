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

class CPopulation {

public:

  float pCrossover;
  float pMutation;
  float pMutationPerGene;

  CIndividual* Best;

  CIndividual** parents;
  CIndividual** offsprings;

  size_t parentPopulationSize;
  size_t offspringPopulationSize;

  size_t actualParentPopulationSize;
  size_t actualOffspringPopulationSize;

  static CSelectionOperator* selectionOperator;
  static CSelectionOperator* replacementOperator;
  static CSelectionOperator* parentReductionOperator;
  static CSelectionOperator* offspringReductionOperator;

  size_t currentEvaluationNb;
  CRandomGenerator* rg;
  std::vector<CIndividual*> pop_vect;

  Parameters* params;

 public:
  CPopulation();
  CPopulation(size_t parentPopulationSize, size_t offspringPopulationSize,
	     float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params);
  virtual ~CPopulation();

  //virtual void initializeParentPopulation() = 0;
  void addIndividualParentPopulation(CIndividual* indiv);
  void evaluatePopulation(CIndividual** population, size_t populationSize);
  virtual void evaluateParentPopulation();

  void strongElitism(size_t elitismSize, CIndividual** population, size_t populationSize, CIndividual** outPopulation, size_t outPopulationSize);
  void weakElitism(size_t elitismSize, CIndividual** parentsPopulation, CIndividual** offspringPopulation, size_t* parentPopSize, size_t* offPopSize, CIndividual** outPopulation, size_t outPopulationSize);

  virtual void evaluateOffspringPopulation();
  CIndividual** reducePopulations(CIndividual** population, size_t populationSize,
			       CIndividual** reducedPopulation, size_t obSize);
  CIndividual** reduceParentPopulation(size_t obSize);
  CIndividual** reduceOffspringPopulation(size_t obSize);
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

  static void sortPopulation(CIndividual** population, size_t populationSize);

  static void sortRPopulation(CIndividual** population, size_t populationSize);


  void sortParentPopulation(){ CPopulation::sortPopulation(parents,actualParentPopulationSize);}

  void produceOffspringPopulation();

  friend std::ostream& operator << (std::ostream& O, const CPopulation& B);


  void setParentPopulation(CIndividual** population, size_t actualParentPopulationSize){
    this->parents = population;
    this->actualParentPopulationSize = actualParentPopulationSize;
  }

  static void reducePopulation(CIndividual** population, size_t populationSize,
				       CIndividual** reducedPopulation, size_t obSize,
				       CSelectionOperator* replacementOperator);
  void syncOutVector();
  void syncInVector();

};

#endif /* CPOPULATION_H_ */
