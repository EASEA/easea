/*
 * Parameters.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "CEvolutionaryAlgorithm.h"
class CGenerationalCriterion;
class CSelectionOperator;

class Parameters {
public:
	CSelectionOperator* selectionOperator;
	CSelectionOperator* replacementOperator;
	CSelectionOperator* parentReductionOperator;
	CSelectionOperator* offspringReductionOperator;
	CGenerationalCriterion* generationalCriterion;

	float selectionPressure;
	float replacementPressure;
	float parentReductionPressure;
	float offspringReductionPressure;

	float pCrossover;
	float pMutation;
	float pMutationPerGene;
	CRandomGenerator* randomGenerator;

	time_t seed;

	unsigned int parentPopulationSize;
	unsigned int offspringPopulationSize;
	bool minimizing;

	unsigned int offspringReductionSize;
	unsigned int parentReductionSize;
	bool offspringReduction;
	bool parentReduction ;

	bool strongElitism;
	unsigned int elitSize;

	int printStats;
	int plotStats;
	int printInitialPopulation;
	int printFinalPopulation;

public:
#ifdef WIN32
	Parameters();
	~Parameters();
#endif
	virtual void setDefaultParameters(int argc, char** argv) = 0;
	virtual CEvolutionaryAlgorithm* newEvolutionaryAlgorithm() = 0;
};


#endif /* PARAMETERS_H_ */
