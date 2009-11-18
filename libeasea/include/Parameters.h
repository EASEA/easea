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
class CControlCStopingCriterion;
class CSelectionOperator;

class Parameters {
public:
	CSelectionOperator* selectionOperator;
	CSelectionOperator* replacementOperator;
	CSelectionOperator* parentReductionOperator;
	CSelectionOperator* offspringReductionOperator;

	CGenerationalCriterion* generationalCriterion;
	CControlCStopingCriterion* controlCStopingCriterion;
	CTimeCriterion* timeCriterion;

	int nbGen;

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
	int generateCVSFile;
	int generateGnuplotScript;
	int generateRScript;	
	int plotStats;
	int printInitialPopulation;
	int printFinalPopulation;

	char* outputFilename;
	char* plotOutputFilename;

public:
#ifdef WIN32
	Parameters();
	~Parameters();
#endif
	virtual void setDefaultParameters(int argc, char** argv) = 0;
	virtual CEvolutionaryAlgorithm* newEvolutionaryAlgorithm() = 0;
	int setReductionSizes(int popSize, float popReducSize);
};

#endif /* PARAMETERS_H_ */
