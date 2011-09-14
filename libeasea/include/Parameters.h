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

	//Genetic operators parameters
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

	//Elitism parameters
	bool strongElitism;
	unsigned int elitSize;

	//Parameters for memetic algorithm
	bool optimise;
	int optimiseIterations;
	bool baldwinism;

	//Miscalleous parameters
	int printStats;
	int generateCSVFile;
	int generatePlotScript;
	int generateRScript;	
	int plotStats;
	int printInitialPopulation;
	int printFinalPopulation;

	bool savePopulation;
	bool startFromFile;

	//Parameters for the Island Model
	bool remoteIslandModel;
	char* ipFile;
	float migrationProbability;
    int serverPort;

	char* outputFilename;
	char* plotOutputFilename;

public:
#ifdef WIN32
	Parameters();
	~Parameters();
#else
	virtual ~Parameters(){;}
#endif
	virtual void setDefaultParameters(int argc, char** argv) = 0;
	virtual CEvolutionaryAlgorithm* newEvolutionaryAlgorithm() = 0;
	int setReductionSizes(int popSize, float popReducSize);
};

#endif /* PARAMETERS_H_ */
