/*
 * Parameters.h
 *    Copyright (C) 2009  Ogier Maitre

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
	/*----------------------*/
	int IndgenerateCSVFile;
	int generateTXTFileGen;
	int generateGenomeFile;
	/*---------------------*/
	int generatePlotScript;
	int generateRScript;	
	int plotStats;
	int printInitialPopulation;
	int printFinalPopulation;

	bool savePopulation;
	bool startFromFile;

	//Parameters for the Island Model
	
	// socket model
	bool remoteIslandModel;
	char* ipFile;
	float migrationProbability;
	int serverPort;
        int worker_number;

	// fileserver model
	char *expId;
	char *working_path;
	
	char* outputFilename;
	char* plotOutputFilename;

	int fstGpu;
	int lstGpu;

public:
#ifdef WIN32
	Parameters();
	~Parameters();
#else
	virtual ~Parameters(){;
	   delete selectionOperator;
	   delete replacementOperator;
	   delete parentReductionOperator;
	   delete offspringReductionOperator;

	   delete generationalCriterion;
	   delete controlCStopingCriterion;
	   delete timeCriterion;
	   delete randomGenerator;
	  
	
	}
#endif
	virtual void setDefaultParameters(int argc, char** argv) = 0;
	virtual CEvolutionaryAlgorithm* newEvolutionaryAlgorithm() = 0;
	int setReductionSizes(int popSize, float popReducSize);
};

#endif /* PARAMETERS_H_ */
