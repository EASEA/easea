/*
 * Parameters.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <string>
#include <memory>

#include "COptionParser.h"

class CGenerationalCriterion;
class CControlCStopingCriterion;
class CSelectionOperator;
class CRandomGenerator;
class CEvolutionaryAlgorithm;
class CTimeCriterion;

class Parameters {
  protected:
    std::unique_ptr<vm_t> vm;

  public:
    CSelectionOperator* selectionOperator;
    CSelectionOperator* replacementOperator;
    CSelectionOperator* parentReductionOperator;
    CSelectionOperator* offspringReductionOperator;

    CGenerationalCriterion* generationalCriterion;
    CControlCStopingCriterion* controlCStopingCriterion;
    CTimeCriterion* timeCriterion;

    int nbGen;
    int nbCPUThreads;
    bool noLogFile;
    bool reevaluateImmigrants;

    bool alwaysEvaluate;

    float selectionPressure;
    float replacementPressure;
    float parentReductionPressure;
    float offspringReductionPressure;

    //Genetic operators parameters
    float pCrossover;
    float pMutation;
    float pMutationPerGene;
    CRandomGenerator* randomGenerator;

    int seed;

    unsigned int parentPopulationSize;
    unsigned int offspringPopulationSize;
    bool minimizing;

    unsigned int offspringReductionSize;
    unsigned int archivePopulationSize;
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
    bool printStats;
    bool generateCSVFile;
    bool generatePlotScript;
    bool generateRScript;
    bool plotStats;
    bool printInitialPopulation;
    bool printFinalPopulation;

    bool savePopulation;
    bool startFromFile;
    std::string outputFilename;
    std::string inputFilename;

    //Parameters for the Island Model
    bool remoteIslandModel;
    std::string ipFile;
    float migrationProbability;
    int serverPort;
    bool silentNetwork;

    char* plotOutputFilename;

    int fstGpu;
    int lstGpu;

    std::string u1;
    std::string u2;
    std::string u3;
    std::string u4;
    std::string u5;

  public:
    Parameters(std::string const& filename, int argc, char* argv[]);
    virtual ~Parameters();
    virtual CEvolutionaryAlgorithm* newEvolutionaryAlgorithm() = 0;
    int setReductionSizes(int popSize, float popReducSize);
    int getOffspringSize(int defaut_value, int parent_size) const;

    template <typename T>
    auto setVariable(std::string const& argumentName, T&& defaultValue) const {
	    return ::setVariable(argumentName, std::forward<T>(defaultValue), vm);
    }
};

#endif /* PARAMETERS_H_ */
