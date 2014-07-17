/*
 * Parameters.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */
/**
 * @file Parameters.h
 * @author SONIC BFO, Ogier Maitre 
 * @version 1.0
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation; either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Affero General Public License for more details at
 * http://www.gnu.org/licenses/
**/  

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "CEvolutionaryAlgorithm.h"
class CGenerationalCriterion;
class CControlCStopingCriterion;
class CSelectionOperator;

/**
*  \class     Parameters 
*  \brief     A class for passing various parametr to the main EA loop
*  \details   The contents of this class are set by 1) tpl files 
*                                                   2) *.prm at launch and 
*                                                   3) command line arguments.
*  
**/
class Parameters {
  
  public:

    /*TODO: Verify if this distinction is still needed on g++-mingw */
    /*Constructors/Destructors----------------------------------------------------*/
    #ifdef WIN32
      Parameters();
      ~Parameters();
    #else
      virtual ~Parameters(){;}
    #endif
    
    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Initialization from arguments
    * \details  By using the standard C way. Defined in tpl files.
    *
    *  @param   argc  Number of arguments
    *  @param   argv  Array of argument as char*
    **/
    virtual void setDefaultParameters(int argc, char** argv) = 0;
    

    /**
    * \brief    Instanciate a new CEvolutionaryAlgorithm using current variables
    * \details  Defined in tpl files.
    *
    * @return   var   The new EA 
    **/
    virtual CEvolutionaryAlgorithm* newEvolutionaryAlgorithm() = 0;


    /**
    * \brief    Set reduction size
    * \details  Do correctness checking 
    *
    *  @param   popReducSize A percentage, represented by a float between 0 and 1
    *  @param   popSize      The size to be reduced
    * @return   var          The resulting size
    **/
    int setReductionSizes(int popSize, float popReducSize);
  
  public:
    /*Datas-----------------------------------------------------------------------*/
    //Reduction and selection operator
    CSelectionOperator* selectionOperator;
    CSelectionOperator* replacementOperator;
    CSelectionOperator* parentReductionOperator;
    CSelectionOperator* offspringReductionOperator;
    
    //Stopping criterion
    CGenerationalCriterion* generationalCriterion;
    CControlCStopingCriterion* controlCStopingCriterion;
    CTimeCriterion* timeCriterion;

    int nbGen;

    //Reduction and selection pressures
    float selectionPressure;
    float replacementPressure;
    float parentReductionPressure;
    float offspringReductionPressure;

    //Genetic operators parameters
    float pCrossover;
    float pMutation;
    float pMutationPerGene;
    CRandomGenerator* randomGenerator;
    
    //PRNG seed
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
    
    //GPU selection
    int fstGpu;
    int lstGpu;

};

#endif /* PARAMETERS_H_ */
