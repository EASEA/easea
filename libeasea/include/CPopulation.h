/*
 * CPopulation.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */
/**
 * @file CPopulation.h
 * @author SONIC BFO, Ogier Maitre, Pallamidessi Joseph
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
    /*Constructors/Destructors----------------------------------------------------*/
     /**
     * \brief    Constructor of CPopulation
     * \details  Empty constructor. Do not use directly without initialization
     *
     **/
    CPopulation();
    
    /**
    * \brief    Constructor of CPopulation
    * \details  The main constructor.
    *
    * @param    parentPopulationSize    Size of the parent population
    * @param    offspringPopulationSize Size of the offspring population
    * @param    pCrossover              Crossover probabilities
    * @param    pMutation               Mutation probabilities
    * @param    pMutationPerGene        Mutation probabilities per gene
    * @param    rg                      PRNG (currently a MT19973)
    * @param    params                  Parameters of the genetic algorithm
    * @param    cstats                  A statistic tracking object
    **/
    CPopulation(unsigned parentPopulationSize, unsigned offspringPopulationSize,
                float pCrossover, float pMutation, float pMutationPerGene, 
                CRandomGenerator* rg, Parameters* params, CStats* cstats);
    
    /**
    * \brief    Destructor of CPopulation.
    * \details  
    *
    **/
    virtual ~CPopulation();
    

    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Add an individual at a specified index  
    * \details  Only used on population initialization, in tpl files;
    *
    * @param    indiv  The individual to add
    * @param    id     Index in the population array
    **/
    void addIndividualParentPopulation(CIndividual* indiv, unsigned id);
    
    /**
    * \brief    Add an individual at the end 
    * \details  Only used on population initialization, in tpl files;
    *
    *  @param   indiv   The individual to add
    **/
    void addIndividualParentPopulation(CIndividual* indiv);
    
    /**
    * \brief    Evaluation the population
    * \details  use openMP to parallelize thze evaluation
    *
    * @param    population      The population to evaluate
    * @param    populationSize  The size of the population to evaluate
    **/
    void evaluatePopulation(CIndividual** population, unsigned populationSize);
    
    /**
    * \brief    Evaluate the parent population
    * \details  Wrapper for evaluatePopulation
    *
    **/
    virtual void evaluateParentPopulation();
    
    /**
    * \brief    Evaluate the parent population
    * \details  Wrapper for evaluatePopulation
    *
    **/
    virtual void evaluateOffspringPopulation();
    
    /**
    * \brief    Optimise the population
    * \details  Do nothing except when redefined using memetic template
    *
    * @param    population      The population to optimise
    * @param    populationSize  The size of the population to optimise
    **/
    virtual void optimisePopulation(CIndividual** population, unsigned populationSize);
    
    /**
    * \brief    Optimise the parent population
    * \details  Wrapper for optimisePopulation
    *
    **/
    virtual void optimiseParentPopulation();
    
    /**
    * \brief    Optimise the offspring population
    * \details  Wrapper for optimisePopulation
    *
    **/
    virtual void optimiseOffspringPopulation();
        
    /**
    * \brief    Save the elite before reduction/replacement
    * \details  Search the elitismSize best individual in population.
    *           Warning, elitism has O(elitismSize*n) complexity.
    *
    *  @param   elitismSize        The number of elite to preserve 
    *  @param   population         The population in which to search
    *  @param   populationSize     The size of the population
    *  @param   outPopulation      A allocated array for storing the newfound elite
    *  @param   outPopulationSize  The size of the outPopulation, must be equal to
    *                              elitismSize
    **/
    void strongElitism(unsigned elitismSize, CIndividual** population, unsigned populationSize, 
         CIndividual** outPopulation, unsigned outPopulationSize);

    /**
    * \brief    Save the elite before reduction/replacement
    * \details  
    *         
    *
    *  @param   elitismSize          The number of elite to preserve 
    *  @param   parentPopulation     The parent population in which to search
    *  @param   offspringPopulation  The offspring population in which to search
    *  @param   parentPopSize        The size of the parent population
    *  @param   offPopSize           The size of the offspring population
    *  @param   outPopulation        A allocated array for storing the newfound elite
    *  @param   outPopulationSize    The size of the outPopulation, must be equal to
    *                               elitismSize
    **/
    void weakElitism(unsigned elitismSize, CIndividual** parentsPopulation,
         CIndividual** offspringPopulation, unsigned* parentPopSize, unsigned* offPopSize, 
         CIndividual** outPopulation, unsigned outPopulationSize);

    /**
    * \brief    Reduce a population to a specified size, using selection
    * operator
    * \details  Available selection operator:
    *           MaxTournament,MinTournament,MinDeterministic,maxDeterministic,MaxRoulette.
    *           WARNING: NOT IMPLEMENTED 
    *
    * @param    population         The population to reduced
    * @param    populationSize     The size of the population to reduced
    * @param    reducedPopulation  An allocated  population to store the result of
    *                             the reduction of size obSize
    * @param    obSize             The wanted size after reduction
    * @param    pressure           The reduction pressure
    * @return   var 
    **/
    CIndividual** reducePopulations(CIndividual** population, unsigned populationSize,
                  CIndividual** reducedPopulation, unsigned obSize,int pressure);
    
    /**
    * \brief    Reduce the parent population  
    * \details  Wrapper for reducePopulations(..)
    *           WARNING: as reducedPopulations is not implemented, call to these
    *           function will result in compilation(linking) error
    *
    * @param    var   The wanted size after reduction
    * @return   var   The reduced population
    **/
    CIndividual** reduceParentPopulation(unsigned obSize);
    
    /**
    * \brief    Reduce the offspring population  
    * \details  Wrapper for reducePopulations(..)
    *           WARNING: as reducedPopulations is not implemented, call to these
    *           functions will result in compilation(linking) error
    *
    * @param    var   The wanted size after reduction
    * @return   var   The reduced population
    **/
    
    CIndividual** reduceOffspringPopulation(unsigned obSize);
    /**
    * \brief    Reduce the global population    
    * \details  This is the only reduction wrapper that is used 
    *           and work
    *
    * @param    elitPop   Elite population to integrate
    **/
    void reduceTotalPopulation(CIndividual** elitPop);
    
    /**
    * \brief    ?? Not defined anywhere
    * \details  Not virtual so no redefinition == dead code ?
    *
    **/
    void evolve();

    static float selectionPressure;
    static float replacementPressure;
    static float parentReductionPressure;
    static float offspringReductionPressure;
    
    /**
    * \brief    Static initialization of a CPopulation
    * \details  Only used in Multi objective (MO) template  
    *
    * @param  selectionOperator           The selection operator
    * @param  replacementOperator         The replacment operator
    * @param  parentReductionOperator     The parent reduction operator
    * @param  offspringReductionOperator  The offspring Reduction Operator
    * @param  selectionPressure           The selection pressure
    * @param  replacementPressure         The replacement pressure
    * @param  parentReductionPressure     The parent reduction pressure
    * @param  offspringReductionPressure  The offspring reduction pressure
    **/
    static void initPopulation(CSelectionOperator* selectionOperator,
                CSelectionOperator* replacementOperator,
                CSelectionOperator* parentReductionOperator,
                CSelectionOperator* offspringReductionOperator,
                float selectionPressure, float replacementPressure,
                float parentReductionPressure, float offspringReductionPressure);

    /**
    * \brief    Sort a population by fitness
    * \details  Use libc quicksort (not efficient ?). Ascending order.
    *
    *  @param   population     The population to sort
    *  @param   populationSize The size of the population
    **/
    static void sortPopulation(CIndividual** population, unsigned populationSize);

    /**
    * \brief    Reverse sort a population by fitness
    * \details  Use libc quicksort (not efficient ?). Descending order.
    *
    * @param    population     The population to sort
    * @param    populationSize The size of the population
    **/
    static void sortRPopulation(CIndividual** population, unsigned populationSize);
    
    /**
    * \brief    Serialize the parent population to file
    * \details 
    *
    **/
    void serializePopulation();
    
    /**
    * \brief    Search for the index of individual with the worst fitness
    * \details  Use the size of parent population -> Error ?
    *
    * @param    population  The population in which we want he worst individual
    * @return   var         The index of the worst individual 
    **/
    int getWorstIndividualIndex(CIndividual** population);
    
    /**
    * \brief    Sort the parent population
    * \details  Wrapper for sortPopulation()
    *
    **/
    void sortParentPopulation(){ CPopulation::sortPopulation(parents,actualParentPopulationSize);}

    /**
    * \brief    Create the offspring population 
    * \details  Use the crossover function defined by CIndividualImpl after 
    *           the easea compilation step.
    *           Virtual because some template files redefined there own (CUDA,GP).
    *
    **/
    virtual void produceOffspringPopulation();

    friend std::ostream& operator << (std::ostream& O, const CPopulation& B);

    /**
    * \brief    Set the parent population 
    * \details  Only used in Multi Objectives (MO) template
    *
    *  @param  population                 The new parent population
    *  @param  actualParentPopulationSize The new current size of the parent
    *                                     population
    **/
    void setParentPopulation(CIndividual** population, unsigned actualParentPopulationSize){
      this->parents = population;
      this->actualParentPopulationSize = actualParentPopulationSize;
    }
    
    /**
    * \brief    Reduce a population using the specified reduction operator
    * \details  
    *
    *  @param   population          The population to reduce
    *  @param   populationSize      The initial size of te population to reduce
    *  @param   reducedPopulation   The allocated array to store the reduced
    *                               population
    *  @param   obSize              The size of the reduced population 
    *  @param   replacementOperator The CSelectonOperator to use
    *  @param   pressure            The selection pressure for replacementOperator
    **/
    static void reducePopulation(CIndividual** population, unsigned populationSize,
                CIndividual** reducedPopulation, unsigned obSize,
                CSelectionOperator* replacementOperator,int pressure);
    
    /**
    * \brief    Copy from the population vector to the parent population
    * \details  Only ever used in Multi Objective (MO) template
    *
    **/
    void syncInVector();
   
    /**
    * \brief    Copy from the parent population to the vector 
    * \details  Only ever used in Multi Objective (MO) template
    *
    **/
    void syncOutVector();
  
  public:
    /*Datas-----------------------------------------------------------------------*/
    //Genetic operator probabilities
    float pCrossover;
    float pMutation;
    float pMutationPerGene;
    
    //Interesting data
    CIndividual* Best;
    CIndividual* Worst;
    float currentAverageFitness;
    float currentSTDEV;
    
    //Populations
    CIndividual** parents;
    CIndividual** offsprings;
    
    //Populations Size
    unsigned parentPopulationSize;
    unsigned offspringPopulationSize;
    
    unsigned actualParentPopulationSize;
    unsigned actualOffspringPopulationSize;
    
    //Selection and reduction operators
    static CSelectionOperator* selectionOperator;
    static CSelectionOperator* replacementOperator;
    static CSelectionOperator* parentReductionOperator;
    static CSelectionOperator* offspringReductionOperator;

    unsigned currentEvaluationNb;
    CRandomGenerator* rg;
    std::vector<CIndividual*> pop_vect;

    Parameters* params;
    CStats* cstats;


};

#endif /* CPOPULATION_H_ */
