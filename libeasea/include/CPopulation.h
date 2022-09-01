/*
 * CPopulation.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 *
 **/

#ifndef CPOPULATION_H_
#define CPOPULATION_H_


#ifdef DEBUG
 #define DEBUG_PRT(format, ...) fprintf (stdout,"***DBG***  %s-%d: "format"\n",__FILE__,__LINE__, __VA_ARGS__)
 #define DEBUG_YACC(format, ...) fprintf (stdout,"***DBG_YACC***  %s-%d: "format"\n",__FILE__,__LINE__, __VA_ARGS__)
#else
 #ifndef WIN32
  #define DEBUG_PRT(format, ...)
  #define DEBUG_YACC(format, ...)
 #endif
#endif

#include <vector>
#include <functional>
#include <memory>

#include <CSelectionOperator.h>
#include <CComparator.h>
#include <CLogger.h>

class Parameters;
class CStats;

using CIndividual_ptr = std::unique_ptr<CIndividual>;

class CPopulation {



public:

    std::vector<CIndividual_ptr> parents;
    std::vector<CIndividual_ptr> offsprings;
    std::vector<CIndividual_ptr> pop_vect;

    float pCrossover;
    float pMutation;
    float pMutationPerGene;

    std::reference_wrapper<CIndividual_ptr> Best;
    std::reference_wrapper<CIndividual_ptr> Worst;
    float currentAverageFitness;
    float currentSTDEV;

    unsigned parentPopulationSize;
    unsigned offspringPopulationSize;

    unsigned actualParentPopulationSize;
    unsigned actualOffspringPopulationSize;

    static CSelectionOperator* selectionOperator;
    static CSelectionOperator* replacementOperator;
    static CSelectionOperator* parentReductionOperator;
    static CSelectionOperator* offspringReductionOperator;

    unsigned currentEvaluationNb;
    unsigned realEvaluationNb;
    CRandomGenerator* rg;
    Parameters* params;
    CStats* cstats;

  public:
    //CPopulation();

    CPopulation(unsigned parentPopulationSize, unsigned offspringPopulationSize,
                float pCrossover, float pMutation, float pMutationPerGene,
                CRandomGenerator* rg, Parameters* params, CStats* cstats);
    virtual ~CPopulation();

    //virtual void initializeParentPopulation() = 0;
    void addIndividualParentPopulation(CIndividual_ptr&& indiv, unsigned id);
    void addIndividualParentPopulation(CIndividual_ptr&& indiv);
    void evaluatePopulation(std::vector<CIndividual_ptr> const& population);
    virtual void optimisePopulation(std::vector<CIndividual_ptr> const& population);
    virtual void evaluateParentPopulation();
    virtual void optimiseParentPopulation();

    void strongElitism(unsigned elitismSize, std::vector<CIndividual_ptr> const& population, std::vector<CIndividual_ptr>& outPopulation);

    void weakElitism(unsigned elitismSize, std::vector<CIndividual_ptr> const& parentsPopulation, unsigned& actualParentPopulationSize, std::vector<CIndividual_ptr> const& offspringPopulation, unsigned& actualOffspringPopulationSize, std::vector<CIndividual_ptr>& outPopulation);

    virtual void evaluateOffspringPopulation();
    virtual void optimiseOffspringPopulation();

    void reducePopulations(std::vector<CIndividual_ptr> const& population,
                  std::vector<CIndividual_ptr>& reducedPopulation, unsigned obSize,float pressure);
    void reduceParentPopulation(unsigned obSize);
    void reduceOffspringPopulation(unsigned obSize);
    void reduceTotalPopulation(std::vector<CIndividual_ptr>& elitPop);
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

    static void sortPopulation(std::vector<CIndividual_ptr>& pop);

    static void sortRPopulation(std::vector<CIndividual_ptr>& pop);

    void serializePopulation();
    int getWorstIndividualIndex(std::vector<CIndividual_ptr> const& population);

    virtual void produceOffspringPopulation();

    friend std::ostream& operator << (std::ostream& O, const CPopulation& B);


    void setParentPopulation(std::vector<CIndividual_ptr> const& population, unsigned actualParentPopulationSize){
      this->parents = population;
    }

    static void reducePopulation(std::vector<CIndividual_ptr> const& population, unsigned populationSize,
                std::vector<CIndividual_ptr>& reducedPopulation, unsigned obSize,
                CSelectionOperator* replacementOperator,float pressure);

    void syncOutVector();
    void syncInVector();



};
#endif
