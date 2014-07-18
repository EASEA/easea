/*
 * CEvolutionaryAlgorithm.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CEVOLUTIONARYALGORITHM_H_
#define CEVOLUTIONARYALGORITHM_H_
#include <stdlib.h>
#include <string>
#include <time.h>
#include "CEvolutionaryAlgorithm.h"
#include "CSelectionOperator.h"
#include "CPopulation.h"
#include "CStoppingCriterion.h"
#include "CComUDPLayer.h"
#include "CStats.h"
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

class Parameters;
class CGrapher;
/**
* \class    CEvolutionaryAlgorithm 
* \brief    The complete program wrapper
* \details  From which the main loop is started
*  
**/
class CEvolutionaryAlgorithm {

  public:
    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Constructor of CEvolutionaryAlgorithm
    * \details  The given Parameters object contains all EA option and
    *           parameters
    *
    * @param    params All the parameters
    **/
    CEvolutionaryAlgorithm( Parameters* params );
    
    /**
    * \brief    Destructor of CEvolutionaryAlgorithm
    *  \details  
    *
    **/
    virtual ~CEvolutionaryAlgorithm();
    
    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Initialize the first parent population
    * \details  Implemented by the EASEA compiler using templates files and in
    *           ez files the initialiser section 
    *
    **/
    virtual void initializeParentPopulation() = 0;
    
    /**
    * \brief    Add a stopping criterion
    * \details  Will be added to the vector of stopping criterion. Add the end
    *           of each generation the allCriteria() method will be called to check 
    *           in the vector if a stopping criterion is met
    *
    * @param    sc  The stopping criterion to add
    **/
    void addStoppingCriterion(CStoppingCriterion* sc);
    
    /**
    * \brief    The main EA loop
    * \details  The whole GA is done here. At the end of each generation, if a
    *           stopping criterion is met the program will exit
    *
    **/
    void runEvolutionaryLoop();
    
    /**
    * \brief    Check if a stopping criterion is met
    * \details  Check in the vector if a condition is met
    *
    * @return   var   True is a condition is met 
    **/
    bool allCriteria();
    
    /**
    * \brief    Print the general stats of the population 
    * \details  Show average fitness, standard deviation of fitness,the best's
    *           fitness, worst's fitness.
    *           TODO:This function need EXTREME REFACTORING.
    *
    *  @param   beginTime  The absolute time of the previous generation's end 
    **/
    #ifdef WIN32
      void showPopulationStats(clock_t beginTime);
    #else
      void showPopulationStats(struct timeval beginTime);
    #endif
    
    /**
    * \brief        Empy
    * \deprecated   Empty
    * \details      To be removed in next version
    *
    **/
    void outputGraph();
    
    /**
    * \brief    Generate the gnuplot script for plotting the .dat file generated
    *           at execution
    * \details  
    *
    **/
    
    void generatePlotScript();
    /**
    * \brief    Generate the R script for plotting the .dcsv file generated
    *           at execution
    * \details  
    *
    **/
    void generateRScript();

    /*REMOTE ISLAND METHOD*/
    /**
    * \brief        Wrapper for refreshClient
    * \deprecated   Pretty useless, in future version call directly refresh
    *               client
    *
    **/
    void initializeClients();
    
    /**
    * \brief    Add the receive individuals to the parent population
    * \details  Request a lock on the data of the UPD server. For each received
    *           individual, do an "anti" tournament to select a bad individual
    *           from the parent population to replace.
    *
    **/
    void receiveIndividuals();
    
    /**
    * \brief    Randomly send a individual by using the migration probability
    *           parameter.
    * \details  Also randomly select a client in the client array.
    *
    **/
    void sendIndividual();
    
    /**
    * \brief    Read the ip file and load the ip in the client array
    * \details  Only called once at the beginning of the main loop
    *
    **/
    void refreshClient();
    
    /*Getter----------------------------------------------------------------------*/
    /**
    * \brief    Return a pointer on the current generation counter variable
    * \details  Used by CGenerationalCriterion
    *
    * @return   var   The pointer on the current generation counter variable
    **/
    unsigned int *getCurrentGenerationPtr(){ return &currentGeneration;}
    
    /**
    * \brief    Return the current population 
    * \details  This mean the "parent" population
    *
    * @return   var   The current population pointer 
    **/
    CPopulation* getPopulation(){ return population;}
    
    /**
    * \brief    Return the number of the current generation 
    * \details  
    *
    * @return   var   The number of the current population
    **/
    unsigned getCurrentGeneration() { return currentGeneration;}
  
  public:
    /*Datas-----------------------------------------------------------------------*/
    /*EA*/
    unsigned currentGeneration;
    CPopulation* population;
    unsigned reduceParents;
    unsigned reduceOffsprings;
    
    /*REMOTE ISLAND DATA*/
    unsigned treatedIndividuals;
    unsigned numberOfClients;
    unsigned myClientNumber;
    CComUDPServer *server;
    CComUDPClient **Clients;
    
    /*META*/
    Parameters* params;
    CGrapher* grapher;
    CStats* cstats;
    std::vector<CStoppingCriterion*> stoppingCriteria;
    std::string* outputfile;
    std::string* inputfile;

};


#endif /* CEVOLUTIONARYALGORITHM_H_ */
