/*
 * CStoppingCriterion.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

/**
 * @file CStoppingCriterion.h
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

#ifndef CSTOPPINGCRITERION_H_
#define CSTOPPINGCRITERION_H_

#include <stdlib.h>
#include <signal.h>

#ifndef WIN32
  #include <sys/time.h>
#endif

#ifdef WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

#include <time.h>

class CEvolutionaryAlgorithm;

/* ****************************************
   StoppingCriterion class
 ****************************************/
/**
*  \class   CStoppingCriterion 
*  \brief   Defined halt depending on rules
*  \details Three implementation are available and used,
*           based on the number of generation, the time elapsed and system signal 
*  
**/
class CStoppingCriterion {

  public:
    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Must return true when the criterions are met 
    * \details  Pure virtual
    *
    * @return   var   A boolean 
    **/
    virtual bool reached() = 0;

    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Destructor of CStoppingCriterion
    * \details  Empty.
    **/
    virtual ~CStoppingCriterion(){;}
};


/* ****************************************
   GenerationalCriterion class
 ****************************************/
/**
*  \class   CGenerationalCriterion 
*  \brief   Stop the algorithm after a specified amout of generation
*  \details  
*  
**/
class CGenerationalCriterion : public CStoppingCriterion {
  
  public:
    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Constructor of CGenerationalCriterion
    * \details  
    *
    * @param    generationalLimit   The wanted number of generation before
    *                               stopping
    **/
    CGenerationalCriterion(unsigned generationalLimit);
    
    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Return true when the generational limit is met
    * \details  When *currentGenerationPtr>=generationalLimit
    *
    * @return   var   A boolean
    **/
    virtual bool reached();
    
    /*Setters---------------------------------------------------------------------*/
    /**
    * \brief    Set the generation counter
    * \details  The pointer on CEvolutionaryAlgorithm's generation counter
    *
    * @param    var   A pointer to a counter 
    **/
    void setCounterEa(unsigned* ea_counter);
    
    /*Getter----------------------------------------------------------------------*/
    /**
    * \brief    Get the current generation limit 
    * \details  
    *
    * @return   var   The current generation limit
    **/
    unsigned *getGenerationalLimit();
  
  private:
    /*Datas-----------------------------------------------------------------------*/
    unsigned* currentGenerationPtr;
    unsigned generationalLimit;
  
};

/* ****************************************
   TimeCriterion class
 ****************************************/
/**
*  \class   CTimeCriterion 
*  \brief   Stop the algorithm after a specified amout of time 
*  \details  
*  
**/
class CTimeCriterion : public CStoppingCriterion {
  
  public:
    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Constructor of CTimeCriterion
    * \details 
    *
    * @param    timeLimit   The time limit in second
    **/
    CTimeCriterion(unsigned timeLimit);
    
    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Return true when the time limit is met
    * \details  when elapsedTime >= timeLimit
    *
    * @return   var   A boolean
    **/
    virtual bool reached();
  
    /*Setters---------------------------------------------------------------------*/
    /**
    * \brief    Set the elapsed time
    * \details  Done in the main EA loop
    *
    * @param    elapsedTime   Time in second 
    **/
    void setElapsedTime(double elapsedTime);
    
    /*Getter----------------------------------------------------------------------*/
    /**
    * \brief    Get the elapsed time
    * \details  In second
    *
    * @return   var   The elapsed time in second 
    **/
    double getElapsedTime();
  
  private:
    /*Datas-----------------------------------------------------------------------*/
    unsigned timeLimit;
    double elapsedTime;
  
};

/* ****************************************
   ControlCStopingCriterion class
 ****************************************/
extern void signal_handler(int sig);

/**
*  \class   CControlCStopingCriterion 
*  \brief   Stop the algorithm after receiving a SIGINT signal
*  \details  
*  
**/
class CControlCStopingCriterion : public CStoppingCriterion {
  
  private:
  public:
    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Constructor of CControlCStopingCriterion
    *  \details After receiving a SIGINT ask for confirmation, or stop at next
    *           generation
    *
    **/
    CControlCStopingCriterion();
    
    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Return true when a SIGINT signal was cought
    * \details  
    *
    * @return   var   A boolean
    **/
    virtual bool reached();
};
#endif /* CSTOPPINGCRITERION_H_ */
