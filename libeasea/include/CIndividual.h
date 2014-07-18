/*
 * CIndividual.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

/**
 * @file CIndividual.h
 * @author SONIC BFO,Ogier Maitre
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

#ifndef CINDIVIDUAL_H_
#define CINDIVIDUAL_H_

#include <iostream>

class CRandomGenerator;

/**
* \class    CIndividual 
* \brief    Base abstract class for individual description
* \details  This class is implemented using the genome description found in ez
*           file
*  
**/
class CIndividual {
  public:
    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Constructor of CIndividual
    * \details  Implemented by the easea compiler using ez files 
    *           and template
    *
    **/
    CIndividual();
    
    /**
    * \brief    Destructor of CIndividual
    * \details  Implemented by the easea compiler using ez files 
    *           and template
    *
    **/
    virtual ~CIndividual();
    
    
    /*Methods---------------------------------------------------------------------*/
    /*GENETIC OPERATOR*/
    /**
    * \brief    Evaluate the individual
    * \details  Implemented by the easea compiler using ez files 
    *           and template
    *
    **/
    virtual float evaluate() = 0;
    
    /**
    * \brief    "Print" the individual
    * \details  Implemented by the easea compiler using ez files
    *           and template
    *
    * @param    O  Output stream on which to print 
    **/
    virtual void printOn(std::ostream& O) const = 0;
    
    /**
    * \brief    Mutate the individual
    * \details  Implemented by the easea compiler using ez files
    *           and template
    *
    * @param    pMutationPerGene  Probabilities to mutate per gene, [0,1]
    * @return   var               The number of mutated gene
    **/
    virtual unsigned mutate(float pMutationPerGene) = 0;
    
    /**
    * \brief    Crossover the individual with an array of parents
    * \details  Implemented by the easea compiler using ez files
    *           and template
    *
    * @param    p2    The array of parents
    * @return   var   A pointer on the newly formed individual
    **/
    virtual CIndividual* crossover(CIndividual** p2) = 0;
    
    /**
    * \brief    Clone the individuals
    * \details  Implemented by the easea compiler using ez files
    *           and template
    *
    * @return   var   A pointer on the resulting clone
    **/
    virtual CIndividual* clone() = 0;
    
    /*SERIALIZATION METHODS*/
    /**
    * \brief    Serialize the individual
    * \details  Implemented by the easea compiler, with methods from EasySym.cpp
    *
    * @return   var   The binary serialization into a string 
    **/
    virtual std::string serialize() = 0;

    /**
    * \brief    Deserialize the individual
    * \details  Implemented by the easea compiler, with methods from EasySym.cpp
    *
    * @param    EASEA_Line  The binary serialization
    **/
    virtual void deserialize(std::string EASEA_Line) = 0;

    /**
    * \brief    Check bound using the user's defined ones
    * \details  Mainly used in genetic programming, but available in every other
    *           mode too. In ez  : section Bounds.
    *           Implemented by the easea compiler using ez files
    *           and template.
    *
    **/
    virtual void boundChecking() = 0;
    

    /*Getter----------------------------------------------------------------------*/
    /**
    * \brief    Return the crossover arrity
    * \details  EASEA currently only support and use an arrity of 2
    *
    * @return   var   The arrity (2 for the moment)
    **/
    static unsigned getCrossoverArrity(){ return 2; }

    /**
    * \brief    Return the individuals fitness
    * \details  
    *
    * @return   var   The fitness
    **/
    float getFitness(){ return this->fitness; }
  
  public:
    /*Datas-----------------------------------------------------------------------*/
    bool valid;
    bool isImmigrant;
    float fitness;
    static CRandomGenerator* rg;


};

#endif /* CINDIVIDUAL_H_ */
