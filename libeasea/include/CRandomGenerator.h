/*
 * CRandomGenerator.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

/**
 * @file CRandomGenerator.h
 * @author Ogier Maitre,Pallamidessi Joseph
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


#ifndef CRANDOMGENERATOR_H_
#define CRANDOMGENERATOR_H_

#include <iostream>
#include "MersenneTwister.h"

/**
*  \class   CRandomGenerator 
*  \brief   An utility class for providing random number.
*  \details Use Mersenne twister as internal PRNG. None of the values (bounds,
*           bias) are checked, for performance purposes.
*  
**/
class CRandomGenerator {
  private:
    unsigned seed;
    MTRand* mt_rnd;
  public:
    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Constructor of CRandomGenerator
    * \details  Use a non-zero seed, because of how work MT 
    *
    * @param    seed The generator see, as a uint_32.
    **/
    CRandomGenerator(unsigned int seed);
    

    /**
    * \brief    Destructor
    *  \details  
    **/
    ~CRandomGenerator();


    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Return an random signed integer
    *
    * @return   rand A random integer
    **/
    int randInt();


    /**
    * \brief    A simple unbiased tossCoin
    * \details  P(true)=0.5   
    *
    * @return   var A random unbiased boolean
    **/
    bool tossCoin();


    /**
    * \brief    A biased tosscoin
    * \details  Take a bias in argument as a float, between [0,1].
    *           value not checked
    *
    * @param    bias A float representing the bias
    * @return   var  A random biased boolean
    **/
    bool tossCoin(float bias);
    

    /**
    * \brief    Return an random integer in a specified range
    * \details  Exclude the upper bound. Values not checked.
    *
    *  @param   min  The lower bound, included 
    *  @param   max  the upper bound exluded
    * @return   var  A random integer between th specified range
    **/
    int randInt(int min, int max);


    /**
    * \brief    Return a random integer between 0 and max ([0,max[)
    * \details  0 included, max excluded, equivalent to randInt(0,max).
    *           values not checked.
    *
    * @param    max   The upper bound
    * @return   var   A random integer 
    **/
    int getRandomIntMax(int max);
    
    
    /**
    * \brief    Return a random float between a calculated range
    * \details  As [min,max-min]. values not checked.
    *
    *  @param   min  The lower bound 
    *  @param   max  The "upper" bound 
    * @return   var  A random float
    **/
    float randFloat(float min, float max);
    

    /**
    * \brief    Return an random integer in a specified range
    * \details  Exclude the upper bound. Values not checked. 
    *           To be used in .ez code
    *
    *  @param   min  The lower bound, included 
    *  @param   max  The upper bound, exluded
    * @return   var  A random integer between th specified range
    **/
    int random(int min, int max);
    

    /**
    * \brief    Return an random float in a specified range
    * \details  Exclude the upper bound. Values not checked. 
    *           To be used in .ez code
    *
    *  @param   min  The lower bound, included 
    *  @param   max  The upper bound, exluded
    * @return   var  A random float between th specified range
    **/
    float random(float min, float max);


    /**
    * \brief    Return an random float as a double,in a specified range
    * \details  Exclude the upper bound. Values not checked. 
    *           To be used in .ez code
    *
    *  @param   min  The lower bound, included 
    *  @param   max  The upper bound, exluded
    * @return   var  A random float implicitly cat as a double between th specified range
    **/
    double random(double min, double max);


    /**
    * \brief    Box-Muller method for gaussian random distribution
    * \details  Use no internal states. 
    *
    *  @param   mean     Mean of the gaussian distribution
    *  @param   std_dev  Standard deviation of the gaussian distribution
    *  @param   z_0      First result (?)
    *  @param   z_1      Second result (?)
    **/
    void random_gauss(float min, float max, float* z_0, float* z_1);
    
    
    /**
    * \brief    Return a random float following a gaussian distribution 
    * \details  Use no internal states. Do randomly choose one of the 2 result
    *           of random_gauss(..). 
    *
    *  @param   mean    Mean of the gaussian distribution
    *  @param   std_dev Standard deviation of the gaussian distribution
    *  @return  var     A random float 
    **/
    float random_gauss(float mean, float std_dev);
    

    /*Getters--------------------------------------------------------------------*/
    /**
    * \brief    Return the current seed of the internal PRNG 
    * \details  The current PNRG implemented is a Mersenne Twister
    *
    * @return   var A uint_32, the seed
    **/
    unsigned get_seed()const {return this->seed;}


    /*Operators------------------------------------------------------------------*/
    /**
    * \brief    << Operator 
    * \details  Return the string containing the seed, prefixed by "s : "
    *
    * @return   var return stream, for chaining
    **/
    friend std::ostream & operator << (std::ostream & os, const CRandomGenerator& rg);

};

#endif /* CRANDOMGENERATOR_H_ */
