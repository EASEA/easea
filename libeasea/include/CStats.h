/**
 * @file CStats.h
 * @author SONIC BFO
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

#ifndef CSTATS_H_
#define CSTATS_H_

/**
*  \struct   CStats 
*  \brief    A simple class to keep track of immigrant and population's
*            statistic (averaeg, std dev) 
**/
class CStats{

  public:
    /*Constructors/Destructor-----------------------------------------------------*/ 
    /**
    * \brief    Constructor of CStats
    * \details  Initialized every members to zero
    *
    **/
    CStats();
    
    /**
    * \brief    Destructor of CStats
    *
    **/
    ~CStats();
    
    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Reset every stat to zero
    * \details  TODO: inspect behaviours of totalNumberOfImmigrants
    *
    **/
    void resetCurrentStats();
  
  public:
    /*Datas-----------------------------------------------------------------------*/
    int totalNumberOfImmigrants;
    int currentNumberOfImmigrants;

    int totalNumberOfImmigrantReproductions;
    int currentNumberOfImmigrantReproductions;

    double currentAverageFitness;
    double currentStdDev;

};

#endif /* CSTATS_H_ */
