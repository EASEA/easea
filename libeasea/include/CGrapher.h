/*
 * CGrapher.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */
/**
 * @file CGrapher.h
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

#ifndef CGRAPHER_H_
#define CGRAPHER_H_

#include <iostream>
#include <stdlib.h>
#include "Parameters.h"

class CRandomGenerator;

/**
 *  \class   CGrapher 
 *  \brief   Launch the grapher within EASEA
 *  \details Launch the java grapher in a exec'd fork.
 *           TODO:(re)Implement the constructor for Windows.
 **/

class CGrapher {
  
  public:
    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Constructor of CGrapher
    * \details  Launch a java program in a exec'd fork and open a pipe between
    *           the parent and the grapher
    *
    * @param  param  The genetic algorithme parameter (used for knowing the
    *                total number of evaluation, etc ..)
    * @param  title  Window title (the program name )  
    **/
    CGrapher(Parameters* param, char* title);
    
    /**
    * \brief    Destructor of CGrapher
    * \details  Close the pipe 
    *
    **/
    ~CGrapher();
  

  public:
    /*Datas-----------------------------------------------------------------------*/
    FILE *fWrit;
    FILE *fRead;
    int pid;
    int valid;
};

#endif /* CGRAPHER_H_ */
