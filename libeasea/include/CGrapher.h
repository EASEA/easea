/*
 * CGrapher.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef CGRAPHER_H_
#define CGRAPHER_H_

#include <iostream>
#include <stdlib.h>
#include "Parameters.h"

/**
 *  \class   CGrapher 
 *  \brief   Launch the grapher within EASEA
 *  \details Launch the java grapher in a exec'd fork.
 *           TODO:(re)Implement the constructor for Windows.
 **/

class CGrapher {
  public:
    FILE *fWrit;
    FILE *fRead;
    int pid;
    int valid;
  public:
    CGrapher(Parameters* param, char* title);
    ~CGrapher();
};

#endif /* CGRAPHER_H_ */
