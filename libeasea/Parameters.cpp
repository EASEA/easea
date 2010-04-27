/*
 * Parameters.cpp
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#include "include/Parameters.h"
#include <stdio.h>

#ifdef WIN32
Parameters::Parameters(){
}

Parameters::~Parameters(){
}
#endif

int Parameters::setReductionSizes(int popSize, float popReducSize){
        if(popReducSize<1.0 && popReducSize>=0.0)
                return (int)(popReducSize*popSize);
        if(popReducSize<0.0)
                return 0;
	if(popReducSize == 1.0)
		return popSize;
        if((int)popReducSize>popSize){
                printf("*WARNING* ReductionSize greater than PopulationSize !!!\n");
                printf("*WARNING* ReductionSize will be PopulationSize\n");
                return popSize;
        }
        else
                return (int)popReducSize;
}

