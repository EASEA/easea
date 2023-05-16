/*
 * Parameters.cpp
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#include "Parameters.h"
#include "COptionParser.h"
#include <stdio.h>

#define SV(s, df) (setVariable(s, df, vm, vm_file))

Parameters::Parameters(std::string const& filename, int argc, char* argv[]) : optimise(false), baldwinism(false) {
	parseArguments(filename.c_str(), argc, argv, vm, vm_file);
	
	// set parameters that can only be defined using cmd line arguments or files
	nbCPUThreads = SV("nbCPUThreads", 1);
	noLogFile = SV("noLogFile", false);
	reevaluateImmigrants = SV("reevaluateImmigrants", false);
	alwaysEvaluate = SV("alwaysEvaluate", false);
	seed = SV("seed", 0);
	printInitialPopulation = SV("printInitialPopulation", false);
	printFinalPopulation = SV("printFinalPopulation", false);
	u1 = SV("u1", "");
	u2 = SV("u2", "");
	u3 = SV("u3", "");
	u4 = SV("u4", "");
	u5 = SV("u5", "");
}

Parameters::~Parameters(){
}

int Parameters::setReductionSizes(int popSize, float popReducSize){
        if(popReducSize<1.0 && popReducSize>=0.0)
                return static_cast<int>(popReducSize*static_cast<float>(popSize));
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
                return static_cast<int>(popReducSize);
}

