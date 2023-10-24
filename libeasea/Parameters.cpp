/*
 * Parameters.cpp
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#include "Parameters.h"
#include "COptionParser.h"
#include "CRandomGenerator.h"
#include <stdio.h>
#include <cmath>
#include <random>

extern CRandomGenerator* globalRandomGenerator;


Parameters::Parameters(std::string const& filename, int argc, char* argv[]) : optimise(false), baldwinism(false) {
	parseArguments(filename.c_str(), argc, argv, vm);

	// set parameters that can only be defined using cmd line arguments or files
	nbCPUThreads = setVariable("nbCPUThreads", 1);
	noLogFile = setVariable("noLogFile", false);
	reevaluateImmigrants = setVariable("reevaluateImmigrants", false);
	silentNetwork = setVariable("silentNetwork", false);
	alwaysEvaluate = setVariable("alwaysEvaluate", false);

	seed = setVariable("seed", std::random_device{}()); // use hardware based entropy as seed
	globGen = decltype(globGen){static_cast<unsigned int>(seed)};
	globalRandomGenerator = &globGen;
	randomGenerator = globalRandomGenerator;

	printInitialPopulation = setVariable("printInitialPopulation", false);
	printFinalPopulation = setVariable("printFinalPopulation", false);
	u1 = setVariable("u1", "");
	u2 = setVariable("u2", "");
	u3 = setVariable("u3", "");
	u4 = setVariable("u4", "");
	u5 = setVariable("u5", "");
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

int Parameters::getOffspringSize(int defaut_value, int parent_size) const {
	float cur = setVariable("nbOffspring", static_cast<float>(defaut_value));
	if (cur <= 1.0) { // relative
		auto parentSize = setVariable("popSize", parent_size);
		return static_cast<int>(cur * static_cast<float>(parentSize));
	} else {
		return static_cast<int>(std::round(cur));
	}
}

