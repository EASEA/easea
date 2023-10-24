/*
 * global.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <stdlib.h>
#include <vector>
#include "define.h"
#include "CLogFile.h"
#include "CRandomGenerator.h"

class CRandomGenerator;

extern std::ofstream easea::log_file;

// used to track static value and ensure retrocompatibility :(
struct globalRandomGeneratorWrapper {
	[[deprecated("globalRandomGenerator use is deprecated, use free functions 'random' instead")]]
	CRandomGenerator* operator->() const { return &globGen; }
	globalRandomGeneratorWrapper const& operator=(CRandomGenerator* new_gen) const { globGen = *new_gen; return *this; }
	globalRandomGeneratorWrapper const& operator=(CRandomGenerator const& new_gen) const { globGen = new_gen; return *this; }
};

//#define true 1;
//#define false 0;
class CIndividual;

static const globalRandomGeneratorWrapper globalRandomGenerator{};
extern bool bReevaluate;
extern CIndividual** pPopulation;
extern CIndividual* bBest;
extern float* pEZ_MUT_PROB;
extern float* pEZ_XOVER_PROB;
extern unsigned *EZ_NB_GEN;
extern int EZ_POP_SIZE;
extern int OFFSPRING_SIZE;
extern unsigned *EZ_current_generation;
extern std::vector<char *> vArgv;
extern easea::log_stream logg;

#ifdef WIN32
#define RNDMAX (RAND_MAX+1)
#else
#define RNDMAX (RAND_MAX)
#endif
#endif /* GLOBAL_H_ */
