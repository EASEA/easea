/*
 * global.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <stdlib.h>
#include "define.h"

//#define true 1;
//#define false 0;
class CIndividual;

extern CIndividual** pPopulation;
extern CIndividual* bBest;
extern float* pEZ_MUT_PROB;
extern float* pEZ_XOVER_PROB;
extern unsigned *EZ_NB_GEN;
extern unsigned *EZ_current_generation;

extern CRandomGenerator* globalRandomGenerator;

#ifdef WIN32
#define RNDMAX (RAND_MAX+1)
#else
#define RNDMAX (RAND_MAX)
#endif
#endif /* GLOBAL_H_ */
