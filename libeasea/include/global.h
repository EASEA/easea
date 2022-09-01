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

extern std::ofstream easena::log_file;

//#define true 1;
//#define false 0;
class CIndividual;

extern bool bReevaluate;
//extern CIndividual** pPopulation;
//extern CIndividual* bBest;
//extern float* pEZ_MUT_PROB;
//extern float* pEZ_XOVER_PROB;
extern unsigned *EZ_NB_GEN;
extern int EZ_POP_SIZE;
extern int OFFSPRING_SIZE;
extern unsigned *EZ_current_generation;
extern std::vector<char *> vArgv;
extern CRandomGenerator* globalRandomGenerator;
extern easena::log_stream logg;

#ifdef WIN32
#define RNDMAX (RAND_MAX+1)
#else
#define RNDMAX (RAND_MAX)
#endif
#endif /* GLOBAL_H_ */
