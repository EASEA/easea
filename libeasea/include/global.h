/*
 * global.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */
/**
 * @file global.h
 * @author SONIC BFO, Pallamidessi Joseph
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

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <stdlib.h>
#include "define.h"
/**
 *  This header contains global variable that must be shared across the whole
 *  program. Please add similar variable here for clarity.
 **/
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
