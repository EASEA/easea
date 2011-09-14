/*
 * CGrapher.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef CGRAPHER_H_
#define CGRAPHER_H_

class CRandomGenerator;
#include <iostream>
#include <stdlib.h>
#include "Parameters.h"

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
