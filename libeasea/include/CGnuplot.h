/*
 * CGnuplot.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef CGNUPLOT_H_
#define CGNUPLOT_H_

class CRandomGenerator;
#include <iostream>
#include <stdlib.h>
#include "Parameters.h"

class CGnuplot {
public:
	FILE *fWrit;
	FILE *fRead;
	int pid;
	int valid;
public:
	CGnuplot(Parameters* param, char* title);
	 ~CGnuplot();
};

#endif /* CGNUPLOT_H_ */
