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

class CGnuplot {
public:
	FILE *fWrit;
	FILE *fRead;
	int pid;
	int valid;
public:
	CGnuplot();
	 ~CGnuplot();
};

#endif /* CGNUPLOT_H_ */
