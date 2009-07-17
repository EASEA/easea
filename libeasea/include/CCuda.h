/*
 * CCuda.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef CCUDA_H_
#define CCUDA_H_

#include <iostream>

struct gpuOptions{};

class CCuda {
public:
	void* cudaParentBuffer;
	void* cudaOffspringBuffer;
	size_t sizeOfIndividualImpl;
	struct gpuOptions initOpts;
public:
	CCuda(size_t parentSize, size_t offSize, size_t individualImplSize);
	~CCuda();
};

size_t partieEntiereSup(float E);
int puissanceDeuxSup(float n);
bool repartition(size_t popSize, size_t* nbBlock, size_t* nbThreadPB, size_t* nbThreadLB, size_t nbMP, size_t maxBlockSize);

#endif /* CCUDA_H_ */
