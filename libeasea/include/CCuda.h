/*
 * CCuda.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef CCUDA_H_
#define CCUDA_H_

#include <iostream>
#include <semaphore.h>
//#include <cuda_runtime_api.h>



#define CUDA_SAFE_CALL(f)				\
  {							\
  cudaError_t err;					\
    err = f;						\
    if( err != cudaSuccess ){				\
      printf("Error : %s\n",cudaGetErrorString(err));	\
      exit(-1);						\
    }							\
  }
   

struct gpuOptions{};

struct my_struct_gpu{
   int indiv_start;
   int sh_pop_size;
   
   int num_MP;
   int num_thread_max;
   int num_Warp;
   
   int dimGrid;
   int dimBlock;
};

struct gpuArg{
  int threadId;
  sem_t sem_in;
  sem_t sem_out;
  
  void* d_population;
  float* d_fitness;

};

class CCuda {
public:
	void* cudaBuffer;
	unsigned sizeOfIndividualImpl;
	struct gpuOptions initOpts;
public:
	CCuda(unsigned parentSize, unsigned offSize, unsigned individualImplSize);
	~CCuda();
};

bool repartition(struct my_struct_gpu* gpu_infos);

#endif /* CCUDA_H_ */
