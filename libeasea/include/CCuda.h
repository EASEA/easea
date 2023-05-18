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
// NOLINTNEXTLINE
#include <cuda.h>
// NOLINTNEXTLINE
#include <cuda_runtime_api.h>
//#include <cuda_runtime_api.h>

#define CUDA_SAFE_CALL(f)				\
  {							\
  cudaError_t err;					\
    err = f;						\
    if( err != cudaSuccess ){				\
      printf("Cuda Execution Error : %s at line : %s:%d\n",cudaGetErrorString(err),__FILE__,__LINE__); \
      exit(1);						\
    }							\
  }
   
template <typename T>
struct gpuEvaluationData{
   int indiv_start;
   int sh_pop_size;
   
   int num_MP;
   int num_thread_max;
   int num_Warp;
   
   int dimGrid;
   int dimBlock;

  cudaDeviceProp gpuProp; // NOLINT

  int gpuId;
  int threadId;
  sem_t sem_in;
  sem_t sem_out;
  
  void* d_population;
  T* d_fitness;

  float* progs;
  float* d_progs;

  int* indexes;
  int* d_indexes;

  T* fitness;
  
  float* flatInputs; // flattened inputs for GP

  float* d_inputs;
  float* d_outputs;

};

#endif /* CCUDA_H_ */
