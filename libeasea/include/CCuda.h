/*
 *    Copyright (C) 2009  Ogier Maitre

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef CCUDA_H_
#define CCUDA_H_

#include <iostream>
#include <semaphore.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
//#include <cuda_runtime_api.h>



#define CUDA_SAFE_CALL(f)				\
  {							\
  cudaError_t err;					\
    err = f;						\
    if( err != cudaSuccess ){				\
      printf("Cuda Execution Error : %s at line : %s:%d\n",cudaGetErrorString(err),__FILE__,__LINE__); \
      exit(-1);						\
    }							\
  }
   

struct gpuEvaluationData{
   int indiv_start;
   int sh_pop_size;
   
   int num_MP;
   int num_thread_max;
   int num_Warp;
   
   int dimGrid;
   int dimBlock;

  cudaDeviceProp gpuProp;

  int gpuId;
  int threadId;
  sem_t sem_in;
  sem_t sem_out;
  
  void* d_population;
  float* d_fitness;

  float* progs;
  float* d_progs;

  int* indexes;
  int* d_indexes;

  float* fitness;
  
  float* flatInputs; // flattened inputs for GP

  float* d_inputs;
  float* d_outputs;

};

#endif /* CCUDA_H_ */
