/***********************************************************************
| CcuRandom.h                                                           |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2020-03                                                         |
|                                                                       |
 ***********************************************************************/
#include "CcuError.h"
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
// NOLINTNEXTLINE
#include <cuda.h>
// NOLINTNEXTLINE
#include <curand_kernel.h>

template <typename T>
__device__ T cuRandom (T min_v, T max_v )
{
    int ind = threadIdx.x+blockDim.x*blockIdx.x;
    curandState state;
    float RANDOM;
    curand_init((unsigned long long)clock() + ind, 0, 0, &state);
    RANDOM = curand_uniform(&state);
    
    return min_v+RANDOM*(max_v-min_v);
};
