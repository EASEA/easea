/***********************************************************************
| CCuError.h                                                            |
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
#pragma once

#define CUDA_CHECK(call)                                                                             \
{                                                                                                    \
    const cudaError_t error = call;                                                                  \
    if (error != cudaSuccess)                                                                        \
    {                                                                                                \
        fprintf(stderr, "EASEA LOG [CUDA ERROR] Error: %s:%d, ", __FILE__, __LINE__);                \
        fprintf(stderr, "code: %d, reason: %s\n", error,                                             \
                cudaGetErrorString(error));                                                          \
    }                                                                                                \
}

#define CURAND_CHECK(call)                                                                           \
{                                                                                                    \
    curandStatus_t err;                                                                              \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                                                     \
    {                                                                                                \
        fprintf(stderr, "EASEA LOG [CUDA ERRIR] Got CURAND error %d at %s:%d\n", err, __FILE__,      \
                __LINE__);                                                                           \
        exit(1);                                                                                     \
    }                                                                                                \
}
