#pragma once

#define PROJECT_NAME "EASENA"
#define PROJECT_VER  "RE"
#define PROJECT_VER_MAJOR 2
#define PROJECT_VER_MINOR 20
/* #undef PROJECT_VER_PATCH */

#define USE_OPENMP

#ifdef USE_OPENMP
  #define EASEA_PRAGMA_OMP_PARALLEL _Pragma("omp parallel for schedule(runtime)")
  #define EASEA_PRAGMA_OMP_ATOMIC   _Pragma("omp atomic")
#else
  #define OMP_NUM_THREADS 1
#endif
