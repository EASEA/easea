#pragma once

#define PROJECT_NAME "EASENA"
#define PROJECT_VERSION  "3.0.0"
#define PROJECT_VERSION_MAJOR 3
#define PROJECT_VERSION_MINOR 0
#define PROJECT_VERSION_PATCH 0

#define USE_OPENMP

#ifdef USE_OPENMP
  #define EASEA_PRAGMA_OMP_PARALLEL _Pragma("omp parallel for schedule(runtime)")
  #define EASEA_PRAGMA_OMP_ATOMIC   _Pragma("omp atomic")
#else
  #define OMP_NUM_THREADS 1
#endif

// OS defines
#if defined(linux) || defined(__linux) || (__APPLE__) || (macintosh) || (Macintosh)
	#define OS_UNIX
#elif defined(__CYGWIN__) || defined(__MINGW32__) || defined(__MINGW64__) || defined(_WIN32) || defined(_WIN64) || defined(__WIN32__) || defined(__WINDOWS__) || defined(__TOS_WIN__)
	#define OS_WINDOWS
#endif
