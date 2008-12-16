#include <time.h>             //gettimeofday
#include <sys/time.h>
#include <stdio.h>

#ifndef TIMING_H
#define TIMING_H


#ifdef TIMING
#define DECLARE_TIME(t)				\
  struct timeval t##_beg, t##_end, t##_res
#define TIME_ST(t)				\
  gettimeofday(&t##_beg,NULL)
#define TIME_END(t)				\
  gettimeofday(&t##_end,NULL)
#define SHOW_TIME(t)						\
  timersub(&t##_end,&t##_beg,&t##_res);				\
  printf("%s : %d.%06d\n",#t,t##_res.tv_sec,t##_res.tv_usec)
#define SHOW_SIMPLE_TIME(t)					\
  printf("%s : %d.%06d\n",#t,t.tv_sec,t.tv_usec)
#define COMPUTE_TIME(t)						\
  timersub(&t##_end,&t##_beg,&t##_res)
#else
#define DECLARE_TIME(t)
#define TIME_ST(t)
#define TIME_END(t)
#define SHOW_TIME(t)
#define SHOW_SIMPLE_TIME(t)
#endif


#endif
