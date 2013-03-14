#ifndef _bbobStructures_H
#define _bbobStructures_H

/* some structures, to try to imitate that gorgious Matlab Code !!! 
*/

/* sometimes, M_PI is not defined ????? */
#ifndef M_PI
  #define M_PI        3.14159265358979323846
#endif

/* the return type of all benchmark functions: 2 doubles, the true value and the noisy value - are equal in case of non-noisy functions */
struct twoDoubles {
double Ftrue;
double Fval;
};

typedef struct twoDoubles TwoDoubles;

/* and now the type of the benchmark functions themselves */
typedef struct twoDoubles (*bbobFunction)(double *);

/* need to put all parameters into a single structure that 
can be passed around 
*/
/* all static char* (increase if your system uses huge file names) */
#define DefaultStringLength 1024
/* for not having to allocate the x in struct  lastEvalStruct */
#define DIM_MAX 200
/* These ones might be defined somewhere in the C headers, but in case it's system-dependent ... */
/*#define MAX_FLOAT 1.0E308
#define MIN_FLOAT 1.0E-308
#define MAX_INT 32767*/

void fgeneric_initialize();
double fgeneric_finalize(void);
double fgeneric_evaluate(double * X);
double fgeneric_ftarget(void);

#endif
