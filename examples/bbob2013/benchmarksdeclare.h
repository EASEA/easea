#ifndef INCLUDE_BDECLARE_H
#define INCLUDE_BDECLARE_H

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

#define SIZE 2
#define DIM SIZE
#define BBOB_FUNCTION_ID 1
#define BBOB_INSTANCE_ID 1
#define X_MIN -5.
#define X_MAX 5.
#define ITER 120      
#define Abs(x) ((x) < 0 ? -(x) : (x))
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define SIGMA  1.                     /*  mutation parameter */
#define PI 3.141592654

/*Global declarations */
extern int trialid;
extern int funcId; //passed by parameter
extern int instanceId;
extern double * peaks;
extern double * Xopt;
extern double Fopt;
extern unsigned int isInitDone;

#endif
