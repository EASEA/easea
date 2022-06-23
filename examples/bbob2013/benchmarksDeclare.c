#include "benchmarksdeclare.h"

int DIM = SIZE;
int trialid = BBOB_INSTANCE_ID;
int funcId = BBOB_FUNCTION_ID; //passed by parameter
int instanceId = BBOB_INSTANCE_ID;

double * peaks;
double * Xopt; /*Initialized in benchmarkhelper.c*/
double Fopt;
unsigned int isInitDone=0;

