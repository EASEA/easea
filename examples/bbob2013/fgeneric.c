/*
Loosely inspired by fgeneric.m, the matlab version

%    Example: Optimize function f11 with MATLABS FMINUNC:
%
%       DIM = 5;
%       ftarget = fgeneric('initialize', 11, 1, 'testfminunc');
%       opt = optimset('TolFun', 1e-11);
%       X = fminunc('fgeneric', 8*rand(DIM,1) - 4, opt);
%       disp(fgeneric(X) - ftarget);
%       fgeneric('finalize');
%
%    This will create folder testfminunc. In this folder, the info file
%    bbobexp_f11.info will provide meta-information on the different
%    optimization runs obtained. Data of these runs are located in folder
%    testfminunc/data_f11 which will contain the file bbobexp_f11_DIM5.dat.
%
%
    fgeneric.m
    Author: Raymond Ros, Nikolaus Hansen, Steffen Finck, Marc Schoenauer
        (firstName.lastName@lri.fr)
    Version = 'Revision: $Revision: 646 $'
    Last Modified: $Date: 2009-02-18 13:54:40 +0100 (Wed, 18 Feb 2009) $
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <string.h>

/*int strcasecmp (const char *s1, const char *s2);*/
/*int strncasecmp (const char *s1, const char *s2, size_t n);*/

/* the specific includes for BBOB */
#include "bbobStructures.h"
#include "benchmarksdeclare.h"
#include "benchmarkshelper.h"
#include "benchmarks.h"
#include "benchmarksnoisy.h"


/* The following are some of the global static variables of the Matlab program */
  /*
  int DIM;
  unsigned int isInitDone;
  unsigned int trialid;
  double Fopt; */
  int initDone = 0;

  bbobFunction actFunc = NULL;



int fgeneric_exist(unsigned int FUNC_ID)
{
    if ( (FUNC_ID <= handlesLength ) ||
         ( (100 < FUNC_ID) && (FUNC_ID <= 100+handlesNoisyLength) )
       )
        return 1;
    return 0;
}


void fgeneric_initialize()
{
    double * X;
    TwoDoubles res;


    if (DIM == 0);
        //ERROR("You need to set the dimension of the problem greater than 0");

    if (DIM  > DIM_MAX);
        //ERROR("You need to recompile the program to increase DIM_MAX if you want to run with dimension %d\nPlease also pay attention at the definition of lastEvalInit", DIM);

    /* Once the dimension is known, do the memory allocation if needed */
      initbenchmarkshelper();
      initbenchmarks();
      initbenchmarksnoisy();
    /* tests the func ID and sets the global variable */
    /**************************************************/
    if (funcId-1 < handlesLength )
        actFunc = handles[funcId-1];
    else
    {
        if ( (100 < funcId) && (funcId-101 < handlesNoisyLength) )
            actFunc = handlesNoisy[funcId - 101];
        else;
            //ERROR("funcId in PARAMS is %d which is not a valid function identifier", funcId);
    }


    /* the target value and book-keeping stuffs related to stopping criteria */
    /************************************************************************/
    Fopt = computeFopt(funcId, instanceId);

    printf("Fopt for functionid %i is %8.4f\n", funcId, Fopt);
    /* These lines are used to align the call to myrand with the ones in Matlab.
     * Don't forget to declare X (double * X;) and res (TwoDoubles res).*/
    X = (double*)malloc(DIM * sizeof(double));
    res = (*actFunc)(X);
    free(X);
}
/* --------------------------------------------------------------------*/

/* Sets the seed of the noise (in benchmarkshelper) with a modulo 1e9 (constraint
 * on the seed provided to benchmarkshelper, this can happen when using time(NULL)).
 */
void fgeneric_noiseseed(unsigned long seed)
{
    seed = seed % 1000000000;
    setNoiseSeed(seed, seed);
}


double fgeneric_evaluate(double * X)
{
    double Fvalue, Ftrue;
    TwoDoubles res;

    if (actFunc == NULL);
        //ERROR("fgeneric has not been initialized. Please call 'fgeneric_initialize' first.");

    res = (*actFunc)(X);
    Fvalue = res.Fval;
    Ftrue = res.Ftrue;
    return Fvalue;
}
