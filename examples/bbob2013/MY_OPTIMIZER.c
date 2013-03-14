
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

void MY_OPTIMIZER(double(*fitnessfunction)(double*), unsigned int dim, double ftarget, double maxfunevals)
{
    double * x = (double *)malloc(sizeof(double) * dim);
    double f;
    double iter;
    unsigned int j;

    if (maxfunevals > 1000000000. * dim)
        maxfunevals = 1000000000. * dim;

    for (iter = 0.; iter < maxfunevals; iter++)
    {
        /* Generate individual */
        for (j = 0; j < dim; j++)
             x[j] = 10. * ((double)rand() / RAND_MAX) - 5.;

        /* evaluate x on the objective function */
        f = fitnessfunction(x);

        if (f < ftarget)
            break;
    }
    free(x);
}
