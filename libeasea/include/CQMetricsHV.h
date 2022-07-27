/*
 * CQMetricsHV.h
 *
 *  Created on: 31 août 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CQMETRICSHV_H
#define CQMETRICSHV_H

#include <CQMetricsAbs.h>

/* Hyper volume */


class CQMetricsHV : public CQMetricsAbs {

private:

    bool dominates(double* point1, double* point2, int nbObj);

    void swap(double** front, int id_1, int id_2);

    int  filterNondominatedPop(double** front, int noPoints, int nbObj);

    double surfaceUnchangedTo(double** front, int nbPoints, int objective);

    int reduceNondominatedSet(double** front, int nbPoints, int objective, double threshold);



public:

    CQMetricsHV();
    ~CQMetricsHV();

    double calculate(double** front, int nbPoints, int nbObj);
    double get(vector< vector<double> > const& takenFront, vector< vector<double> > const& realFront, int nbObj);

};

#endif /* CQMETRICSHV */
