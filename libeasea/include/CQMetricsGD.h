/*
 * CQMetricsGD.h
 *
 *  Created on: 2 septembre 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CQMETRICSGD_H
#define CQMETRICSGD_H

#include <CQMetricsAbs.h>
#include <vector>
/* Generational Distance */
using namespace std;

class CQMetricsGD :  public CQMetricsAbs {
    static const int c_pow = 2;
public:
    CQMetricsGD();
    ~CQMetricsGD();

    double get(vector< vector<double> > const& takenFront, vector< vector<double> > const& realFront, int nbObj);

};

#endif/* CQMETRICSGD */
