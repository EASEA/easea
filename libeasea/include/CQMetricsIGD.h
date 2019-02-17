/*
 * CQMetricsIGD.h
 *
 *  Created on: 2 septembre 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CQMETRICSIGD_H
#define CQMETRICSIGD_H

#include <CMeta.h>
#include <CQMetricsAbs.h>
#include <vector>
#include <cstdlib>

/* Inverted Generational Distance */
class CQMetricsIGD : public CQMetricsAbs{

    static const int c_pow = 2;

public:

    CQMetricsIGD();
    ~CQMetricsIGD();

    double get(vector< vector<double> > takenFront, vector< vector<double> > realFront, int nbObj);

};

#endif/* CQMETRICSIGV */
