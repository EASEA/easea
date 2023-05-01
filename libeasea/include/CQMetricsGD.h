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

class CQMetricsGD :  public CQMetricsAbs {
    static const int c_pow = 2;
public:
    CQMetricsGD();
    ~CQMetricsGD();

    double get(std::vector< std::vector<double> > const& takenFront, std::vector< std::vector<double> > const& realFront, int nbObj);

};

#endif/* CQMETRICSGD */
