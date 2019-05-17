/*
 * CQMetricsIGD.cpp
 *
 *  Created on: 31 août 2018
 *      Author: Anna Ouskova Leonteva
 */
#include <iostream>

#include <CQMetricsIGD.h>

    CQMetricsIGD::CQMetricsIGD():CQMetricsAbs() {};

    CQMetricsIGD::~CQMetricsIGD() {};

    double CQMetricsIGD::get(vector< vector<double> > takenFront, vector< vector<double> > realFront, int nbObj) {
        vector<double> maxValues = getFunctions()->getMaxValues(realFront, nbObj);
        vector<double> minValues = getFunctions()->getMinValues(realFront, nbObj);

        vector< vector<double> > normalizedTakenFront = getFunctions()->getNormalizedFront(takenFront,
                                                maxValues,
                                                minValues);
        vector< vector<double> >  normalizedRealFront = getFunctions()->getNormalizedFront(realFront,
                                                maxValues,
                                                minValues);

        double sum = 0.0;
        for (auto i = 0; i < normalizedRealFront.size(); i++)
            sum += power<c_pow>(getFunctions()->distanceToClosedPoint(normalizedRealFront[i],
                                             normalizedTakenFront));


        sum = pow(sum,1.0/c_pow);

        double metricIGD = sum / normalizedRealFront.size();

        return metricIGD;

    };

