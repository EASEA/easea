/*
 * CQMetricsGD.cpp
 *
 *  Created on: 31 août 2018
 *      Author: Anna Ouskova Leonteva
 */
#include <iostream>


#include <CQMetricsGD.h>

    CQMetricsGD::CQMetricsGD():CQMetricsAbs() {};

    CQMetricsGD::~CQMetricsGD() {};

    double CQMetricsGD::get(vector< vector<double> > takenFront, vector< vector<double> > realFront, int nbObj) {

        vector<double> maxValues = getFunctions()->getMaxValues(realFront, nbObj);
        vector<double> minValues = getFunctions()->getMinValues(realFront, nbObj);

        vector< vector<double> > normalizedTakenFront = getFunctions()->getNormalizedFront(takenFront,
                                                maxValues,
                                                minValues);
        vector< vector<double> >  normalizedRealFront = getFunctions()->getNormalizedFront(realFront,
                                                maxValues,
                                                minValues);

        double sum = 0.0;
        for (auto i = 0; i < takenFront.size(); i++)
            sum += power<c_pow>(getFunctions()->distanceToClosedPoint(normalizedTakenFront[i],
                                             normalizedRealFront));


        sum = pow(sum,1.0/c_pow);

        double metricGD = sum / normalizedTakenFront.size();

        return metricGD;



    };
