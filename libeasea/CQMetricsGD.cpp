/*
 * CQMetricsGD.cpp
 *
 *  Created on: 31 août 2018
 *      Author: Anna Ouskova Leonteva
 */
#include <iostream>


#include <CQMetricsGD.h>

    CQMetricsGD::CQMetricsGD():CQMetricsAbs() {}

    CQMetricsGD::~CQMetricsGD() {}

    double CQMetricsGD::get(std::vector< std::vector<double> > const& takenFront, std::vector< std::vector<double> > const& realFront, int nbObj) {

        std::vector<double> maxValues = getFunctions()->getMaxValues(realFront, nbObj);
        std::vector<double> minValues = getFunctions()->getMinValues(realFront, nbObj);

        std::vector< std::vector<double> > normalizedTakenFront = getFunctions()->getNormalizedFront(takenFront,
                                                maxValues,
                                                minValues);
        std::vector< std::vector<double> >  normalizedRealFront = getFunctions()->getNormalizedFront(realFront,
                                                maxValues,
                                                minValues);

        double sum = 0.0;
        for (decltype(takenFront.size()) i = 0; i < takenFront.size(); i++)
            sum += power<c_pow>(getFunctions()->distanceToClosedPoint(normalizedTakenFront[i],
                                             normalizedRealFront));


        sum = pow(sum,1.0/c_pow);

        double metricGD = sum / static_cast<double>(normalizedTakenFront.size());

        return metricGD;
    }
