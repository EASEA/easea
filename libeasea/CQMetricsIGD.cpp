/*
 * CQMetricsIGD.cpp
 *
 *  Created on: 31 août 2018
 *      Author: Anna Ouskova Leonteva
 */
#include <iostream>

#include <CQMetricsIGD.h>

    CQMetricsIGD::CQMetricsIGD():CQMetricsAbs() {}

    CQMetricsIGD::~CQMetricsIGD() {}

    double CQMetricsIGD::get(std::vector< std::vector<double> > const& takenFront, std::vector< std::vector<double> > const& realFront, int nbObj) {
        std::vector<double> maxValues = getFunctions()->getMaxValues(realFront, nbObj);
        std::vector<double> minValues = getFunctions()->getMinValues(realFront, nbObj);

        std::vector< std::vector<double> > normalizedTakenFront = getFunctions()->getNormalizedFront(takenFront,
                                                maxValues,
                                                minValues);
        std::vector< std::vector<double> >  normalizedRealFront = getFunctions()->getNormalizedFront(realFront,
                                                maxValues,
                                                minValues);

        double sum = 0.0;
        for (std::size_t i = 0; i < normalizedRealFront.size(); i++)
            sum += power<c_pow>(getFunctions()->distanceToClosedPoint(normalizedRealFront[i],
                                             normalizedTakenFront));


        sum = pow(sum,1.0/c_pow);

        double metricIGD = sum / static_cast<double>(normalizedRealFront.size());

        return metricIGD;

    }

