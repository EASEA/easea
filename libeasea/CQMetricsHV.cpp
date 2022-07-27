/*
 * CQMetricsHV.cpp
 *
 *  Created on: 31 août 2018
 *      Author: Anna Ouskova Leonteva
 */
#include <iostream>

#include <CQMetricsHV.h>

    CQMetricsHV::CQMetricsHV():CQMetricsAbs() {};

    CQMetricsHV::~CQMetricsHV() {};

    bool  CQMetricsHV::dominates(double* point1, double* point2, int nbObj) {

        bool betterInAnyObjective = false;
        int i = 0;

        for (i = 0; i < nbObj && point1[i] >= point2[i]; i++) {
            if (point1[i] > point2[i]) {
                betterInAnyObjective = true;
            }
        }

        return ((i >= nbObj) && betterInAnyObjective);
    }

    void CQMetricsHV::swap(double** front, int id_1, int id_2) {

        double* temp = front[id_1];

        front[id_1] = front[id_2];
        front[id_2] = temp;
    }

    int CQMetricsHV::filterNondominatedPop(double** front, int noPoints, int nbObj) {

        int i = 0;
        int j = 0;
        int n = noPoints;

        while (i < n) {
            j = i + 1;
            while (j < n) {
                if (dominates(front[i], front[j], nbObj)) {
                    n--;
                    swap(front, j, n);
                } else if (dominates(front[j], front[i], nbObj)) {
                    n--;
                    swap(front, i, n);
                    i--;
                    break;
                } else
                    j++;
            }
            i++;
        }

        return n;
    }

    double CQMetricsHV::surfaceUnchangedTo(double** front, int nbPoints, int objective) {

        if (nbPoints < 1) {
            cout << "run-time error" << endl;
            exit(-1);
    }

        double minValue = front[0][objective];
        for (auto i = 1; i < nbPoints; i++) {
            double tmpValue = front[i][objective];
            if (tmpValue < minValue)
                minValue = tmpValue;
        }
        return minValue;
    }

    int CQMetricsHV::reduceNondominatedSet(double** front, int nbPoints, int objective, double threshold) {

        int n = nbPoints;

        for (int i = 0; i < n; i++) {
            if (front[i][objective] <= threshold) {
                n--;
                swap(front, i, n);
            }
        }
        return n;
    }



    double CQMetricsHV::calculate(double** front, int nbPoints, int nbObj) {

        double volume = 0.0;
        double distance = 0.0;
        int n = nbPoints;

        while (n > 0) {

            int nbNondominatedPoints = filterNondominatedPop(front, n, nbObj - 1);
            double tmpVolume = 0.0;
            if (nbObj < 3) {
                if (nbNondominatedPoints < 1) {
                    cout << "run-time error" << endl;
                    exit(-1);
                }
                tmpVolume = front[0][0];
            } else
                tmpVolume = calculate(front, nbNondominatedPoints, nbObj - 1);

            double tmpDistance = surfaceUnchangedTo(front, n, nbObj - 1);
            volume += tmpVolume * (tmpDistance - distance);
            distance = tmpDistance;
            n = reduceNondominatedSet(front, n, nbObj - 1, distance);
        }
        return volume;
    }

    double CQMetricsHV::get(vector< vector<double> > const& takenFront, vector< vector<double> > const& realFront, int nbObj) {

        vector<double> maxValues = getFunctions()->getMaxValues(realFront, nbObj);
        vector<double> minValues = getFunctions()->getMinValues(realFront, nbObj);

        vector< vector<double> > normalizedFront = getFunctions()->getNormalizedFront(takenFront, maxValues, minValues);
        vector< vector<double> > invertedFront = getFunctions()->getInvertedFront(normalizedFront);

        double ** invertedFront2 = new double*[invertedFront.size()];
        for (std::size_t i = 0; i < invertedFront.size(); i++) {
            invertedFront2[i] = new double[invertedFront[i].size()];
            for (std::size_t j = 0; j < invertedFront[i].size(); j++)
                invertedFront2[i][j] = invertedFront[i][j];
        }

        double hv = calculate(invertedFront2,static_cast<int>(invertedFront.size()),nbObj);

        for(auto y = 0 ; y < invertedFront.size() ; y++ )
            delete [] invertedFront2[y] ;

        delete [] invertedFront2;

        return hv;
    }




