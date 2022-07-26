/*
 * CQMetricsFunctions.h
 *
 *  Created on: 31 août 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CQMETRICSFUNCTIONS_H
#define CQMETRICSFUNCTIONS_H

#include <limits>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>

#include "CLogger.h"
#include "CMeta.h"

using namespace std;

class CQMetricsFunctions{
public:
    /* Read Pareto front from file */
	vector< vector<double> > readFrontFromFile(string path){

        vector< vector <double> > front;

        std::ifstream in(path.c_str());
        if( !in )
           LOG_ERROR(errorCode::io, "impossible to read file");

        string line;

        while( getline(in, line ) ) {

            vector<double> list;

            istringstream iss(line);

            while (iss) {
                string token;
                iss >> token;
                if (token.compare("")!=0)
                    list.push_back(atof(token.c_str()));
            }
            front.push_back(list);
        }

        in.close();

        return front;
    };

    vector<double> getMaxValues(vector< vector<double> > const& front, int nbObj) {

        vector<double> maxValues;

	    for (auto i = 0; i < nbObj; i++)
		    maxValues.push_back(-std::numeric_limits<double>::max());

	    for (const auto& row : front){
             for (size_t j = 0; j < row.size(); j++) {
	
                    if (row[j] > maxValues[j])
				    maxValues[j] = row[j];
		    }
	    }

	    return maxValues;
    };

    vector<double> getMinValues(vector< vector<double> > const& front, int nbObj) {

        vector<double> minValues;

       for (auto i = 0; i < nbObj; i++)
            minValues.push_back(std::numeric_limits<double>::max());

        for (const auto& row : front){
            for (size_t j = 0; j < row.size(); j++) {
                if (row[j] < minValues[j])
                    minValues[j] = row[j];
             }
        }
        return minValues;
    };

	double distance(vector<double> const& a, vector<double> const& b){

        double distance = 0.0;

        for (size_t i = 0; i < a.size(); i++)
            distance += power<2>(a[i]-b[i]);


        return sqrt(distance);
    };

    double distanceToClosedPoint(vector<double> const& point, vector< vector<double> > const& front){

        double minDistance = distance(point,front[0]);

          for (const auto& item : front){
            double aux = distance(point,item);
            if (aux < minDistance)
                minDistance = aux;

        }
        return minDistance;
    };

    vector< vector<double> > getNormalizedFront(vector< vector<double> > const& front, vector<double> const& maxValue, vector<double> const& minValue) {

        vector< vector<double> > normalizedFront;

        for (const auto& row : front){
            vector<double> list;
            for (size_t j = 0; j < row.size(); j++)
                list.push_back((row[j] - minValue[j]) / (maxValue[j] - minValue[j]));

            normalizedFront.push_back(list);
        }
        return normalizedFront;
    };

	vector< vector<double> > getInvertedFront(vector< vector<double> > const& front) {

	vector< vector<double> > invertedFront;

        for (const auto& row : front){
            vector<double> list;
            for( const auto& item : row){
                if (item <= 1.0 && item >= 0.0)
			list.push_back(1.0 - item);
		else if (item > 1.0)
			list.push_back(0.0);
		else if (item < 0.0)
			list.push_back(1.0);
		    }
		    invertedFront.push_back(list);
	    }

	    return invertedFront;
    };

};

#endif /*CQMETRICSFUNCTIONS_H  */
