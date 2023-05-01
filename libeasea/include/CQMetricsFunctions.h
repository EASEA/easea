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


class CQMetricsFunctions{
public:
    /* Read Pareto front from file */
	std::vector< std::vector<double> > readFrontFromFile(std::string const& path){

        std::vector< std::vector <double> > front;

        std::ifstream in(path.c_str());
        if( !in )
           LOG_ERROR(errorCode::io, "impossible to read file");

        std::string line;

        while( std::getline(in, line ) ) {

            std::vector<double> list;

            std::stringstream iss(line);

            while (iss) {
                std::string token;
                iss >> token;
                if (token.compare("")!=0)
                    list.push_back(atof(token.c_str()));
            }
            front.push_back(list);
        }

        in.close();

        return front;
    };

    std::vector<double> getMaxValues(std::vector< std::vector<double> > const& front, int nbObj) {

        std::vector<double> maxValues;
	maxValues.reserve(nbObj);

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

    std::vector<double> getMinValues(std::vector< std::vector<double> > const& front, int nbObj) {

        std::vector<double> minValues;
	minValues.reserve(nbObj);

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

	double distance(std::vector<double> const& a, std::vector<double> const& b){

        double distance = 0.0;

        for (size_t i = 0; i < a.size(); i++)
            distance += power<2>(a[i]-b[i]);


        return sqrt(distance);
    };

    double distanceToClosedPoint(std::vector<double> const& point, std::vector< std::vector<double> > const& front){

        double minDistance = distance(point,front[0]);

          for (const auto& item : front){
            double aux = distance(point,item);
            if (aux < minDistance)
                minDistance = aux;

        }
        return minDistance;
    };

    std::vector< std::vector<double> > getNormalizedFront(std::vector< std::vector<double> > const& front, std::vector<double> const& maxValue, std::vector<double> const& minValue) {

        std::vector< std::vector<double> > normalizedFront;

        for (const auto& row : front){
            std::vector<double> list;
            for (size_t j = 0; j < row.size(); j++)
                list.push_back((row[j] - minValue[j]) / (maxValue[j] - minValue[j]));

            normalizedFront.push_back(list);
        }
        return normalizedFront;
    };

	std::vector< std::vector<double> > getInvertedFront(std::vector< std::vector<double> > const& front) {

	std::vector< std::vector<double> > invertedFront;

        for (const auto& row : front){
            std::vector<double> list;
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
