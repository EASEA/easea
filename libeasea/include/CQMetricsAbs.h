/*
 * CQMetricsAbs.h
 *
 *  Created on: 31 août 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CQMETRICSABS_H
#define CQMETRICSABS_H

#include <CQMetricsFunctions.h>
#include <vector>
#include <iostream>
#include <cstdlib>

//using namespase std;

class CQMetricsAbs {

private :
    CQMetricsFunctions * functions;

public :
    CQMetricsFunctions *getFunctions(){ return functions; };
    CQMetricsAbs() { functions = new CQMetricsFunctions(); }
    virtual ~CQMetricsAbs(){ delete functions; };
//    virtual double get(vector< vector<double> > takenFront, vector< vector<double> > realFront, int nbObj) {};

};
#endif /* CQMETRICSABS */

