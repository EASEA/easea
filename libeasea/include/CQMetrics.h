/*
 * CQMetrics.h
 *
 *  Created on: 4 septembre 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CQMETRICS_H
#define CQMETRICS_H

#include <string>

#include <CQMetricsAbs.h>
#include <memory>

using namespace std;

class CQMetrics : public CQMetricsAbs {

private:
	int nbObj;
	vector<vector<double>> paretoTakenFront;
	vector<vector<double>> paretoRealFront;

public:
	CQMetrics(string const& paretoTakenFrontFile, string const& paretoRealFrontFile, int nbObj) : CQMetricsAbs() {
		this->nbObj = nbObj;
		paretoTakenFront = getFunctions()->readFrontFromFile(paretoTakenFrontFile);
		paretoRealFront = getFunctions()->readFrontFromFile(paretoRealFrontFile);
	};


	~CQMetrics(){};
	template <class Tclass>
	double getMetric(){
		auto metric = make_unique<Tclass>();
		return metric->get(paretoTakenFront, paretoRealFront, nbObj);
	};

};

#endif /* CQMETRICS_H */
