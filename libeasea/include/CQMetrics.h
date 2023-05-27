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


class CQMetrics : public CQMetricsAbs {

private:
	int nbObj;
	std::vector<std::vector<double>> paretoTakenFront;
	std::vector<std::vector<double>> paretoRealFront;

public:
	CQMetrics(std::string const& paretoTakenFrontFile, std::string const& paretoRealFrontFile, int nbObj) : CQMetricsAbs() {
		this->nbObj = nbObj;
		paretoTakenFront = getFunctions()->readFrontFromFile(paretoTakenFrontFile);
		paretoRealFront = getFunctions()->readFrontFromFile(paretoRealFrontFile);
	};


	~CQMetrics(){};
	template <class Tclass>
	double getMetric(){
		auto metric = std::make_unique<Tclass>();
		return metric->get(paretoTakenFront, paretoRealFront, nbObj);
	}

};

#endif /* CQMETRICS_H */
