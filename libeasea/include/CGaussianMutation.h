/*
 * CPolynomialMutation.H
 *
 *  Created on: 11 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CGAUSSIANMUTATION_H
#define CGAUSSIANMUTATION_H

#include <math.h>
#include <float.h>
#include <random>
#include <COperator.h>
#include <CPseudoRandom.h>
#include <CWrapIndividual.h>

#include <sstream>
#include <memory>


//constexpr double eta_m_ = 0.05;

template<typename TIndividual>
class CGaussianMutation :  public COperator<double, double>  {
public :
    CGaussianMutation() : COperator(){};
    CGaussianMutation(double probability, double deviation) : COperator(probability, deviation) {
        std::ostringstream ss;
        ss << "EASEA LOG [DEBUG]: Mutation " <<  "probability  =  "<< get<double>(0) << std::endl
	<< "EASEA LOG [DEBUG]: Mutation " <<  "deviation  =  "<< get<double>(1) << std::endl;
        LOG_MSG(msgType::DEBUG, ss.str());
        
    };

    ~CGaussianMutation(){};

    const TIndividual * run(const TIndividual * individual){
        mutation( get<double>(0), get<double>(1), individual);
        return individual;
    };

private:
    double static nextGaussian(){
	double v1, v2, s;
	do {
	    v1 = 2.0 * CPseudoRandom::randDouble() - 1.0;
	    v2 = 2.0 * CPseudoRandom::randDouble() - 1.0;
	    s = v1 * v1 + v2 *v2;
	}while (s >= 1.0 || s == 0);
	s = std::sqrt((-2.0 * std::log(s))/s);
	return v1 * s;
    }
    double static nextGaussian1(double mean, double deviation){
	return mean + nextGaussian()*deviation;
    }
    double nextGaussian2(double mean, double deviation){
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, deviation);
	return distribution(generator);
    }

//    TIndividual * 
	void mutation(const double &probability, const double &deviation, const TIndividual * individual){
        double y, yl, yu;
	double sigma;
        std::unique_ptr<CWrapIndividual>  x = std::make_unique<CWrapIndividual>(individual);

        for (auto var=0; var < individual->getNumberOfVariables(); var++) {
            if ( CPseudoRandom::randDouble() <= probability)  {
/*		if (probability < 0.35)
		    sigma = deviation * 0.85;
		else if (probability > 0.35)
		    sigma = deviation/0.85;
		else*/ sigma = deviation*0.85;
		double diff = nextGaussian1(0.1,sigma);
//		diff = (diff*0.1)+0.1;
		y = x->getValue(var);
                yl = x->getLowerBound(var);
                yu = x->getUpperBound(var);
//		double diff = nextGaussian1((yu+yl)/2, deviation);

                y = y + diff;
                if (y < yl)
                    y = yl;
                if (y > yu)
                    y = yu;
                x->setValue(var, y);
            }
        }

    };
};

#endif
