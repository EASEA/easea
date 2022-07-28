/*
 * CCDGMutation.H
 *
 *  Created on: 12 octobre 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CCDGMutationMUTATION_H
#define CCDGMutationMUTATION_H

#include <math.h>
#include <float.h>

#include <COperator.h>
#include <CPseudoRandom.h>
#include <CWrapIndividual.h>

#include <sstream>
#include <memory>


constexpr double DEFAULT_PROBABILITY_ = 0.01;
constexpr double DEFAULT_DELTA_ = 0.5;

template<typename TIndividual>
class CCDGMutation :  public COperator<double, double>  {
protected:

public :
    CCDGMutation() : COperator(DEFAULT_PROBABILITY_, DEFAULT_DELTA_ ){};
    CCDGMutation(double probability, double delta) : COperator(probability, delta) {
        std::ostringstream ss;
        ss << "EASEA LOG [DEBUG]: Mutation " <<  "probability  =  "<< get<double>(0) << std::endl
        << "EASEA LOG [DEBUG]: Mutation "   << "delta = " << get<double>(1) << std::endl;

        LOG_MSG(msgType::DEBUG, ss.str());

    };

    ~CCDGMutation(){};

    TIndividual * run(const TIndividual * individual){
        mutation(get<double>(0), get<double>(1), individual);
        return individual;
    };

private:

    TIndividual * mutation(const double &probability, const double &delta, const TIndividual * individual){
        double rnd, tmpDelta, deltaq;
        double y, yl, yu, val, xy;

        std::unique_ptr<CWrapIndividual>  x = std::make_unique<CWrapIndividual>(individual);

        for (auto var=0; var < individual->getNumberOfVariables(); var++) {
            if ( CPseudoRandom::randDouble() <= probability) {
                y  = x->getValue(var);
                yl = x->getLowerBound(var);
                yu = x->getUpperBound(var);
                rnd = CPseudoRandom::randDouble();
		tmpDelta = pow(rnd, -delta);
		deltaq = 0.5 * (rnd - 0.5)*(1-tmpDelta);
		y = y + deltaq * (yu - yl);

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
