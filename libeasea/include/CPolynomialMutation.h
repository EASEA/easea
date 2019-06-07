/*
 * CPolynomialMutation.H
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CPOLYNOMIALMUTATION_H
#define CPOLYNOMIALMUTATION_H

#include <math.h>
#include <float.h>

#include <COperator.h>
#include <CPseudoRandom.h>
#include <CWrapIndividual.h>


constexpr double eta_m_ = 20.0;

template<typename TIndividual>
class CPolynomialMutation :  public COperator<double, double>  {
protected:

public :
    CPolynomialMutation() : COperator(1.0, eta_m_){};
    CPolynomialMutation(double probability, double distributionId) : COperator(probability, distributionId) {
        std::ostringstream ss;
        ss << "EASEA LOG [DEBUG]: Mutation " <<  "probability  =  "<< get<double>(0) << std::endl
        << "EASEA LOG [DEBUG]: Mutation "   << "distribution Index = " << get<double>(1) << std::endl;

        LOG_MSG(msgType::DEBUG, ss.str());

    };

    ~CPolynomialMutation(){};

    TIndividual * run( TIndividual * individual){
        mutation(get<double>(0), get<double>(1), individual);
        return individual;
    };

private:

    void  mutation(const double &probability, const double &distributionId,TIndividual * individual){
        double rnd, delta1, delta2, mut_pow, deltaq;
        double y, yl, yu, val, xy;

        std::unique_ptr<CWrapIndividual>  x = std::make_unique<CWrapIndividual>(individual);

        for (auto var=0; var < individual->getNumberOfVariables(); var++) {
            if ( CPseudoRandom::randDouble() <= probability) {
                y  = x->getValue(var);
                yl = x->getLowerBound(var);
                yu = x->getUpperBound(var);
                delta1 = (y - yl)/(yu - yl);
                delta2 = (yu - y)/(yu - yl);
                rnd = CPseudoRandom::randDouble();
                mut_pow = 1.0/(distributionId+1.0);
                if (rnd <= 0.5) {
                    xy = 1.0 - delta1;
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy,(eta_m_ + 1.0)));
                    deltaq = pow(val,mut_pow) - 1.0;
                } else {
                    xy = 1.0 - delta2;
                    val    = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5)*(pow(xy,(eta_m_ + 1.0)));
                    deltaq = 1.0 - (pow(val,mut_pow));
                }
                y = y + deltaq * (yu - yl);
                y  = x->getValue(var);
                yl = x->getLowerBound(var);
                yu = x->getUpperBound(var);
                delta1 = (y - yl)/(yu - yl);
                delta2 = (yu - y)/(yu - yl);
                rnd = CPseudoRandom::randDouble();
                mut_pow = 1.0/(distributionId + 1.0);
                if (rnd <= 0.5) {
                    xy = 1.0 - delta1;
                    val    = 2.0 * rnd+(1.0 - 2.0*rnd)*(pow(xy,(eta_m_ + 1.0)));
                    deltaq = pow(val,mut_pow) - 1.0;
                } else {
                    xy = 1.0 - delta2;
                    val    = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5)*(pow(xy,(eta_m_ + 1.0)));
                    deltaq = 1.0 - (pow(val,mut_pow));
                }
                y = y + deltaq * (yu - yl);
                if (y < yl)
                    y = yl;
                if (y > yu)
                    y = yu;
                x->setValue(var, y);
            }
        }

    }
};

#endif
