/*
 * CBLXBetaCrossover.h 
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CBLXBETACROSSOVER_H
#define CBLXBETACROSSOVER_H

#include <math.h>

#include <COperator.h>
#include <CPseudoRandom.h>
#include <CWrapIndividual.h>

#include <sstream>


/*  In this class a Blind Alpha Beta Crossover is realised
 *  It could be used for any kind types of variables,
 *  because class CWrapIndividual was implemented
 *  */
constexpr double DEFAULT_ALPHA_ = 0.5;  
constexpr double DEFAULT_BETA_ = 0.33;

template<typename TIndividual>
class CBLXBetaCrossover : public COperator<double,double,double> {

public:
    /* In Constructor:
     * First Parameter - Probability
     * Second Parameter - Distribution Index
     */
    CBLXBetaCrossover() : COperator(0.9, DEFAULT_ALPHA_, DEFAULT_BETA_){};

    CBLXBetaCrossover(const double &probability, const double &alpha, const double &beta) : COperator(probability, alpha, beta) {
        std::ostringstream ss;

	if (probability < 0){
		ss << "Wrong value of crossover probability: it's < 0" << std::endl;
		LOG_ERROR(errorCode::value, ss.str());
        }
	if (alpha < 0){
		ss << "Wrong value of crossover alpha: it's < 0" << std::endl;
		LOG_ERROR(errorCode::value, ss.str());
	}
	if (beta < 0){
		ss << "Wrong value of crossover beta: it's < 0" << std::endl;
		LOG_ERROR(errorCode::value, ss.str());
	}

	ss << "EASEA LOG [DEBUG]: Crossover: " <<  "probability  =  "<< get<double>(0) << std::endl
        << "EASEA LOG [DEBUG]: Crossover: "   << "alpha = " << get<double>(1) << std::endl
	<< "EASEA LOG [DEBUG]: Crossover: "   << "beta = " << get<double>(2) << std::endl;
	
        LOG_MSG(msgType::DEBUG, ss.str());
    };

    ~CBLXBetaCrossover(){}

    TIndividual ** run(const TIndividual *const * parents){
        return crossover(get<double>(0), get<double>(1), get<double>(2),parents[0], parents[1]);
    }

private:
    
    TIndividual ** crossover(const double &probability, const double &alpha, const double &beta, const TIndividual * parent1, const TIndividual * parent2){

        TIndividual** offSpring = new TIndividual*[2];

        if (offSpring == nullptr)
            LOG_ERROR(errorCode::memory,"memory for offsprings individuals wasn't allocated!");

        offSpring[0] = new TIndividual(parent1);
        offSpring[1] = new TIndividual(parent2);

        double rand;
        double i; 
        double y2; 
        double y1;
        double x2;
        double x1; 
        double upBound; 
        double lowBound;

        auto ptrParent1 = make_unique<CWrapIndividual>(parent1);
        auto ptrParent2 = make_unique<CWrapIndividual>(parent2);
        auto ptrOffspring1 = make_unique<CWrapIndividual>(offSpring[0]);
        auto ptrOffspring2 = make_unique<CWrapIndividual>(offSpring[1]);

        int numberOfVariables = ptrParent1->getNumberOfIndividualVariables();

        if (CPseudoRandom::randDouble() <= probability){
            for (auto i=0; i<numberOfVariables; i++){
		upBound = ptrParent1->getUpperBound(i);
		lowBound = ptrParent2->getLowerBound(i);
		x1 = ptrParent1->getValue(i);
		x2 = ptrParent2->getValue(i);

		double max = 0;
		double min = 0;
		double range = abs(x1 - x2);
		if ( x1 <= x2 ){
		    min = x1 - range * alpha;
		    max = x2 + range * beta;
		}else{
		    min = x2 - range * beta;
		    max = x1 + range * alpha;
		}
		rand = CPseudoRandom::randDouble();
		y1 = min + rand * (max - min);
		rand = CPseudoRandom::randDouble();
		y2 = min + rand * (max - min);

		
/*
		if (x2 > x1){
			max = x2;
			min = x1;
		}else {
			max = x1;
			min = x2;
		}
		double range = max - min; 
		
		double minRange = min - range * alpha;
		double maxRange = max + range * alpha;


                rand = CPseudoRandom::randDouble();
		y1 = minRange + rand * (maxRange - minRange);
		rand = CPseudoRandom::randDouble();
		y2 = minRange + rand * (maxRange - minRange);
	                        
*/
                        if (y1 < lowBound)
                            y1 = lowBound;

                        if (y2 < lowBound)
                            y2 = lowBound;

                        if (y1 > upBound)
                            y1 = upBound;

                        if (y2 > upBound)
                            y2 = upBound;

                        ptrOffspring1->setValue(i, y1);
                        ptrOffspring2->setValue(i, y2);
        	}
	}
       return offSpring;
    }
};

#endif
