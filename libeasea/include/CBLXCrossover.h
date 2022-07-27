/*
 * CBLXCrossover.h 
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CBLXCROSSOVER_H
#define CBLXCROSSOVER_H

#include <math.h>

#include <COperator.h>
#include <CPseudoRandom.h>
#include <CWrapIndividual.h>

#include <sstream>


/*  In this class a Simulated Binary Crossover is realised
 *  It could be used for any kind types of variables,
 *  because class CWrapIndividual was implemented
 *  */
constexpr double DEFAULT_ALPHA_ = 0.5;  


template<typename TIndividual>
class CBLXCrossover : public COperator<double,double> {

public:
    /* In Constructor:
     * First Parameter - Probability
     * Second Parameter - Distribution Index
     */
    CBLXCrossover() : COperator(0.9, DEFAULT_ALPHA_){};

    CBLXCrossover(const double &probability, const double &alpha) : COperator(probability, alpha) {
        std::ostringstream ss;

	if (probability < 0){
		ss << "Wrong value of crossover probability: it's < 0" << std::endl;
		LOG_ERROR(errorCode::value, ss.str());
        }
	if (alpha < 0){
		ss << "Wrong value of crossover alpha: it's < 0" << std::endl;
		LOG_ERROR(errorCode::value, ss.str());
	
	}
	ss << "EASEA LOG [DEBUG]: Crossover: " <<  "probability  =  "<< get<double>(0) << std::endl
        << "EASEA LOG [DEBUG]: Crossover: "   << "alpha = " << get<double>(1) << std::endl;
	
        LOG_MSG(msgType::DEBUG, ss.str());
    };

    ~CBLXCrossover(){}

    TIndividual ** run(const TIndividual *const * parents){
        return crossover(get<double>(0), get<double>(1), parents[0], parents[1]);
    }

private:
    
    TIndividual ** crossover(const double &probability, const double &alpha, const TIndividual * parent1, const TIndividual * parent2){

        TIndividual** offSpring = new TIndividual*[2];

        if (offSpring == nullptr)
            LOG_ERROR(errorCode::memory,"memory for offsprings individuals wasn't allocated!");

        offSpring[0] = new TIndividual(parent1);
        offSpring[1] = new TIndividual(parent2);

        double rand;
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
