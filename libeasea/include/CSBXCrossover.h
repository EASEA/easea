/*
 * CSBXCrossover.h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CSBXCROSSOVER_H
#define CSBXCROSSOVER_H

#include <math.h>

#include <COperator.h>
#include <CPseudoRandom.h>
#include <CWrapIndividual.h>

#include <sstream>


/*  In this class a Simulated Binary Crossover is realised
 *  It could be used for any kind types of variables,
 *  because class CWrapIndividual was implemented
 *  */
constexpr double ETA_ = 1.0e-14 ; //precision error tolerance
constexpr double EPS_ =  20.0;    //default distribution index


template<typename TIndividual>
class CSBXCrossover : public COperator<double,double> {

public:
    /* In Constructor:
     * First Parameter - Probability
     * Second Parameter - Distribution Index
     */
    CSBXCrossover() : COperator(0.9, EPS_){};

    CSBXCrossover(const double &probability, const double &distributionId) : COperator(probability, distributionId) {
        std::ostringstream ss;
        ss << "EASEA LOG [DEBUG]: Crossover: " <<  "probability  =  "<< get<double>(0) << std::endl
        << "EASEA LOG [DEBUG]: Crossover: "   << "distribution Index = " << get<double>(1) << std::endl;

        LOG_MSG(msgType::DEBUG, ss.str());
    };

    ~CSBXCrossover(){}

//    TIndividual ** run(const TIndividual *const * parents){
    TIndividual ** run( TIndividual **parents){

        return crossover(get<double>(0), get<double>(1), parents[0], parents[1]);
    }

private:
    /* Concept is taken from the author's implementation - K.Deb
      * The descrition of the algorithm was found in this paper:
      * Title: An Efficient Constraint Handling Method for Genetic Algorithms
      * Author: Kalyanmoy Deb
      * More info: Appendix A. Page 30.
      * URL: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.7291&rep=rep1&type=pdf
     */
    TIndividual ** crossover(const double &probability, const double &distributionId,  TIndividual * parent1,  TIndividual * parent2){

        TIndividual** offSpring = new TIndividual*[2];

        if (offSpring == nullptr)
            LOG_ERROR(errorCode::memory,"memory for offsprings individuals wasn't allocated!");

        offSpring[0] = new TIndividual(parent1);
        offSpring[1] = new TIndividual(parent2);

        double rand;
        double y1; // value of 1st child
        double y2; // value of 2nd child
        double yl; // lower limit for the variable
        double yu; // upper limit for the variable
        double c1; // 1st child
        double c2; // 2nd child
        double alpha;
        double beta; // spread value
        double betaq;// corresponding distribution index

        auto ptrParent1 = make_unique<CWrapIndividual>(parent1);
        auto ptrParent2 = make_unique<CWrapIndividual>(parent2);
        auto ptrOffspring1 = make_unique<CWrapIndividual>(offSpring[0]);
        auto ptrOffspring2 = make_unique<CWrapIndividual>(offSpring[1]);

        int numberOfVariables = ptrParent1->getNumberOfIndividualVariables();

        if (CPseudoRandom::randDouble() <= probability){
            for (auto i=0; i<numberOfVariables; i++){

                if (CPseudoRandom::randDouble() <= 0.5 ){// according to the paper, each variable in a solution has a 50% chance of changing its value. This should be removed when dealing with one-dimensional solutions.
                    if (fabs( ptrParent1->getValue(i) - ptrParent2->getValue(i)) > EPS_){ // if the value in parent1 is not the same of parent2
                        if ( ptrParent1->getValue(i) <  ptrParent2->getValue(i)){
                            y1 =  ptrParent1->getValue(i);
                            y2 =  ptrParent2->getValue(i);
                        } else {
                            y1 =  ptrParent2->getValue(i);
                            y2 =  ptrParent1->getValue(i);
                        }
                        yl = ptrParent1->getLowerBound(i);
                        yu = ptrParent1->getUpperBound(i);

                        rand = CPseudoRandom::randDouble();
                        // Calculation of the 1st child
                        // Here is one small difference with the paper:
                        // Calcul one beta for each child
                        beta = 1.0 + (2.0 * (y1 - yl)/(y2 - y1));
                        alpha = 2.0 - pow(beta,-(distributionId + 1.0));

                        if (rand <= (1.0/alpha))
                            betaq = pow ((rand * alpha),(1.0/(distributionId + 1.0)));
                        else
                            betaq = pow ((1.0/(2.0 - rand * alpha)),(1.0/(distributionId + 1.0)));

                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
                        beta = 1.0 + (2.0 * (yu - y2)/(y2 - y1));
                        alpha = 2.0 - pow(beta,-(distributionId + 1.0));
                        if (rand <= (1.0/alpha))
                            betaq = pow ((rand * alpha),(1.0/(distributionId + 1.0)));
                        else
                            betaq = pow ((1.0/(2.0 - rand * alpha)),(1.0/(distributionId + 1.0)));

                        c1 = 0.5*((y1 + y2) - betaq*(y2 - y1));
                        // Calculation of the 2nd child
                        // Changing of original algo in the paper:
                        // instead of it algo from Deb's source code is used.
                        // Different value of beta fpr 2nd child.
                        beta = 1.0 + (2.0*(yu - y2)/(y2 - y1));
                        alpha = 2.0 - pow(beta,-(distributionId + 1.0));

                        if (rand <= (1.0/alpha))
                            betaq = pow ((rand * alpha),(1.0/(distributionId + 1.0)));
                        else
                            betaq = pow ((1.0/(2.0 - rand*alpha)),(1.0/(distributionId+1.0)));

                        c2 = 0.5*((y1 + y2) + betaq*(y2 - y1));
                        // Checking that values of both children are in the correct limits [varLowLimit, varUpLimit].
                        // According to the paper, this should not be needed...
                        if (c1 < yl)
                            c1 = yl;

                        if (c2 < yl)
                            c2 = yl;

                        if (c1 > yu)
                            c1 = yu;

                        if (c2 > yu)
                            c2 = yu;

                        if (CPseudoRandom::randDouble() <= 0.5) {
                            ptrOffspring1->setValue(i,c2);
                            ptrOffspring2->setValue(i,c1);
                        } else {
                            ptrOffspring1->setValue(i,c1);
                            ptrOffspring2->setValue(i,c2);
                        }
                    } else {
                        ptrOffspring1->setValue(i, ptrParent1->getValue(i));
                        ptrOffspring2->setValue(i, ptrParent2->getValue(i));
                    }
                } else {
                    ptrOffspring1->setValue(i, ptrParent2->getValue(i));
                    ptrOffspring2->setValue(i, ptrParent1->getValue(i));
                }
            }
        }
       return offSpring;
    }
};

#endif
