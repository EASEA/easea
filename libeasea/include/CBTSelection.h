/*
 *  Created on: 11 octobre 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CBTSELECTION_H
#define CBTSELECTION_H


#include <memory>
#include "CPseudoRandom.h"
#include "CPermutation.h"
#include "CDominanceComparator.h"
#include "CCrowdingDistanceComparator.h"
#include "COperator.h"
#include "CLogger.h"

template<typename TIndividual, typename TPopulation>
class CBTSelection : public COperator<> {

private:
    CComparator<TIndividual> *dominance_;
public:
    CBTSelection() : COperator(){

        dominance_ = new CDominanceComparator<TIndividual>();
        if (dominance_ == nullptr)
            LOG_ERROR(errorCode::memory, "memory for dominance_ wasn't allocated");

    };

    ~CBTSelection(){
        delete dominance_;
    };

    TIndividual *run(TPopulation *p){
	int index1 = CPseudoRandom::randInt(0,p->size()-1);
	int index2 = CPseudoRandom::randInt(0,p->size()-1);
    
	if (p->size() >= 2){
	    while (index1 == index2)
		index2 = CPseudoRandom::randInt(0,p->size()-1);
	}

        /* Selection of two individuals */
	TIndividual *ind1 = p->get(index1);

        TIndividual *ind2 = p->get(index2);

        auto res = int{dominance_->match(ind1,ind2)};

        if (res == -1)
            return ind1;                                                    // ind1 has lower rank
        else if (res == 1)
            return ind2;                                                    // ind2 has lower rank
        else if (CPseudoRandom::randDouble() < 0.5)                         // ind1 = ind2 than we have to select randomly
            return ind1;
        else
            return ind2;
    };
};
#endif
