/*
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CBT2SELECTION_H
#define CBT2SELECTION_H


#include <memory>
#include "CPseudoRandom.h"
#include "CPermutation.h"
#include "CDominanceComparator.h"
#include "COperator.h"
#include "CLogger.h"

/*  Concept is following:
 *  1. Randomly select two individuals (solutions: ind1 and ind2);
 *  2. If the individuals are in the same non dominated front, than
 *  3. The individual with a higher crowding distance wins.
 *  4. Otherwise, the individual with the lowest rank is selected.
 */
template<typename TIndividual, typename TPopulation>
class CBT2Selection : public COperator<> {

private:
    int index_;
    size_t lastSize_;
    int *permutation_;
    CComparator<TIndividual> *dominance_;
public:
    CBT2Selection() : COperator(){

        index_ = 0;
	lastSize_ = 0;
        permutation_ = new int[1]; // must be initialized
        if (permutation_ == nullptr)
            LOG_ERROR(errorCode::memory, "memory for a_ wasn't allocated");
        dominance_ = new CDominanceComparator<TIndividual>();
        if (dominance_ == nullptr)
            LOG_ERROR(errorCode::memory, "memory for dominance_ wasn't allocated");

    };

    ~CBT2Selection(){
        delete dominance_;
        delete [] permutation_;
    };
    
    TIndividual *run(TPopulation *p){
	if (lastSize_ != p->size())
	    index_ = 0;
        if (index_ == 0) //Create the permutation for selecting two individuals randomly
        {
            auto permutation = std::make_unique<CPermutation>();
            delete [] permutation_;
            permutation_ = permutation->intPermutation(p->size());
	    lastSize_ = p->size();
        }
	
        /* Selection of two individuals */
        TIndividual *ind1 = p->get(permutation_[index_]);
        TIndividual *ind2 = p->get(permutation_[index_+1]);

        index_ = (index_ + 2) % p->size();

        auto res = int{dominance_->match(ind1,ind2)};
        if (res == -1)
            return ind1;                                                    // ind1 has lower rank
        else if (res == 1)
            return ind2;                                                    // ind2 has lower rank
        else if (ind1->crowdingDistance_ > ind2->crowdingDistance_) // ind1 and ind2 in the same dominated front
            return ind1;                                                    // ind1 has higher crowding distance
        else if (ind2->crowdingDistance_ > ind1->crowdingDistance_)
            return ind2;                                                    // ind2 has higher crowding distance
        else if (CPseudoRandom::randDouble() < 0.5)                         // ind1 = ind2 than we have to select randomly
            return ind1;
        else
            return ind2;
    };
};
#endif
