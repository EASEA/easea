/*
 *  Created on: 11 octobre 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CBESTSELECTION_H
#define CBESTSELECTION_H


#include <memory>
#include "CPseudoRandom.h"
#include "CPermutation.h"
#include "CDominanceComparator.h"
#include "COperator.h"
#include "CLogger.h"

template<typename TIndividual, typename TPopulation>
class CBestSelection : public COperator<> {

private:
    int best_;
    int current_;
    CComparator<TIndividual> *dominance_;
public:
    CBestSelection() : COperator(){
	best_ = 0;
        dominance_ = new CDominanceComparator<TIndividual>();
        if (dominance_ == nullptr)
            LOG_ERROR(errorCode::memory, "memory for dominance_ wasn't allocated");

    };

    ~CBestSelection(){
        delete dominance_;
    };

    TIndividual *run(TPopulation *p){
	if (p->size() == 0)
	    return nullptr;

	if (p->size() == 4){
	    best_=0;
	    current_ = 0;
	}
	for (size_t i = current_; i< p->size(); i++){
	     if (dominance_->match(p->get(i),p->get(best_)) < 0)
		best_ = static_cast<int>(i);
	}
	current_ = p->size();
	TIndividual *res = p->get(best_);
	return res;
    };
};
#endif
