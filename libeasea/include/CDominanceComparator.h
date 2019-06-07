/*
 * CDominanceComparator.h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CDOMINANCECOMPARATOR_H
#define CDOMINANCECOMPARATOR_H

#include <CComparator.h>
#include <CConstraintComparator.h>

/* This is class is used for comparing two individuals (solutions) by dominance.
 * It gets two objects of individauls as the inputs parameters.
 * The result of comaration is -1, if the first individual dominates to second
 * individual. The result = 1, if the second individual dominates to first one.
 * And the result = 0 in case of two individuals are equals.
 */

template<typename TIndividual>
class CDominanceComparator : public CComparator<TIndividual> {

private:
    CComparator<TIndividual> * c_;

public:
    CDominanceComparator(){
        c_ = new CConstraintComparator<TIndividual>();
    };

    ~CDominanceComparator(){
         delete c_;
    };

    int match(TIndividual *ind1, TIndividual *ind2){
        if (ind1==nullptr)
            return 1;
        else if (ind2 == nullptr)
            return -1;

        auto value1 = double {0.0};
	auto value2 = double {0.0};
	auto flag1 = int{0};
	auto flag2 = int{0};
	int nbObj = ind1->getNumberOfObjectives();
	for(int i = 0; i < nbObj; ++i){
	    value1 = ind1->objective_[i];
	    value2 = ind2->objective_[i];
	    if (value1 < value2)
		flag1 = 1;
	    else{
		if (value1 > value2)
		    flag2 = 1;
	    }
		
	}
        if (flag1==1 && flag2==0)
        {
    	    return -1;
        }
        else
        {
            if (flag1==0 && flag2==1)
            {
        	return 1;
    	    }
            else
            {
               return 0;
            }
        }

    };
};

#endif
