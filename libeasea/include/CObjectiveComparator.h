/*
 * CCrowdingComparator.h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef COBJECTIVECOMPARATOR_H
#define COBJECTIVECOMPARATOR_H

#include "CComparator.h"
/* This is class for objective comparation */
template<typename TIndividual>
class CObjectiveComparator : public CComparator<TIndividual> {

private:
    int nbObj_;
    bool incrisOrder_;

public:
    CObjectiveComparator(int nbObj) {
        nbObj_ = nbObj;
        incrisOrder_ = true;
    };

    CObjectiveComparator(int nbObj, bool decrisOrder) {
        nbObj_ = nbObj;
        incrisOrder_ = !decrisOrder;
    };

    int match(TIndividual * ind1, TIndividual * ind2){
        if (ind1 == nullptr)
            return 1;
        else if (ind2 == nullptr)
            return -1;

    double objetive1 = ind1->getObjective(nbObj_);
    double objetive2 = ind2->getObjective(nbObj_);
    if (incrisOrder_) {
        if (objetive1 < objetive2)
            return -1;
        else if (objetive1 > objetive2)
            return 1;
        else
            return 0;
    }else {
        if (objetive1 < objetive2)
            return 1;
        else if (objetive1 > objetive2)
            return -1;
        else
            return 0;
    }

};

};
#endif
