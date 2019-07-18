/*
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CRANDOMSELECTION_H
#define CRANDOMSELECTION_H


#include <memory>
#include "CPseudoRandom.h"
#include "CLogger.h"
#include "COperator.h"

/* Random selection */
template<typename TIndividual, typename TPopulation>
class CRandomSelection : public COperator<> {

public:
    CRandomSelection() : COperator(){};

    ~CRandomSelection(){};

    void *run(void *object){

        TPopulation * population = (TPopulation *) object;

        int id1 = CPseudoRandom::randInt(0,population->size()-1);
        int id2 = CPseudoRandom::randInt(0,population->size()-1);

        while ((id1 == id2)  && (population->size()>1) )
            id2 = CPseudoRandom::randInt(0,population->size()-1);


        TIndividual **individuals = new TIndividual*[2];
        if (individuals == nullptr)
            LOG_ERROR(errorCode::memory, "memory for parents wasn't allocated!");
        individuals[0] = population->get(id1);
        individuals[1] = population->get(id2);

        return individuals;
    };
};
#endif
