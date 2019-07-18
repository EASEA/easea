/***********************************************************************
| C2x2Crossover.h                                                       |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2019-05                                                         |
|                                                                       |
 ***********************************************************************/
#pragma once


#include <vector>
#include <core/CmoIndividual.h>

namespace easea
{
namespace operators
{
namespace crossover
{
template <typename TObjective, typename TVariable>
class C2x2Crossover
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CmoIndividual<TO, TV> TI;

        C2x2Crossover(void);
        virtual ~C2x2Crossover(void);
        void operator ()(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2);

protected:
        virtual void runCrossover(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2) = 0;
};


template <typename TObjective, typename TVariable>
C2x2Crossover<TObjective, TVariable>::C2x2Crossover(void)
{
}

template <typename TObjective, typename TVariable>
C2x2Crossover<TObjective, TVariable>::~C2x2Crossover(void)
{
}

template <typename TObjective, typename TVariable>
void C2x2Crossover<TObjective, TVariable>::operator ()(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2)
{
        runCrossover(parent1, parent2, offspring1, offspring2);
}
}
}
}