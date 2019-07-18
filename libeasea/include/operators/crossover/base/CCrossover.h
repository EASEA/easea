/***********************************************************************
| CCrossover.h                                                          |
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
class CCrossover
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CmoIndividual<TO, TV> TI;

        CCrossover(void);
        virtual ~CCrossover(void);
        void operator ()(std::vector<const TI *> &parent, std::vector<TI *> &offspring);

protected:
        virtual void runCrossover(std::vector<const TI *> &parent, std::vector<TI *> &offspring) = 0;
};

template <typename TObjective, typename TVariable>
CCrossover<TObjective, TVariable>::CCrossover(void)
{
}

template <typename TObjective, typename TVariable>
CCrossover<TObjective, TVariable>::~CCrossover(void)
{
}

template <typename TObjective, typename TVariable>
void CCrossover<TObjective, TVariable>::operator ()(std::vector<const TI *> &parent, std::vector<TI *> &offspring)
{
        runCrossover(parent, offspring);
}
}
}
}
