/***********************************************************************
| CWrapCrossover.h                                                      |
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

#include <operators/crossover/base/CCrossover.h>

namespace easea
{
namespace operators
{
namespace crossover
{
template <typename TObjective, typename TVariable>
class CWrapCrossover
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CCrossover<TO, TV> TC;

        CWrapCrossover(TC &crossover);
        ~CWrapCrossover(void);
        TC &getCrossover(void) const;

private:
        TC &m_crossover;
};

template <typename TObjective, typename TVariable>
CWrapCrossover<TObjective, TVariable>::CWrapCrossover(TC &crossover) : m_crossover(crossover)
{
}

template <typename TObjective, typename TVariable>
CWrapCrossover<TObjective, TVariable>::~CWrapCrossover(void)
{
}

template <typename TObjective, typename TVariable>
typename CWrapCrossover<TObjective, TVariable>::TC &CWrapCrossover<TObjective, TVariable>::getCrossover(void) const
{
        return m_crossover;
}
}
}
}
