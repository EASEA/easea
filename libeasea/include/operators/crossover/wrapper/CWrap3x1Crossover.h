/***********************************************************************
| CWrap3x1Crossover.h                                                   |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2023-03                                                         |
|                                                                       |
 ***********************************************************************/
#pragma once

#include <operators/crossover/base/C3x1Crossover.h>

namespace easea
{
namespace operators
{
namespace crossover
{
template <typename TObjective, typename TVariable>
class CWrap3x1Crossover
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef C3x1Crossover<TO, TV> TC;

        CWrap3x1Crossover(TC &crossover);
        ~CWrap3x1Crossover(void);
        TC &getCrossover(void) const;

private:
        TC &m_crossover;
};


template <typename TObjective, typename TVariable>
CWrap3x1Crossover<TObjective, TVariable>::CWrap3x1Crossover(TC &crossover) : m_crossover(crossover)
{
}

template <typename TObjective, typename TVariable>
CWrap3x1Crossover<TObjective, TVariable>::~CWrap3x1Crossover(void)
{
}

template <typename TObjective, typename TVariable>
typename CWrap3x1Crossover<TObjective, TVariable>::TC &CWrap3x1Crossover<TObjective, TVariable>::getCrossover(void) const
{
        return m_crossover;
}
}
}
}