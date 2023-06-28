/***********************************************************************
| CWrap2x2Crossover.h                                                   |
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

#include <functional>

#include <operators/crossover/base/C2x2Crossover.h>

namespace easea
{
namespace operators
{
namespace crossover
{
template <typename TObjective, typename TVariable>
class CWrap2x2Crossover
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef C2x2Crossover<TO, TV> TC;

        CWrap2x2Crossover(TC &crossover);
        ~CWrap2x2Crossover(void);
        TC &getCrossover(void) const;

private:
	std::reference_wrapper<TC> m_crossover;
};


template <typename TObjective, typename TVariable>
CWrap2x2Crossover<TObjective, TVariable>::CWrap2x2Crossover(TC &crossover) : m_crossover(crossover)
{
}

template <typename TObjective, typename TVariable>
CWrap2x2Crossover<TObjective, TVariable>::~CWrap2x2Crossover(void)
{
}

template <typename TObjective, typename TVariable>
typename CWrap2x2Crossover<TObjective, TVariable>::TC &CWrap2x2Crossover<TObjective, TVariable>::getCrossover(void) const
{
        return m_crossover;
}

// reduce compilation time and check for errors while compiling lib
extern template class CWrap2x2Crossover<float, std::vector<float>>;
extern template class CWrap2x2Crossover<double, std::vector<double>>;

}
}
}
