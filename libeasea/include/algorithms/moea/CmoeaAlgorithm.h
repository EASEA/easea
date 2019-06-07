/***********************************************************************
| CmoeaAlgorithm.h                                                      |
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

#include <algorithms/CAlgorithm.h>

namespace easea
{
namespace algorithms
{
template <typename TPopulation>
class CmoeaAlgorithm : public CAlgorithm<typename TPopulation::value_type::TO, typename TPopulation::value_type::TV>
{
public:
        typedef TPopulation TPop;
        typedef typename TPop::value_type TI;
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;
        typedef CAlgorithm<TO, TV> TA;
        typedef typename TA::TP TP;

        CmoeaAlgorithm(TP &problem);
        virtual ~CmoeaAlgorithm(void);
        const TPop &getPopulation(void) const;

protected:
        TPop m_population;
};

template <typename TPopulation>
CmoeaAlgorithm<TPopulation>::CmoeaAlgorithm(TP &problem) : TA(problem)
{
}

template <typename TPopulation>
CmoeaAlgorithm<TPopulation>::~CmoeaAlgorithm(void)
{
}

template <typename TPopulation>
const TPopulation &CmoeaAlgorithm<TPopulation>::getPopulation(void) const
{
        return m_population;
}
}
}
