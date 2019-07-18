/***********************************************************************
| crowdingDistance.h 							|
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA		|
| (EAsy Specification of Evolutionary Algorithms) 			|
| https://github.com/EASEA/                                 		|
|    									|	
| Copyright (c)      							|
| ICUBE Strasbourg		                           		|
| Date: 2019-05                                                         |
|                                                                       |
 ***********************************************************************/

#pragma once

#include <CLogger.h>
#include <limits>
#include <algorithm>
#include <iterator>
#include <shared/CConstant.h>

#undef max

namespace easea
{
namespace shared
{
namespace functions
{
/*
 * \brief crowding distance assigement function
 *
 * \param[in] begin    - pointer to first element of population
 * \param[in] end      - pointer to last element of population
 * \param[in] extValue - extreme distance value
 */

template <typename TIter, typename TObjective>
void crowdingDistance(TIter begin, TIter end, const TObjective extValue)
{
        if (extValue < 0) LOG_ERROR(errorCode::value, "Wrong extrem value");
        if (begin == end) LOG_ERROR(errorCode::value, "Wrong size of population");

        for (TIter i = begin; i != end; ++i)
                (**i).m_crowdingDistance = 0;
        const size_t nObjectives = (**begin).m_objective.size();
        for (size_t objective = 0; objective < nObjectives; ++objective)
        {
                typedef typename std::iterator_traits<TIter>::value_type TPtr;
                std::sort(begin, end, [objective](TPtr individual1, TPtr individual2)->bool{return individual1->m_objective[objective] < individual2->m_objective[objective];});
                TIter last = end - 1;
                const TObjective range = (**last).m_objective[objective] - (**begin).m_objective[objective];
                if (range > 0)
                {
                        for (TIter i = begin + 1; i != last; ++i)
                        {
               			 if ((**(i - 1)).m_objective[objective] > (**i).m_objective[objective] && (**i).m_objective[objective] > (**(i + 1)).m_objective[objective])
					 LOG_ERROR(errorCode::value, "Wrong location by crowding distance");
          			 (**i).m_crowdingDistance += ((**(i + 1)).m_objective[objective] - (**(i - 1)).m_objective[objective]) / range;
                        }
                        (**begin).m_crowdingDistance = extValue;
                        (**last).m_crowdingDistance = extValue;
                }
                else
                {
                        for (TIter i = begin; i != end; ++i)
                                (**i).m_crowdingDistance = extValue;
                        return;
                }
        }
}

template <typename TObjective, typename TIter>
void setCrowdingDistance(TIter begin, TIter end)
{
        crowdingDistance(begin, end, std::numeric_limits<TObjective>::max());
}


}
}
}

