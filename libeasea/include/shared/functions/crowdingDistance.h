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

#include <limits>
#include <algorithm>
#include <iterator>
#include <third_party/aixlog/aixlog.hpp>

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
        if (extValue < 0){ LOG(ERROR) << COLOR(red) << "Wrong crowding distance value - " <<  extValue << COLOR(none) << std::endl; exit(-1);};
        if (begin == end){ LOG(ERROR) << COLOR(red) << "Wrong size of population" << COLOR(none) << std::endl; exit(-1); }

        for (TIter i = begin; i != end; ++i)
                (**i).m_crowdingDistance = 0;
        const size_t nObjectives = (**begin).m_objective.size();
        for (size_t objective = 0; objective < nObjectives; ++objective)
        {
                typedef typename std::iterator_traits<TIter>::value_type _TPointer;
                std::sort(begin, end, [objective](_TPointer individual1, _TPointer individual2)->bool{return individual1->m_objective[objective] < individual2->m_objective[objective];});
                TIter last = end - 1;
                const TObjective range = (**last).m_objective[objective] - (**begin).m_objective[objective];
                if (range > 0)
                {
                        for (TIter i = begin + 1; i != last; ++i)
                        {
               			 if ((**(i - 1)).m_objective[objective] > (**i).m_objective[objective] && (**i).m_objective[objective] > (**(i + 1)).m_objective[objective])
				 {
					 LOG(ERROR) << COLOR(red) << "Wrong individual place" << COLOR(none) << std::endl; 
					 exit(-1);
				 }
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
void crowdingDistance(TIter begin, TIter end)
{
        crowdingDistance(begin, end, std::numeric_limits<TObjective>::max());
}
}
}
}

