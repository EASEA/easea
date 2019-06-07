/***********************************************************************
| nondominateSelection.h                                                |
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

#include <list>
#include <shared/functions/dominance.h>
#include <third_party/aixlog/aixlog.hpp>


namespace easea
{
namespace operators
{
namespace selection
{
/*
 * \brief Nondominate selection operator
 * \param[in] - population  - current population
 * \retutn - Pointer to selected individual
 */

template <typename TI, typename TIter, typename TDom, typename TNonCritic, typename TCritic>
TIter nondominateSelection(std::list<TI> &population, TIter begin, TIter end, TDom dominate, TNonCritic noncritical, TCritic critical)
{
        if (population.size() < std::distance(begin, end))
	{ 
		LOG(ERROR) << COLOR(red) << "Wrong popultion size " << std::endl << COLOR(none);
		exit(-1);
	}
        TIter selected = begin;
        while (!population.empty())
        {
                std::list<TI> nondominate = easea::shared::functions::getNondominated(population, dominate);
                
		if(nondominate.empty())
		{
			LOG(ERROR) << COLOR(red) << "No nontominated solutions " << std::endl << COLOR(none);
			exit(-1);
		}
                
		if (std::distance(selected, end) > nondominate.size())
                        selected = noncritical(nondominate, selected, end);
                else
                {
                        selected = critical(nondominate, selected, end);
                        if (selected != end)
			{
				LOG(ERROR) << COLOR(red) << "Critical selection error " << std::endl << COLOR(none);
				exit(-1);
			}
                        return selected;
                }
        }
        return selected;
}

}
}
}
