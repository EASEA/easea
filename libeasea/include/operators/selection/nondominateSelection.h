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
#include <CLogger.h>
#include <shared/CConstant.h>
#include <shared/functions/dominance.h>



namespace easea
{
namespace operators
{
namespace selection
{
/*
 * \brief Nondominate selection operator
 * \param[in] - donorPop  - current population form here individuals will be selected to new population
 * \retutn - Pointer to the first individual of recipient population (new population)
 */

template <typename TI, typename TIter, typename TDom, typename TNonCritic, typename TCritic>
TIter nondominateSelection(std::list<TI> &donorPop, TIter recipPopBegin, TIter recipPopEnd, TDom dominate, TNonCritic noncritical, TCritic critical)
{
        if (donorPop.size() < std::distance(recipPopBegin, recipPopEnd)) 	LOG_ERROR(errorCode::value, "Wrong popultion size");

        TIter selected = recipPopBegin;
        while (!donorPop.empty())
        {
                std::list<TI> nondominate = easea::shared::functions::getNondominated(donorPop, dominate);
                
		if(nondominate.empty()) 	LOG_ERROR(errorCode::value, "No nontominated solutions");

		if (std::distance(selected, recipPopEnd) > nondominate.size())
                        selected = noncritical(nondominate, selected, recipPopEnd);	/* Selecting nondominated solutions by crowding distance */
                else
                {
                        selected = critical(nondominate, selected, recipPopEnd);
                        if (selected != recipPopEnd)		LOG_ERROR(errorCode::value, "Critical selection error");
                        return selected;
                }
        }
        return selected;
}

}
}
}
