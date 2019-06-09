/***********************************************************************
| totalSelection.h                                                      |
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
#include <shared/CConstant.h>
#include <shared/functions/dominance.h>



namespace easea
{
namespace operators
{
namespace selection
{
/*
 * \brief Total selection operator - it copies donor populatio to recipient population
 * \param[in] - donorPop  - donor population
 * \retutn - Pointer to first individual in recipient population
 */


template <typename TI, typename TIter>
TIter totalSelection(std::vector<TI> &donorPop, TIter recipPopBegin, TIter recipPopEnd)
{
        if (donorPop.size() > std::distance(recipPopBegin, recipPopEnd)) 	LOG_FATAL("Donor populaton size must be smaller or the same as the recipient population!");

        TIter selected = recipPopBegin;
		if (std::distance(selected, recipPopEnd) > donorPop.size())
		{
			for (size_t i = 0; i < donorPop.size(); ++i, ++selected)
                    	    *selected = *donorPop[i];

		}
        return selected;
}

}
}
}
