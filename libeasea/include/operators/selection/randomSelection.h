/***********************************************************************
| randomSelection.h                                                     |
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

#include <CLogger.h>
#include <list>
#include <shared/functions/dominance.h>
#include <shared/CConstant.h>


namespace easea
{
namespace operators
{
namespace selection
{

/*
 * \brief Random selection operator
 * \param[in] - random  - random generator
 * \param[in] - begin  -  pointer on the first individual in population
 * \param[in] - end    -  pointer on  the last individual in population
 * \retutn - Pointer to selected individual
 */

template <typename TRandom, typename TIter>
TIter randomSelection(TRandom &random, TIter begin, TIter end)
{
        if (begin == end) LOG_ERROR(errorCode::value, "Popultion is empty");
    
        std::uniform_int_distribution<size_t> uniform(0, std::distance(begin, end) - 1);
        TIter selected = begin;

        for (size_t count = uniform(random); count; --count)
                ++selected; 
        return selected;
}
}
}
}