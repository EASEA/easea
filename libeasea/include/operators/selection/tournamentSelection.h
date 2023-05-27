/***********************************************************************
| tournamentSelection.h                                                 |
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

#include <random>
#include <algorithm>
#include <vector>
#include <iterator>

namespace easea
{
namespace operators
{
namespace selection
{
/*
 * \brief Tournament selection operator
 * \param[in] - random  - random generator
 * \param[in] - begin  -  pointer on the first individual in parent population
 * \param[in] - end    -  pointer on  the last individual in parent population
 * \param[in] - fComparator - comparation function
 * \param[in] - nbParents   - number of parents
 * \retutn -    Pointer to new offspring population
 */


template <typename TRandom, typename TptrParent, typename TptrOffspring, typename TComp>
TptrOffspring tournamentSelection(TRandom &random, TptrParent pBegin, TptrParent pEnd, TptrOffspring oBegin, TptrOffspring oEnd, TComp fComparator , const size_t nbParents = 2)
{
        typedef typename std::iterator_traits<TptrParent>::value_type TPtr;
        TptrOffspring selected = oBegin;
        for (TptrParent src = pEnd; selected != oEnd; ++selected)
        {
                std::vector<TPtr> tournament(nbParents);
                for (size_t i = 0; i < tournament.size(); ++i)
                {
                        if (src == pEnd)
                        {
                                std::shuffle(pBegin, pEnd, random);
                                src = pBegin;
                        }
                        tournament[i] = *src;
                        ++src;
                }
                *selected = fComparator(tournament);
        }
        return selected;
}
}
}
}
