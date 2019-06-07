/***********************************************************************
| breeding.h                                                    	|
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

#include <vector>
#include <iterator>
#include <core/CmoIndividual.h>
#include <operators/crossover/base/CCrossover.h>
#include <operators/selection/tournamentSelection.h>

namespace easea
{
namespace shared
{
namespace functions
{
template <typename TIter, typename TRandom, typename TComparator>
std::vector<typename std::iterator_traits<TIter>::pointer> runSelection(const size_t offspringSize, TIter begin, TIter end, TRandom &random, TComparator comparator)
{
        typedef typename std::iterator_traits<TIter>::pointer TPtr;

        std::vector<TPtr> parent(std::distance(begin, end));
        {
                TIter src = begin;
                for (size_t i = 0; i < parent.size(); ++i, ++src)
                        parent[i] = &*src;
        }

        std::vector<TPtr> selected(offspringSize);
        easea::operators::selection::tournamentSelection(random, parent.begin(), parent.end(), selected.begin(), selected.end(), comparator);

        return selected;
}

template <typename TI, typename TPtr>
void runCrossover(std::vector<TI> &offspring, const std::vector<TPtr> &parent, easea::operators::crossover::CCrossover<typename TI::TO, typename TI::TV> &crossover)
{
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;
        typedef typename easea::operators::crossover::CCrossover<TO, TV>::TI TIndividual;
        
	std::vector<const TIndividual *> i_parent(parent.size());
        
	for (size_t i = 0; i < parent.size(); ++i)
                i_parent[i] = parent[i];
        
	std::vector<TIndividual *> i_offspring(offspring.size());
        
	for (size_t i = 0; i < offspring.size(); ++i)
                i_offspring[i] = &offspring[i];
        
	crossover(i_parent,i_offspring);
}

template <typename TIter, typename TRandom, typename TComparator>
std::vector<typename std::iterator_traits<TIter>::value_type> runBreeding(const size_t offspringSize, TIter begin, TIter end, TRandom &random, TComparator comparator, easea::operators::crossover::CCrossover<typename std::iterator_traits<TIter>::value_type::TO, typename std::iterator_traits<TIter>::value_type::TV> &crossover)
{
        typedef typename std::iterator_traits<TIter>::value_type TI;

        const auto selected = runSelection(offspringSize, begin, end, random, comparator);

        std::vector<TI> offspring(selected.size());

        runCrossover(offspring, selected, crossover);

        return offspring;
}
}
}
}
