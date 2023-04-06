/***********************************************************************
| Ccde.h                                                                     |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2023-04                                                         |
|                                                                       |
 ***********************************************************************/
#pragma once

#include <vector>
#include <list>
#include <CLogger.h>
#include <algorithms/moea/CmoeaAlgorithm.h>

#include <operators/crossover/wrapper/CWrap3x1Crossover.h>
#include <operators/mutation/wrapper/CWrapMutation.h>
#include <operators/selection/totalSelection.h>

#include <shared/CRandom.h>
#include <shared/CConstant.h>
#include <shared/functions/breeding.h>
#include <config.h>


namespace easea
{
namespace algorithms
{
namespace cde
{
template <typename TIndividual, typename TRandom>
class Ccde : public CmoeaAlgorithm<std::vector<TIndividual>, TRandom>, public easea::operators::crossover::CWrap3x1Crossover<typename TIndividual::TO, typename TIndividual::TV>
{
public:
        typedef TIndividual TI  ;
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;

        typedef std::vector<TI> TPopulation;
        typedef CmoeaAlgorithm<TPopulation, TRandom> TBase;
        typedef typename TBase::TP TP;
        typedef typename easea::operators::crossover::CWrap3x1Crossover<TO, TV>::TC TC;

        Ccde(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover);
        ~Ccde(void);
        TI runBreeding(const TPopulation &parent, const size_t index);


protected:
        void makeOneGeneration(void);
private:
    std::uniform_int_distribution<size_t> m_distribution;
};

template <typename TIndividual, typename TRandom>
Ccde<TIndividual, TRandom>::Ccde(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover)
        : TBase(random, problem, initial)
        , easea::operators::crossover::CWrap3x1Crossover<TO, TV>(crossover)
	, m_distribution(0, initial.size()-1)
{
        typedef typename TPopulation::pointer TPtr;
        std::list<TPtr> population;
        for (size_t i = 0; i < TBase::m_population.size(); ++i)
	        population.push_back(&TBase::m_population[i]);
	
}

template <typename TIndividual, typename TRandom>
Ccde<TIndividual, TRandom>::~Ccde(void)
{
}

template <typename TIndividual, typename TRandom>
typename Ccde<TIndividual, TRandom>::TI Ccde<TIndividual, TRandom>::runBreeding(const TPopulation &parent, const size_t index)
{	

        if ( parent.size() < 3 ) LOG_ERROR( errorCode::value, "Population size < 3, DE mutation is impossible! " );
	size_t p1, p2, p3;
	do
	{
	    p1 = m_distribution( this->getRandom() );
	}while ( p1 == index );
	do
	{
	    p2 = m_distribution( this->getRandom() );
	}while ( p2 == index || p2 == p1 );
	do
	{
	    p3 = m_distribution( this->getRandom() );
	}while ( p2 == index || p2 == p1 || p3 == p2);
	assert( p1 != index && p2 != index && p3 != index );
	assert( p2 != p1 && p3 != p1 );
	assert ( p3 != p2 );
	TI child;
	this->getCrossover()( parent[index], parent[p1], parent[p2], parent[p3], child);
	TBase::getProblem()(child);
	return child;
}

template <typename TIndividual, typename TRandom>
void Ccde<TIndividual, TRandom>::makeOneGeneration(void)
{
        TPopulation parent = TBase::m_population;
	typedef typename TPopulation::pointer TPtr;
	std::vector<TPtr>updated{parent.size()};
#ifdef USE_OPENMP
    EASEA_PRAGMA_OMP_PARALLEL
#endif
	for (size_t i = 0; i < parent.size(); ++i)
	{
	    TIndividual child = runBreeding(parent, i);
	    if (child.m_objective[0] < parent[i].m_objective[0])
		parent[i] = child;
	    updated[i] = &parent[i];
	}
        easea::operators::selection::totalSelection(updated,TBase::m_population.begin(),TBase::m_population.end());
        std::sort(parent.begin(),parent.end(),[](TIndividual individual1, TIndividual individual2)->bool{return individual1.m_objective[0] < individual2.m_objective[0];});
}
}
}
}
