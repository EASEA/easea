/***********************************************************************
| Cgde.h                                                                |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2023-03                                                         |
|                                                                       |
 ***********************************************************************/
#pragma once

#include <vector>
#include <list>
#include <CLogger.h>
#include <algorithms/moea/CmoeaAlgorithm.h>

#include <operators/crossover/wrapper/CWrap3x1Crossover.h>
#include <operators/selection/nondominateSelection.h>

#include <shared/CRandom.h>
#include <shared/CConstant.h>
#include <shared/functions/breeding.h>
#include <shared/functions/crowdingDistance.h>
#include <shared/functions/dominance.h>
#include <config.h>


namespace easea
{
namespace algorithms
{
namespace gde
{
template <typename TIndividual, typename TRandom>
class Cgde : public CmoeaAlgorithm<std::vector<TIndividual>, TRandom>, public easea::operators::crossover::CWrap3x1Crossover<typename TIndividual::TO, typename TIndividual::TV>
{
public:
        typedef TIndividual TI  ;
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;

        typedef std::vector<TI> TPopulation;
        typedef CmoeaAlgorithm<TPopulation, TRandom> TBase;
        typedef typename TBase::TP TP;
        typedef typename easea::operators::crossover::CWrap3x1Crossover<TO, TV>::TC TC;

        Cgde(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover);
        ~Cgde(void);
        TI runBreeding(const TPopulation &parent, const size_t index);
        static bool isDominated(const TIndividual &individual1, const TIndividual &individual2);


protected:
        static bool Dominate(const TI *individual1, const TI *individual2);
        void makeOneGeneration(void) override;
  	void initialize() override;
        template <typename TPtr, typename TIter> static TIter selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end);
        template <typename TPtr, typename TIter> static TIter selectCrit(const std::list<TPtr> &front, TIter begin, TIter end);
private:
    std::uniform_int_distribution<size_t> m_distribution;

};

template <typename TIndividual, typename TRandom>
Cgde<TIndividual, TRandom>::Cgde(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover)
        : TBase(random, problem, initial)
        , easea::operators::crossover::CWrap3x1Crossover<TO, TV>(crossover)
	, m_distribution(0, initial.size()-1)
{
}

template <typename TIndividual, typename TRandom>
void Cgde<TIndividual, TRandom>::initialize()
{
	TBase::initialize();
 	typedef typename TPopulation::pointer TPtr;
        std::list<TPtr> population;
        for (size_t i = 0; i < TBase::m_population.size(); ++i)
	        population.push_back(&TBase::m_population[i]);
	
        while (!population.empty())
        {
                std::list<TPtr> nondominate = easea::shared::functions::getNondominated(population, &Dominate);
                std::vector<TPtr> _nondominate(nondominate.begin(), nondominate.end());
                easea::shared::functions::setCrowdingDistance<TO>(_nondominate.begin(), _nondominate.end());
        }
}

template <typename TIndividual, typename TRandom>
Cgde<TIndividual, TRandom>::~Cgde(void)
{
}

template <typename TIndividual, typename TRandom>
typename Cgde<TIndividual, TRandom>::TI Cgde<TIndividual, TRandom>::runBreeding(const TPopulation &parent, const size_t index)
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
bool Cgde<TIndividual, TRandom>::isDominated(const TI &individual1, const TI &individual2)
{
        return easea::shared::functions::isDominated(individual1.m_objective, individual2.m_objective);
}

template <typename TIndividual, typename TRandom>
bool Cgde<TIndividual, TRandom>::Dominate(const TI *individual1, const TI *individual2)
{
        return isDominated(*individual1, *individual2);
}

template <typename TIndividual, typename TRandom>
void Cgde<TIndividual, TRandom>::makeOneGeneration(void)
{
        TPopulation parent = TBase::m_population;
        std::list<TIndividual> offspring;
	std::vector<TIndividual> children;
        typedef typename TPopulation::pointer TPtr;
	std::list<TPtr> unionPop;

#ifdef USE_OPENMP
    EASEA_PRAGMA_OMP_PARALLEL
#endif
	for ( size_t i = 0; i < parent.size(); ++i )
	{
	    TIndividual &ind = parent[i];
	    TIndividual child = runBreeding( parent, i );
#ifdef USE_OPENMP
    EASEA_PRAGMA_OMP_CRITICAL
#endif
	    if ( isDominated(ind, child) )
		unionPop.push_back(&ind);
	    else if ( isDominated(child, ind) ){
                offspring.push_back( child );
		unionPop.push_back( &offspring.back() );
	    }else{
		unionPop.push_back( &ind );
		offspring.push_back( child );
		unionPop.push_back( &offspring.back() );
	    }

        }
	
	typedef typename TPopulation::iterator TIter;
        easea::operators::selection::nondominateSelection(unionPop, TBase::m_population.begin(), TBase::m_population.end(), &Dominate, &selectNoncrit<TPtr, TIter>, &selectCrit<TPtr, TIter>);

}


template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Cgde<TIndividual, TRandom>::selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end)
{
        TIter dest = begin;
        for ( auto i = front.begin(); i != front.end(); ++i, ++dest )
                *dest = **i;
        return dest;
}

template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Cgde<TIndividual, TRandom>::selectCrit(const std::list<TPtr> &front, TIter begin, TIter end)
{
        std::vector<TPtr> iFront(front.begin(), front.end());
        easea::shared::functions::setCrowdingDistance<TO>(iFront.begin(), iFront.end());
        std::partial_sort(iFront.begin(), iFront.begin() + std::distance(begin, end), iFront.end()
                , [](TPtr individual1, TPtr individual2)->bool{return individual1->m_crowdingDistance > individual2->m_crowdingDistance;});
        if (iFront.size() < std::distance(begin, end)) LOG_ERROR(errorCode::value, "Select critical : Error of front size!");
        TIter dest = begin;
        for (size_t i = 0; dest != end; ++i, ++dest)
                *dest = *iFront[i];
        return dest;
}

}
}
}
