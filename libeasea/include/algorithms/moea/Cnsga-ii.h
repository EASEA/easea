/***********************************************************************
| Cnsga-ii.h                                                            |
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
#include <list>

#include <algorithms/moea/CmoeaAlgorithm.h>

#include <operators/crossover/wrapper/CWrapCrossover.h>
#include <operators/mutation/wrapper/CWrapMutation.h>
#include <operators/selection/nondominateSelection.h>

#include <shared/CRandom.h>
#include <shared/functions/breeding.h>
#include <shared/functions/crowdingDistance.h>
#include <shared/functions/dominance.h>



namespace easea
{
namespace algorithms
{
namespace nsga_ii
{
template <typename TIndividual, typename TRandom>
class Cnsga_ii : public CmoeaAlgorithm<std::vector< TIndividual >>, public easea::shared::CRandom<TRandom>, public easea::operators::crossover::CWrapCrossover<typename TIndividual::TO, typename TIndividual::TV>, public easea::operators::mutation::CWrapMutation<typename TIndividual::TO, typename TIndividual::TV>
{
public:
        typedef TIndividual TI  ;
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;
        typedef TRandom TR;

        typedef std::vector<TI> TPopulation;
        typedef CmoeaAlgorithm<TPopulation> TBase;
        typedef typename TBase::TP TP;
        typedef typename easea::operators::crossover::CWrapCrossover<TO, TV>::TC TC;
        typedef typename easea::operators::mutation::CWrapMutation<TO, TV>::TM TM;

        Cnsga_ii(TR random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation);
        ~Cnsga_ii(void);
        TPopulation runBreeding(const TPopulation &parent);
        static bool isDominated(const TIndividual &individual1, const TIndividual &individual2);


protected:
        static bool Dominate(const TI *individual1, const TI *individual2);
        void makeOneGeneration(void);
        template <typename TPtr, typename TIter> static TIter selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end);
        template <typename TPtr, typename TIter> static TIter selectCrit(const std::list<TPtr> &front, TIter begin, TIter end);
        static const TI *comparer(const std::vector<const TI *> &competition);
};

template <typename TIndividual, typename TRandom>
Cnsga_ii<TIndividual, TRandom>::Cnsga_ii(TR random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation)
        : TBase(problem), easea::shared::CRandom<TR>(random)
        , easea::operators::crossover::CWrapCrossover<TO, TV>(crossover), easea::operators::mutation::CWrapMutation<TO, TV>(mutation)
{
        TBase::m_population.resize(initial.size());
        for (size_t i = 0; i < initial.size(); ++i)
        {
                TIndividual &individual = TBase::m_population[i];
                individual.m_variable = initial[i];
                TBase::getProblem()(individual);
        }

        typedef typename TPopulation::pointer TPtr;
        std::list<TPtr> population;
        for (size_t i = 0; i < TBase::m_population.size(); ++i)
                population.push_back(&TBase::m_population[i]);
        while (!population.empty())
        {
                std::list<TPtr> nondominate = easea::shared::functions::getNondominated(population, &Dominate);
                std::vector<TPtr> _nondominate(nondominate.begin(), nondominate.end());
                easea::shared::functions::crowdingDistance<TO>(_nondominate.begin(), _nondominate.end());
        }
}

template <typename TIndividual, typename TRandom>
Cnsga_ii<TIndividual, TRandom>::~Cnsga_ii(void)
{
}

template <typename TIndividual, typename TRandom>
typename Cnsga_ii<TIndividual, TRandom>::TPopulation Cnsga_ii<TIndividual, TRandom>::runBreeding(const TPopulation &parent)
{
        TPopulation offspring = easea::shared::functions::runBreeding(parent.size(), parent.begin(), parent.end(), this->getRandom(), &comparer, this->getCrossover());
        for (size_t i = 0; i < offspring.size(); ++i)
        {
                TI &child = offspring[i];
                this->getMutation()(child);
                TBase::getProblem()(child);
        }
        return offspring;
}

template <typename TIndividual, typename TRandom>
bool Cnsga_ii<TIndividual, TRandom>::isDominated(const TI &individual1, const TI &individual2)
{
        return easea::shared::functions::isDominated(individual1.m_objective, individual2.m_objective);
}

template <typename TIndividual, typename TRandom>
bool Cnsga_ii<TIndividual, TRandom>::Dominate(const TI *individual1, const TI *individual2)
{
        return isDominated(*individual1, *individual2);
}

template <typename TIndividual, typename TRandom>
void Cnsga_ii<TIndividual, TRandom>::makeOneGeneration(void)
{
        TPopulation parent = TBase::m_population;
        TPopulation offspring = runBreeding(parent);
        typedef typename TPopulation::pointer TPtr;
        
	std::list<TPtr> unionPop;
        
	for (size_t i = 0; i < parent.size(); ++i)
                unionPop.push_back(&parent[i]);
        for (size_t i = 0; i < offspring.size(); ++i)
                unionPop.push_back(&offspring[i]);
        
	typedef typename TPopulation::iterator TIter;
        easea::operators::selection::nondominateSelection(unionPop, TBase::m_population.begin(), TBase::m_population.end(), &Dominate, &selectNoncrit<TPtr, TIter>, &selectCrit<TPtr, TIter>);
}


template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Cnsga_ii<TIndividual, TRandom>::selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end)
{
        std::vector<TPtr> _front(front.begin(), front.end());
        easea::shared::functions::crowdingDistance<TO>(_front.begin(), _front.end());
        if (_front.size() > std::distance(begin, end))
        {
                LOG(ERROR) << COLOR(red) << "Select Noncritical : Error of front size = " << _front.size() << std::endl << COLOR(none);
                exit(-1);
        }
        TIter dest = begin;
        for (size_t i = 0; i < _front.size(); ++i, ++dest)
                *dest = *_front[i];
        return dest;
}

template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Cnsga_ii<TIndividual, TRandom>::selectCrit(const std::list<TPtr> &front, TIter begin, TIter end)
{
        std::vector<TPtr> _front(front.begin(), front.end());
        easea::shared::functions::crowdingDistance<TO>(_front.begin(), _front.end());
        std::partial_sort(_front.begin(), _front.begin() + std::distance(begin, end), _front.end()
                , [](TPtr individual1, TPtr individual2)->bool{return individual1->m_crowdingDistance > individual2->m_crowdingDistance;});
        if (_front.size() < std::distance(begin, end))
        {
                LOG(ERROR) << COLOR(red) << "Select critical : Error of front size = " << _front.size() << std::endl << COLOR(none);
                exit(-1);
        }
        TIter dest = begin;
        for (size_t i = 0; dest != end; ++i, ++dest)
                *dest = *_front[i];
        return dest;
}

template <typename TIndividual, typename TRandom>
const typename Cnsga_ii<TIndividual, TRandom>::TI *Cnsga_ii<TIndividual, TRandom>::comparer(const std::vector<const TI *> &competition)
{
        if (isDominated(*competition[0], *competition[1]))
                return competition[0];
        else if (isDominated(*competition[1], *competition[0]))
                return competition[1];
        if (competition[0]->m_crowdingDistance < 0){ LOG(ERROR) << COLOR(red) << "Crowding distance < 0 " << std::endl << COLOR(none); exit(-1); };
        if (competition[1]->m_crowdingDistance < 0){ LOG(ERROR) << COLOR(red) << "Crowding distance < 0 " << std::endl << COLOR(none); exit(-1); };

        return competition[0]->m_crowdingDistance > competition[1]->m_crowdingDistance ? competition[0] : competition[1];
}
}
}
}
