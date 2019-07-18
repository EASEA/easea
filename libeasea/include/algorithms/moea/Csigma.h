/***********************************************************************
| Csigma.h                                                              |
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
#include <CLogger.h>
#include <algorithms/moea/CmoeaAlgorithm.h>

#include <operators/crossover/wrapper/CWrapCrossover.h>
#include <operators/mutation/wrapper/CWrapMutation.h>
#include <operators/selection/nondominateSelection.h>
#include <operators/selection/totalSelection.h>

#include <shared/CRandom.h>
#include <shared/CConstant.h>
#include <shared/CArchive.h>
#include <shared/functions/breeding.h>
#include <shared/functions/crowdingDistance.h>
#include <shared/functions/dominance.h>



namespace easea
{
namespace algorithms
{
namespace sigma
{
template <typename TIndividual, typename TRandom>
class Csigma : public CmoeaAlgorithm<std::vector< TIndividual >, TRandom>, public easea::shared::CArchive<TIndividual>, public easea::operators::crossover::CWrapCrossover<typename TIndividual::TO, typename TIndividual::TV>, public easea::operators::mutation::CWrapMutation<typename TIndividual::TO, typename TIndividual::TV>
{
public:
        typedef TIndividual TI  ;
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;
//        typedef TRandom TR;

        typedef std::vector<TI> TPopulation;
        typedef CmoeaAlgorithm<TPopulation, TRandom> TBase;
	typedef easea::shared::CArchive<TI> TBaseArchive;
        typedef typename TBase::TP TP;
        typedef typename easea::operators::crossover::CWrapCrossover<TO, TV>::TC TC;
        typedef typename easea::operators::mutation::CWrapMutation<TO, TV>::TM TM;

        Csigma(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation, size_t maxArchSize);
        ~Csigma(void);
        TPopulation runBreeding(const TPopulation &parent);
        static bool isDominated(const TIndividual &individual1, const TIndividual &individual2);


protected:
        static bool Dominate(const TI *individual1, const TI *individual2);
        void makeOneGeneration(){};
	void makeOneGeneration(bool stop);
        template <typename TPtr, typename TIter> static TIter selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end);
        template <typename TPtr, typename TIter> static TIter selectCrit(const std::list<TPtr> &front, TIter begin, TIter end);
        static const TI *comparer(const std::vector<const TI *> &comparator);
private:
	std::uniform_real_distribution<TO> m_distribution;
	size_t m_size;
};

template <typename TIndividual, typename TRandom>
Csigma<TIndividual, TRandom>::Csigma(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation, size_t maxArchSize)
        : TBase(random, problem, initial), TBaseArchive(maxArchSize)
        , easea::operators::crossover::CWrapCrossover<TO, TV>(crossover), easea::operators::mutation::CWrapMutation<TO, TV>(mutation)
	, m_distribution(0,1)	
{
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
Csigma<TIndividual, TRandom>::~Csigma(void)
{
}

template <typename TIndividual, typename TRandom>
typename Csigma<TIndividual, TRandom>::TPopulation Csigma<TIndividual, TRandom>::runBreeding(const TPopulation &parent)
{
        TPopulation offspring = easea::shared::functions::runBreeding(m_size, parent.begin(), parent.end(), this->getRandom(), &comparer, this->getCrossover());

        for (size_t i = 0; i < offspring.size(); ++i)
        {
                TI &child = offspring[i];

                this->getMutation()(child);
                TBase::getProblem()(child);
        }
        return offspring;
}

template <typename TIndividual, typename TRandom>
bool Csigma<TIndividual, TRandom>::isDominated(const TI &individual1, const TI &individual2)
{
        return easea::shared::functions::isDominated(individual1.m_objective, individual2.m_objective);
}

template <typename TIndividual, typename TRandom>
bool Csigma<TIndividual, TRandom>::Dominate(const TI *individual1, const TI *individual2)
{
        return isDominated(*individual1, *individual2);
}



template <typename TIndividual, typename TRandom>
void Csigma<TIndividual, TRandom>::makeOneGeneration(bool stop)
{
        TPopulation parent = TBase::m_population;
        TPopulation offspring = runBreeding(parent);
        typedef typename TPopulation::pointer TPtr;
	typedef typename TPopulation::iterator TIter;
	
	if (stop == true){
		easea::shared::CArchive<TIndividual>::setMaxSize(offspring.size());
	}
	for (size_t i = 0; i < offspring.size(); ++i)
		easea::shared::CArchive<TIndividual>::updateArchive(offspring[i]);
	size_t szPopDiv2 = (offspring.size() - TBaseArchive::m_archive.size()) / 2;
	size_t icounter = TBaseArchive::m_archive.size();

	std::vector<TPtr> archPop;
	for (size_t i = 0; i < TBaseArchive::m_archive.size(); ++i)
    		archPop.push_back(&TBaseArchive::m_archive[i]);

	
	TBase::m_population.resize(archPop.size());
	
	easea::operators::selection::totalSelection(archPop, TBase::m_population.begin(), TBase::m_population.end());
}


template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Csigma<TIndividual, TRandom>::selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end)
{
        std::vector<TPtr> iFront(front.begin(), front.end());
        easea::shared::functions::setCrowdingDistance<TO>(iFront.begin(), iFront.end());
        if (iFront.size() > std::distance(begin, end)) LOG_ERROR(errorCode::value, "Select Noncritical : Error of front size");
        TIter dest = begin;
        for (size_t i = 0; i < iFront.size(); ++i, ++dest)
                *dest = *iFront[i];
        return dest;
}

template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Csigma<TIndividual, TRandom>::selectCrit(const std::list<TPtr> &front, TIter begin, TIter end)
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

template <typename TIndividual, typename TRandom>
const typename Csigma<TIndividual, TRandom>::TI *Csigma<TIndividual, TRandom>::comparer(const std::vector<const TI *> &comparator)
{
        if (isDominated(*comparator[0], *comparator[1]))
                return comparator[0];
        else if (isDominated(*comparator[1], *comparator[0]))
                return comparator[1];

        return comparator[0]->m_crowdingDistance > comparator[1]->m_crowdingDistance ? comparator[0] : comparator[1];
}
}
}
}
