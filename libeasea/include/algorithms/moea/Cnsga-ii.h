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
#include <CLogger.h>
#include <algorithms/moea/CmoeaAlgorithm.h>

#include <operators/crossover/wrapper/CWrapCrossover.h>
#include <operators/mutation/wrapper/CWrapMutation.h>
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
namespace nsga_ii
{
template <typename TIndividual, typename TRandom>
class Cnsga_ii : public CmoeaAlgorithm<std::vector<TIndividual>, TRandom>, public easea::operators::crossover::CWrapCrossover<typename TIndividual::TO, typename TIndividual::TV>, public easea::operators::mutation::CWrapMutation<typename TIndividual::TO, typename TIndividual::TV>
{
public:
        typedef TIndividual TI  ;
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;

        typedef std::vector<TI> TPopulation;
        typedef CmoeaAlgorithm<TPopulation, TRandom> TBase;
        typedef typename TBase::TP TP;
        typedef typename easea::operators::crossover::CWrapCrossover<TO, TV>::TC TC;
        typedef typename easea::operators::mutation::CWrapMutation<TO, TV>::TM TM;

        Cnsga_ii(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation);
        ~Cnsga_ii(void);
        TPopulation runBreeding(const TPopulation &parent);
        static bool isDominated(const TIndividual &individual1, const TIndividual &individual2);
	void on_individuals_received() override;


protected:
        static bool Dominate(const TI *individual1, const TI *individual2);
        void makeOneGeneration(void) override;
	void initialize() override;
        template <typename TPtr, typename TIter> static TIter selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end);
        template <typename TPtr, typename TIter> static TIter selectCrit(const std::list<TPtr> &front, TIter begin, TIter end);
        static const TI *comparer(const std::vector<const TI *> &comparator);
};

template <typename TIndividual, typename TRandom>
Cnsga_ii<TIndividual, TRandom>::Cnsga_ii(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation)
        : TBase(random, problem, initial)
        , easea::operators::crossover::CWrapCrossover<TO, TV>(crossover), easea::operators::mutation::CWrapMutation<TO, TV>(mutation)
{
}

template <typename TIndividual, typename TRandom>
void Cnsga_ii<TIndividual, TRandom>::initialize() {
	TBase::initialize();
  	typedef typename TPopulation::pointer TPtr;
  std::list<TPtr> population;
        for (size_t i = 0; i < TBase::m_population.size(); ++i)
	        population.push_back(&TBase::m_population[i]);

	bool first_layer = true;
        while (!population.empty())
        {
                std::list<TPtr> nondominate = easea::shared::functions::getNondominated(population, &Dominate, first_layer);
                std::vector<TPtr> _nondominate(nondominate.begin(), nondominate.end());
                easea::shared::functions::setCrowdingDistance<TO>(_nondominate.begin(), _nondominate.end());
		first_layer = false;
        }
}

template <typename TIndividual, typename TRandom>
void Cnsga_ii<TIndividual, TRandom>::on_individuals_received() {
typedef typename TPopulation::pointer TPtr;
  std::list<TPtr> population;
        for (size_t i = 0; i < TBase::m_population.size(); ++i)
	        population.push_back(&TBase::m_population[i]);

	bool first_layer = true;
        while (!population.empty())
        {
                std::list<TPtr> nondominate = easea::shared::functions::getNondominated(population, &Dominate, first_layer);
                std::vector<TPtr> _nondominate(nondominate.begin(), nondominate.end());
                easea::shared::functions::setCrowdingDistance<TO>(_nondominate.begin(), _nondominate.end());
		first_layer = false;
        }
}

template <typename TIndividual, typename TRandom>
Cnsga_ii<TIndividual, TRandom>::~Cnsga_ii(void)
{
}

template <typename TIndividual, typename TRandom>
typename Cnsga_ii<TIndividual, TRandom>::TPopulation Cnsga_ii<TIndividual, TRandom>::runBreeding(const TPopulation &parent)
{
	this->getCrossover().setLimitGen(this->getLimitGeneration());
        this->getCrossover().setCurrentGen(this->getCurrentGeneration());

        TPopulation offspring = easea::shared::functions::runBreeding(parent.size(), parent.begin(), parent.end(), this->getRandom(), &comparer, this->getCrossover());

#ifdef USE_OPENMP
    EASEA_PRAGMA_OMP_PARALLEL
#endif
        for (int i = 0; i < static_cast<int>(offspring.size()); ++i)
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
        return easea::shared::functions::isDominated(individual1.m_objective, individual2.m_objective, std::less<TO>{});
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
        std::vector<TPtr> iFront(front.begin(), front.end());
        easea::shared::functions::setCrowdingDistance<TO>(iFront.begin(), iFront.end());
        if (static_cast<long int>(iFront.size()) > std::distance(begin, end)) LOG_ERROR(errorCode::value, "Select Noncritical : Error of front size");
        TIter dest = begin;
        for (size_t i = 0; i < iFront.size(); ++i, ++dest)
                *dest = *iFront[i];
        return dest;
}

template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Cnsga_ii<TIndividual, TRandom>::selectCrit(const std::list<TPtr> &front, TIter begin, TIter end)
{
        std::vector<TPtr> iFront(front.begin(), front.end());
        easea::shared::functions::setCrowdingDistance<TO>(iFront.begin(), iFront.end());
        std::partial_sort(iFront.begin(), iFront.begin() + std::distance(begin, end), iFront.end()
                , [](TPtr individual1, TPtr individual2)->bool{return individual1->m_crowdingDistance > individual2->m_crowdingDistance;});
        if (static_cast<long int>(iFront.size()) < std::distance(begin, end)) LOG_ERROR(errorCode::value, "Select critical : Error of front size!");
        TIter dest = begin;
        for (size_t i = 0; dest != end; ++i, ++dest)
                *dest = *iFront[i];
        return dest;
}

template <typename TIndividual, typename TRandom>
const typename Cnsga_ii<TIndividual, TRandom>::TI *Cnsga_ii<TIndividual, TRandom>::comparer(const std::vector<const TI *> &comparator)
{
        if (isDominated(*comparator[0], *comparator[1]))
                return comparator[0];
        else if (isDominated(*comparator[1], *comparator[0]))
                return comparator[1];
        if (comparator[0]->m_crowdingDistance < 0) LOG_ERROR(errorCode::value, "Crowding distance < 0 ");
        if (comparator[1]->m_crowdingDistance < 0) LOG_ERROR(errorCode::value, "Crowding distance < 0 ");


        return comparator[0]->m_crowdingDistance > comparator[1]->m_crowdingDistance ? comparator[0] : comparator[1];
}
}
}
}
