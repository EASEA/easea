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

#include <operators/crossover/base/C2x2Crossover.h>

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
#include <config.h>

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

	TO calcIndicator(const TIndividual &individual1, const TIndividual &individual2);
	template <typename TIter, typename TF> void setFitness(TIter begin, TIter end, TF f);
	void on_individuals_received() override;

protected:
        static bool Dominate(const TI *individual1, const TI *individual2);
        void makeOneGeneration() override;
	void initialize() override;
        static const TI *comparer(const std::vector<const TI *> &comparator);

	TO runCalcIndicator(const TIndividual &individual1, const TIndividual &individual2);
        static TO calcIndicator(const std::vector<TO> &objective1, const std::vector<TO> &objective2);

private:
	std::uniform_real_distribution<TO> m_distribution;
	size_t m_size;
	TO m_scale;
};

template <typename TIndividual, typename TRandom>
Csigma<TIndividual, TRandom>::Csigma(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation, size_t maxArchSize)
        : TBase(random, problem, initial), TBaseArchive(maxArchSize)
        , easea::operators::crossover::CWrapCrossover<TO, TV>(crossover), easea::operators::mutation::CWrapMutation<TO, TV>(mutation)
	, m_distribution(0,1)	
{
}

template <typename TIndividual, typename TRandom>
void Csigma<TIndividual, TRandom>::initialize() {
	TBase::initialize();
	typedef typename TPopulation::pointer TPtr;
        std::list<TPtr> population;
        for (size_t i = 0; i < TBase::m_population.size(); ++i)
                population.push_back(&TBase::m_population[i]);
	m_size = TBase::m_population.size();
	m_scale =  0.05;
        while (!population.empty())
        {
                std::list<TPtr> nondominate = easea::shared::functions::getNondominated(population, &Dominate);
                std::vector<TPtr> _nondominate(nondominate.begin(), nondominate.end());
                easea::shared::functions::setCrowdingDistance<TO>(_nondominate.begin(), _nondominate.end());
        }
}

template <typename TIndividual, typename TRandom>
void Csigma<TIndividual, TRandom>::on_individuals_received()
{
	// recalculate crowding distance
	typedef typename TPopulation::pointer TPtr;
	std::list<TPtr> population;
	for (size_t i = 0; i < TBase::m_population.size(); ++i)
		population.push_back(&TBase::m_population[i]);

	while (!population.empty()) {
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
	this->getCrossover().setLimitGen(this->getLimitGeneration());
	this->getCrossover().setCurrentGen(this->getCurrentGeneration());
	this->getMutation().setLimitGen(this->getLimitGeneration());
	this->getMutation().setCurrentGen(this->getCurrentGeneration());

        TPopulation offspring = easea::shared::functions::runBreeding(m_size,  parent.begin(), parent.end(), this->getRandom(), &comparer, this->getCrossover());

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
typename Csigma<TIndividual, TRandom>::TO Csigma<TIndividual, TRandom>::runCalcIndicator(const TIndividual &individual1, const TIndividual &individual2)
{
        return calcIndicator(individual1.m_objective, individual2.m_objective);
}
template <typename TIndividual, typename TRandom>
typename Csigma<TIndividual, TRandom>::TO Csigma<TIndividual, TRandom>::calcIndicator(const std::vector<TO> &objective1, const std::vector<TO> &objective2)
{
        if (objective1.size() <=  0) LOG_ERROR(errorCode::value, "Wrong number of objective");
        if (objective1.size() != objective2.size()) LOG_ERROR(errorCode::value, "Wrong number of objective");
        TO maxEpsilon = objective2[0] - objective1[0];
        for (size_t i = 1; i < objective1.size(); ++i)
        {
                const TO epsilon = objective2[i] -objective1[i];
                if (epsilon > maxEpsilon)
                        maxEpsilon = epsilon;
        }
        return maxEpsilon;
}
template <typename TIndividual, typename TRandom>
typename TIndividual::TO Csigma<TIndividual, TRandom>::calcIndicator(const TIndividual &individual1, const TIndividual &individual2)
{
        if (individual1.m_objective.size() != individual2.m_objective.size()) LOG_ERROR(errorCode::value,"Wrong number of objective");
        return runCalcIndicator(individual1, individual2);
}
template <typename TIndividual, typename TRandom>
template <typename TIter, typename TF> void Csigma<TIndividual, TRandom>::setFitness(TIter begin, TIter end, [[maybe_unused]] TF f)
{
        for (TIter individual = begin; individual != end; ++individual)
        {
                TIndividual &iindividual = *individual;//f(*individual);
                iindividual.m_fitness= 0;
                for (TIter remaining = begin; remaining != end; ++remaining)
                {
                        if (remaining != individual)
                        {
                                const TIndividual &iremaining = *remaining;//f(*remaining);
                                const TO indicator = calcIndicator(iindividual, iremaining);
                                iindividual.m_fitness -= exp(-indicator /m_scale);

                        }
                }
        }
}


template <typename TIndividual, typename TRandom>
void Csigma<TIndividual, TRandom>::makeOneGeneration()
{
        TPopulation parent = TBase::m_population;

        TPopulation offspring = runBreeding(parent);
        typedef typename TPopulation::pointer TPtr;
	//typedef typename TPopulation::iterator TIter; // unused
	bool epsilon = false;
	
	if (TBase::checkIsLast() == true){
		easea::shared::CArchive<TIndividual>::setMaxSize(offspring.size());
		epsilon = false;
	}
	else{
		if (TBase::getCurrentGeneration() > 1){
		    setFitness(TBaseArchive::m_archive.begin(), TBaseArchive::m_archive.end(), [](TPtr individual)->TIndividual &{return *individual;});
		    epsilon = true;
		}
	}
	for (size_t i = 0; i < offspring.size(); ++i){
		easea::shared::CArchive<TIndividual>::updateArchiveEpsilon(offspring[i], epsilon);
		//if (TBaseArchive::m_same == true) cc++;
	}
	[[maybe_unused]] size_t szPopDiv2 = (offspring.size() - TBaseArchive::m_archive.size()) / 2;
	size_t icounter = TBaseArchive::m_archive.size();
	TBase::m_population.resize(icounter);


	for (size_t i = 0; i < TBaseArchive::m_archive.size(); ++i)
	{
		
		TBase::m_population[i] = TBaseArchive::m_archive[i];
		TBase::m_population[i].m_fitness = TBaseArchive::m_archive[i].m_fitness;

/*		if (i > 0){
		    if ((TBase::m_population[i].m_crowdingDistance-TBase::m_population[i-1].m_crowdingDistance)<0.001)
			cc++;
		}*/
	}
	
//	easea::operators::selection::totalSelection(archPop, TBase::m_population.begin(), TBase::m_population.end());
}


template <typename TIndividual, typename TRandom>
const typename Csigma<TIndividual, TRandom>::TI *Csigma<TIndividual, TRandom>::comparer(const std::vector<const TI *> &comparator)
{
        if (isDominated(*comparator[0], *comparator[1]))
                return comparator[0];
        else if (isDominated(*comparator[1], *comparator[0]))
                return comparator[1];
return comparator[0]->m_fitness > comparator[1]->m_fitness ? comparator[0] : comparator[1];
///        return comparator[0]->m_crowdingDistance > comparator[1]->m_crowdingDistance ? comparator[0] : comparator[1];


}
}
}
}
