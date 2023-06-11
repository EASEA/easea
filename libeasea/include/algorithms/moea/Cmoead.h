/***********************************************************************
| Cmoead.h                                                              |
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

#include <operators/crossover/wrapper/CWrap2x2Crossover.h>
#include <operators/mutation/wrapper/CWrapMutation.h>
#include <operators/selection/nondominateSelection.h>

#include <shared/CRandom.h>
#include <shared/CConstant.h>
#include <shared/functions/breeding.h>
#include <shared/functions/neighbors.h>
#include <shared/functions/aggregation.h>
#include <shared/functions/weight.h>
#include <shared/functions/dominance.h>

#include <config.h>

namespace easea
{
namespace algorithms
{
namespace moead
{

template <typename TIndividual, typename TRandom>
class Cmoead : public CmoeaAlgorithm<std::vector<TIndividual>, TRandom>, public easea::operators::crossover::CWrap2x2Crossover<typename TIndividual::TO, 
typename TIndividual::TV>, public easea::operators::mutation::CWrapMutation<typename TIndividual::TO, typename TIndividual::TV>
{
public:

        typedef TIndividual TI;
        typedef typename TI::TO TO;
	typedef typename TI::TV TV;
        
        typedef std::vector<TI> TPopulation;
        typedef CmoeaAlgorithm<TPopulation, TRandom> TBase;

        typedef typename TBase::TP TP;
        typedef typename easea::operators::crossover::CWrap2x2Crossover<TO, TV>::TC TC;
        typedef typename easea::operators::mutation::CWrapMutation<TO, TV>::TM TM;

	typedef std::vector<TO> TPoint;
	typedef std::vector<size_t> TNb;

        Cmoead(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation, const std::vector<TPoint> &weight, size_t nbNeighbors );
        ~Cmoead(void);
	const std::vector<TPoint> &getWeight(void) const;
	TO aggregate(const TPoint &obj, const TPoint &weight);
	    
protected:
	std::vector<TO> m_refPoint;
	std::vector<TNb> m_neighbors;
        void makeOneGeneration(void) override;
  	void initialize() override;
	void updateRef(const TI &individual);
	void updateNeighbors(const TI &individual, const TNb &neibors);
	TO doAggregate(const TPoint &obj, const TPoint &w);

private:
        std::vector<TPoint> m_weight;
	size_t m_nbNeighbors;
};

template <typename TIndividual, typename TRandom>
Cmoead<TIndividual, TRandom>::Cmoead(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation
        , const std::vector<TPoint> &weight, size_t nbNeighbors)
        : TBase(random, problem, initial)
        , easea::operators::crossover::CWrap2x2Crossover<TO, TV>(/*this->getCrossover()*/crossover)
        , easea::operators::mutation::CWrapMutation<TO, TV>(mutation)
        , m_weight(weight)
	, m_nbNeighbors(nbNeighbors)
{
    if (weight.size() !=  initial.size()) LOG_ERROR(errorCode::value, "Wrong number of initial population");
}

template <typename TIndividual, typename TRandom>
void Cmoead<TIndividual, TRandom>::initialize() {
	TBase::initialize();

	m_refPoint = TBase::m_population[0].m_objective;

	for (size_t i = 1; i < TBase::m_population.size(); ++i)
		updateRef(TBase::m_population[i]);
	m_neighbors = easea::shared::function::initNeighbors(easea::shared::function::calcAdjacencyMatrix<TO>(m_weight.begin(), m_weight.end()), m_nbNeighbors);
}

template <typename TIndividual, typename TRandom>
Cmoead<TIndividual, TRandom>::~Cmoead(void)
{
}

template <typename TIndividual, typename TRandom> 
typename Cmoead<TIndividual, TRandom>::TO Cmoead<TIndividual, TRandom>::doAggregate(const TPoint &obj, const TPoint &w) 
{
	const auto dir = easea::shared::function::computeDirection(obj, m_refPoint);
	return easea::shared::function::aggregation::chebuchev(w, dir);
}

template <typename TIndividual, typename TRandom> 
typename Cmoead<TIndividual, TRandom>::TO Cmoead<TIndividual, TRandom>::aggregate(const TPoint &obj, const TPoint &w)
{
	return doAggregate(obj, w);
}

template <typename TIndividual, typename TRandom> 
void Cmoead<TIndividual, TRandom>::updateRef(const TIndividual &individual)
{
	if (m_refPoint.size() != individual.m_objective.size()) LOG_ERROR(errorCode::value, "Wrong number of objective");
	for (size_t i = 0; i < m_refPoint.size(); ++i)
	{
		if (individual.m_objective[i] < m_refPoint[i])
			m_refPoint[i] = individual.m_objective[i];
	}

}
template <typename TIndividual, typename TRandom>
void Cmoead<TIndividual, TRandom>::updateNeighbors(const TIndividual &individual, const TNb &neighbors)
{
	for (size_t i = 0; i < neighbors.size(); ++i)
	{
		const size_t inx = neighbors[i];
		const TPoint &w = m_weight[inx];
		
		TIndividual &tmp_individual = TBase::m_population[inx];
		if (aggregate(individual.m_objective, w) < aggregate(tmp_individual.m_objective, w))
			tmp_individual = individual;
	}
}

template <typename TIndividual, typename TRandom>
void Cmoead<TIndividual, TRandom>::makeOneGeneration(void)
{
    this->getCrossover().setLimitGen(this->getLimitGeneration());
    this->getCrossover().setCurrentGen(this->getCurrentGeneration());

#ifdef USE_OPENMP
    EASEA_PRAGMA_OMP_PARALLEL
#endif
    for (int i = 0; i < static_cast<int>(TBase::m_population.size()); ++i)
    {
	const std::vector<size_t> &neighbors = m_neighbors[i];
	if (neighbors.size() <= 0) LOG_ERROR(errorCode::value, "Wrong neighbors  size");

	std::uniform_int_distribution<size_t> dist(0, neighbors.size() - 1);
	const TIndividual &parent1 = TBase::m_population[neighbors[dist(this->getRandom())]];
	const TIndividual &parent2 = TBase::m_population[neighbors[dist(this->getRandom())]];

	TIndividual offspring1, offspring2;

	this->getCrossover()(parent1, parent2, offspring1, offspring2);
	this->getMutation()(offspring1);
	TBase::getProblem()(offspring1);
	updateRef(offspring1);
	updateNeighbors(offspring1, neighbors);
	this->getMutation()(offspring2);

	TBase::getProblem()(offspring2);
	updateRef(offspring2);
	updateNeighbors(offspring2, neighbors);
	
	}
    }
}
}
}


