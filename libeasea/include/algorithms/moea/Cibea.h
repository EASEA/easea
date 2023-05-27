/***********************************************************************
| Cibea.h                                                               |
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
#include <shared/CConstant.h>
#include <shared/functions/breeding.h>
#include <shared/functions/ndi.h>
#include <shared/functions/dominance.h>
#include <config.h>


namespace easea
{
namespace algorithms
{
namespace ibea
{
template <typename TIndividual, typename TRandom>
class Cibea : public CmoeaAlgorithm<std::vector< TIndividual >, TRandom>,  public easea::operators::crossover::CWrapCrossover<typename TIndividual::TO, typename TIndividual::TV>, public easea::operators::mutation::CWrapMutation<typename TIndividual::TO, typename TIndividual::TV>
{
public:

        typedef TIndividual TI;
        typedef typename TI::TO TO;
	typedef typename TI::TV TV;
        
        typedef std::vector<TI> TPopulation;
        typedef CmoeaAlgorithm<TPopulation, TRandom> TBase;

        typedef typename TBase::TP TP;
        typedef typename easea::operators::crossover::CWrapCrossover<TO, TV>::TC TC;
        typedef typename easea::operators::mutation::CWrapMutation<TO, TV>::TM TM;

        Cibea(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation, const TO scale = 0.05);
        virtual ~Cibea(void);
        TO getScale(void) const;
        TO calcIndicator(const TIndividual &individual1, const TIndividual &individual2);
        template <typename TIter, typename TF> void setFitness(TIter begin, TIter end, TF f);
        template <typename TIter, typename TF> void updateFitness(const TI &selected, TIter begin, TIter end, TF f);
	TPopulation runBreeding(const TPopulation &parent);
        template <typename TPtr> void environmentalSelection(std::list<TPtr> &unionPop, TPopulation &solutionSet);
        template <typename TPtr> void reduction(size_t count, std::list<TPtr> &population);
        static bool isDominated(const TI &individual1, const TI &individual2);
protected:
        static bool Dominate(const TI *individual1, const TI *individual2);
        void makeOneGeneration(void);

	static const TIndividual *comparer(const std::vector<const TIndividual *> &comparator);
        TO runCalcIndicator(const TIndividual &individual1, const TIndividual &individual2);
	static TO calcIndicator(const std::vector<TO> &objective1, const std::vector<TO> &objective2);



private:
        TO m_scale;

};

template <typename TIndividual, typename TRandom>
Cibea<TIndividual, TRandom>::Cibea(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation
        , const TO scale)
        : TBase(random, problem, initial)
        , easea::operators::crossover::CWrapCrossover<TO, TV>(crossover)
        , easea::operators::mutation::CWrapMutation<TO, TV>(mutation)
        , m_scale(scale)
{
	setFitness(TBase::m_population.begin(), TBase::m_population.end(), [](TIndividual &individual)->TIndividual &{return individual;});
}

template <typename TIndividual, typename TRandom>
Cibea<TIndividual, TRandom>::~Cibea(void)
{
}

template <typename TIndividual, typename TRandom>
typename Cibea<TIndividual, TRandom>::TO Cibea<TIndividual, TRandom>::runCalcIndicator(const TIndividual &individual1, const TIndividual &individual2)
{
        return calcIndicator(individual1.m_objective, individual2.m_objective);
}
template <typename TIndividual, typename TRandom>
typename Cibea<TIndividual, TRandom>::TO Cibea<TIndividual, TRandom>::calcIndicator(const std::vector<TO> &objective1, const std::vector<TO> &objective2) 
{
        if (objective1.size() <=  0) LOG_ERROR(errorCode::value, "Wrong number of objective");
        if (objective1.size() != objective2.size()) LOG_ERROR(errorCode::value, "Wrong number of objective");
        TO maxEpsilon = objective2[0] - objective1[0];
        for (size_t i = 1; i < objective1.size(); ++i)
        {
                const TO epsilon = objective2[i] - objective1[i];
                if (epsilon > maxEpsilon)
                        maxEpsilon = epsilon;
        }
        return maxEpsilon;
}

template <typename TIndividual, typename TRandom>
typename TIndividual::TO Cibea<TIndividual, TRandom>::getScale(void) const
{
        return m_scale;
}

template <typename TIndividual, typename TRandom>
typename TIndividual::TO Cibea<TIndividual, TRandom>::calcIndicator(const TIndividual &individual1, const TIndividual &individual2)
{
        if (individual1.m_objective.size() != individual2.m_objective.size()) LOG_ERROR(errorCode::value,"Wrong number of objective");
        return runCalcIndicator(individual1, individual2);
}

template <typename TIndividual, typename TRandom>
template <typename TIter, typename TF> void Cibea<TIndividual, TRandom>::setFitness(TIter begin, TIter end, TF f)
{
        for (TIter individual = begin; individual != end; ++individual)
        {
                TIndividual &iindividual = f(*individual);
                iindividual.m_fitness= 0;
                for (TIter remaining = begin; remaining != end; ++remaining)
                {
                        if (remaining != individual)
                        {
                                const TIndividual &iremaining = f(*remaining);
                                const TO indicator = calcIndicator(iindividual, iremaining);
                                iindividual.m_fitness-= exp(-indicator / m_scale);
                        }
                }
        }
}
template <typename TIndividual, typename TRandom>
template <typename TIter, typename TF> void Cibea<TIndividual, TRandom>::updateFitness(const TI &selected, TIter begin, TIter end, TF f)
{
        for (TIter update = begin; update != end; ++update)
        {
                TIndividual &iupdate = f(*update);
                const TO indicator = calcIndicator(iupdate, selected);
                iupdate.m_fitness+= exp(-indicator / m_scale);
        }
}

template <typename TIndividual, typename TRandom>
typename Cibea<TIndividual, TRandom>::TPopulation Cibea<TIndividual, TRandom>::runBreeding(const TPopulation &parent)
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
template <typename TPtr> void Cibea<TIndividual, TRandom>::environmentalSelection(std::list<TPtr> &unionPop, TPopulation &solutionSet)
{
        setFitness(unionPop.begin(), unionPop.end(), [](TPtr individual)->TIndividual &{return *individual;});
        if (unionPop.size() <= solutionSet.size()) LOG_ERROR(errorCode::value,"Wrong solutions number");
        reduction(unionPop.size() - solutionSet.size(), unionPop);
        if (unionPop.size() != solutionSet.size()) LOG_ERROR(errorCode::value, "Wrong solution number");
        size_t index = 0;
        for (auto i = unionPop.begin(); i != unionPop.end(); ++i, ++index)
                solutionSet[index] = **i;
}
template <typename TIndividual, typename TRandom>
template <typename TPtr> void Cibea<TIndividual, TRandom>::reduction(size_t count, std::list<TPtr> &population)
{
        if (count >= population.size()) LOG_ERROR(errorCode::value, "Wrong population size");
        std::list<TPtr> abandon;
        while (count)
        {
                auto worst = std::min_element(population.begin(), population.end(), [](TPtr individual1, TPtr individual2)->bool{return individual1->m_fitness < individual2->m_fitness;});
                abandon.splice(abandon.end(), population, worst);
                updateFitness(*abandon.back(), population.begin(), population.end(), [](TPtr individual)->TIndividual &{return *individual;});
                --count;
        }
}

template <typename TIndividual, typename TRandom>
bool Cibea<TIndividual, TRandom>::isDominated(const TI &individual1, const TI &individual2)
{
        return easea::shared::functions::isDominated(individual1.m_objective, individual2.m_objective);
}

template <typename TIndividual, typename TRandom>
bool Cibea<TIndividual, TRandom>::Dominate(const TI *individual1, const TI *individual2)
{
        return isDominated(*individual1, *individual2);
}

template <typename TIndividual, typename TRandom>
void Cibea<TIndividual, TRandom>::makeOneGeneration(void)
{
        TPopulation parent = TBase::m_population;
        TPopulation offspring = runBreeding(parent);
        typedef typename TPopulation::pointer TPtr;
        std::list<TPtr> unionPop;
        for (size_t i = 0; i < parent.size(); ++i)
                unionPop.push_back(&parent[i]);
        for (size_t i = 0; i < offspring.size(); ++i)
                unionPop.push_back(&offspring[i]);
        environmentalSelection(unionPop, TBase::m_population);
}

template <typename TIndividual, typename TRandom>
const typename Cibea<TIndividual, TRandom>::TI  *Cibea<TIndividual, TRandom>::comparer(const std::vector<const TIndividual *> &comparator)
{
        if (isDominated(*comparator[0], *comparator[1]))
                return comparator[0];
        else if (isDominated(*comparator[1], *comparator[0]))
                return comparator[1];
    //    if (comparator[0]->m_fitness > 0) LOG_ERROR(errorCode::value,"Wrong fitness value");
    //    if (comparator[1]->m_fitness > 0) LOG_ERROR(errorCode::value, "Wrong fitness value");
        return comparator[0]->m_fitness< comparator[1]->m_fitness? comparator[0] : comparator[1];
}
}
}
}


