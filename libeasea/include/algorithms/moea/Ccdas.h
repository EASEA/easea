/***********************************************************************
| Ccdas.h   		                                                |
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
#include <shared/CConstant.h>
#include <shared/CRandom.h>
#include <shared/functions/dominance.h>
#include <shared/functions/crowdingDistance.h>
#include <shared/functions/breeding.h>
#include <algorithms/moea/CmoeaAlgorithm.h>
#include <operators/crossover/wrapper/CWrapCrossover.h>
#include <operators/mutation/wrapper/CWrapMutation.h>
#include <operators/selection/nondominateSelection.h>



namespace easea
{
namespace algorithms
{
namespace cdas
{
template <typename TIndividual, typename TRandom>
class Ccdas : public CmoeaAlgorithm<std::vector<TIndividual>>, public easea::shared::CRandom<TRandom>, public easea::operators::crossover::CWrapCrossover<typename TIndividual::TO,typename  TIndividual::TV>, public easea::operators::mutation::CWrapMutation<typename TIndividual::TO,typename  TIndividual::TV>
{
public:
        typedef TIndividual TI;
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;
        typedef TRandom TR;
        typedef std::vector<TI> TPopulation;
        typedef CmoeaAlgorithm<TPopulation> TBase;
        typedef typename TBase::TP TP;
        typedef typename easea::operators::crossover::CWrapCrossover<TO, TV>::TC TC;
        typedef typename easea::operators::mutation::CWrapMutation<TO, TV>::TM TM;

        Ccdas(TR random, TP &problem, const std::vector<TV> &initPop, TC &crossover, TM &mutation, const std::vector<TO> &angle);
        ~Ccdas(void);
        const std::vector<double> &getAngle(void) const;
        TPopulation runBreeding(const TPopulation &parent);
        static bool isDominated(const TI &individual1, const TI &individual2);
        void convertObjective(const std::vector<TO> &angle, const std::vector<TO> &objective, std::vector<TO> &convertedObjective);


protected:
        static bool Dominate(const TI *individual1, const TI *individual2);
        void makeOneGeneration(void);
        template <typename TPtr, typename TIter> static TIter selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end);
        template <typename TPtr, typename TIter> static TIter selectCrit(const std::list<TPtr> &front, TIter begin, TIter end);
        static const TIndividual *comparer(const std::vector<const TIndividual *> &comparator);

private:
        std::vector<double> m_angle;
};

template <typename TIndividual, typename TRandom>
Ccdas<TIndividual, TRandom>::Ccdas(TR random, TP &problem, const std::vector<TV> &initPop, TC &crossover, TM &mutation, const std::vector<TO> &angle)
        : TBase(problem), easea::shared::CRandom<TR>(random), easea::operators::crossover::CWrapCrossover<TO, TV>(crossover)
        , easea::operators::mutation::CWrapMutation<TO, TV>(mutation), m_angle(angle)
{
        TBase::m_population.resize(initPop.size());
        for (size_t i = 0; i < initPop.size(); ++i)
        {
                TIndividual &individual = TBase::m_population[i];
                individual.m_variable = initPop[i];
                TBase::getProblem()(individual);
                convertObjective(m_angle, individual.m_objective, individual.m_convertedObjective);
        }

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
Ccdas<TIndividual, TRandom>::~Ccdas(void)
{
}

template <typename TIndividual, typename TRandom>
const std::vector<double> &Ccdas<TIndividual, TRandom>::getAngle(void) const
{
        return m_angle;
}

template <typename TIndividual, typename TRandom>
typename Ccdas<TIndividual, TRandom>::TPopulation Ccdas<TIndividual, TRandom>::runBreeding(const TPopulation &parent)
{
        TPopulation offspring = easea::shared::functions::runBreeding(parent.size(), parent.begin(), parent.end(), this->getRandom(), &comparer, this->getCrossover());
        for (size_t i = 0; i < offspring.size(); ++i)
        {
                TIndividual &child = offspring[i];
                this->getMutation()(child);
                TBase::getProblem()(child);
                convertObjective(m_angle, child.m_objective, child.m_convertedObjective);
        }
        return offspring;
}

template <typename TIndividual, typename TRandom>
bool Ccdas<TIndividual, TRandom>::isDominated(const TIndividual &individual1, const TIndividual &individual2)
{
        return easea::shared::functions::isDominated(individual1.m_convertedObjective, individual2.m_convertedObjective);
}

template <typename TIndividual, typename TRandom>
bool Ccdas<TIndividual, TRandom>::Dominate(const TIndividual *individual1, const TIndividual *individual2)
{
        return isDominated(*individual1, *individual2);
}
        

template <typename TIndividual, typename TRandom>
void Ccdas<TIndividual, TRandom>::convertObjective(const std::vector<TO> &angle, const std::vector<TO> &objective, std::vector<TO> &convertedObjective)
{
        assert(objective.size() == angle.size());
        convertedObjective.resize(objective.size());
        const TO radius = sqrt(std::inner_product(objective.begin(), objective.end(), objective.begin(), (TO)0));

        for (size_t i = 0; i < objective.size(); ++i)
        {
                if (angle[i] < 0 && angle[i] > PI)
		{
			LOG(ERROR) << COLOR(red) << "Wrong angle value" << std::endl << COLOR(none);
			exit(1);
		}
                convertedObjective[i] = radius * sin(acos(objective[i] / radius) + angle[i]) / sin(angle[i]);
        }
}
template <typename TIndividual, typename TRandom>
void Ccdas<TIndividual, TRandom>::makeOneGeneration(void)
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
template <typename TPtr, typename TIter> TIter Ccdas<TIndividual, TRandom>::selectNoncrit(const std::list<TPtr> &front, TIter begin, TIter end)
{
        std::vector<TPtr> iFront(front.begin(), front.end());
        easea::shared::functions::setCrowdingDistance<TO>(iFront.begin(), iFront.end());
        if (iFront.size() > std::distance(begin, end))
	{
		LOG(ERROR) << COLOR(red) << "Wrong front size" << std::endl << COLOR(none);
		exit(1);
	}	
        TIter dest = begin;
        for (size_t i = 0; i < iFront.size(); ++i, ++dest)
                *dest = *iFront[i];
        return dest;
}

template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Ccdas<TIndividual, TRandom>::selectCrit(const std::list<TPtr> &front, TIter begin, TIter end)
{
        std::vector<TPtr> iFront(front.begin(), front.end());
        easea::shared::functions::setCrowdingDistance<TO>(iFront.begin(), iFront.end());
        std::partial_sort(iFront.begin(), iFront.begin() + std::distance(begin, end), iFront.end(), [](TPtr individual1, TPtr individual2)->bool{return individual1->m_crowdingDistance > individual2->m_crowdingDistance;}
        );
        if (iFront.size() < std::distance(begin, end))
	{
		LOG(ERROR) << COLOR(red) << "Wrong front size" << std::endl << COLOR(none);
		exit(1);
	}		
        TIter dest = begin;
        for (size_t i = 0; dest != end; ++i, ++dest)
                *dest = *iFront[i];
        return dest;
}

template <typename TIndividual, typename TRandom>
const typename Ccdas<TIndividual, TRandom>::TI *Ccdas<TIndividual, TRandom>::comparer(const std::vector<const TI *> &comparator)
{
        if (isDominated(*comparator[0], *comparator[1]))
                return comparator[0];
        else if (isDominated(*comparator[1], *comparator[0]))
                return comparator[1];
        if (comparator[0]->m_crowdingDistance < 0) LOG_FATAL("Wrong crowding distance value");
        if (comparator[1]->m_crowdingDistance < 0) LOG_FATAL("Wrong crowding distance value");

        return comparator[0]->m_crowdingDistance > comparator[1]->m_crowdingDistance ? comparator[0] : comparator[1];
}
}
}
}

 
