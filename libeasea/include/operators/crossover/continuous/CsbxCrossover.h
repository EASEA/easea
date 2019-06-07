/***********************************************************************
| CsbxIndividual.h 							|
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA		|
| (EAsy Specification of Evolutionary Algorithms) 			|
| https://github.com/EASEA/                                 		|
|    									|	
| Copyright (c)      							|
| ICUBE Strasbourg		                           		|
| Date: 2019-05                                                         |
|                                                                       |
 ***********************************************************************/


#pragma once

#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <shared/CRandom.h>
#include <shared/CProbability.h>
#include <shared/CBoundary.h>
#include <shared/functions/helper.h>
#include <operators/crossover/base/C2x2Crossover.h>


namespace easea
{
namespace operators
{
namespace crossover
{
namespace continuous
{
namespace sbx
{
template <typename TType, typename TRandom>
class CsbxCrossover : public C2x2Crossover<TType, std::vector<TType> >, public easea::shared::CRandom<TRandom>, public easea::shared::CProbability<TType>, public easea::shared::CBoundary<TType>
{
public:
        typedef TType TT;
        typedef TRandom TR;
        typedef std::vector<TT> TVariable;
        typedef C2x2Crossover<TT, TVariable> TBase;
        typedef typename TBase::TI TI;
        typedef typename easea::shared::CBoundary<TT>::TRange TRange;
        typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;

        CsbxCrossover(TRandom random, const TT probability, const TBoundary &boundary, const TT distributionIndex, const TT componentProbability = 0.5);
        ~CsbxCrossover(void);
        TT getDistributionIndex(void) const;
        TT getProbability(void) const;

protected:

	TT calculateSpreadFactorAttenuation(const TT distributionIndex, const TT spreadFactor);
	TT calculateSpreadFactor(const TT distributionIndex, const TT spreadFactorAttenuation, const TT random01);
	void boundedCrossover(TR &random, const TT distributionIndex, const TT parent1, const TT parent2, TT &offspring1, TT &offspring2, const TT lower, const TT upper);

        void runCrossover(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2);
        void launch(const TVariable &parent1, const TVariable &parent2, TVariable &offspring1, TVariable &offspring2);

private:
        std::uniform_real_distribution<TT> m_distribution;
        TT m_distributionributionIndex;
        TT m_probability;
};



template <typename TType, typename TRandom>
CsbxCrossover<TType, TRandom>::CsbxCrossover(TRandom random, const TT probability, const TBoundary &boundary, const TT distributionIndex, const TT componentProbability)
        : easea::shared::CRandom<TRandom>(random), easea::shared::CProbability<TT>(probability), easea::shared::CBoundary<TT>(boundary)
        , m_distribution(0, 1), m_distributionributionIndex(distributionIndex), m_probability(componentProbability)
{
        assert(distributionIndex >= 0);
        assert(0 <= componentProbability && componentProbability <= 1);
}

template <typename TType, typename TRandom>
CsbxCrossover<TType, TRandom>::~CsbxCrossover(void)
{
}

template <typename TType, typename TRandom>
typename CsbxCrossover<TType, TRandom>::TT CsbxCrossover<TType, TRandom>::calculateSpreadFactorAttenuation(const TT distributionIndex, const TT spreadFactor)
{
        assert(distributionIndex >= 0);
        assert(spreadFactor >= 0);
        return 2 - pow(spreadFactor, -(distributionIndex + 1));
}
template <typename TType, typename TRandom>
typename CsbxCrossover<TType, TRandom>::TT CsbxCrossover<TType, TRandom>::calculateSpreadFactor(const TT distributionIndex, const TT spreadFactorAttenuation, const TT random01)
{
        assert(0 <= random01 && random01 < 1);
        if (random01 < 1. / spreadFactorAttenuation)
                return pow(random01 * spreadFactorAttenuation, 1. / (distributionIndex + 1));
        else
                return pow(1. / (2 - random01 * spreadFactorAttenuation), 1 / (distributionIndex + 1));
}
template <typename TType, typename TRandom> 
void CsbxCrossover<TType, TRandom>::boundedCrossover(TRandom &random, const TT distributionIndex, const TT parent1, const TT parent2, TT &offspring1, TT &offspring2, const TT lower, const TT upper)
{
        static std::uniform_real_distribution<TT> dist(0, 1);
        assert(lower < upper);
        const TT distance = std::fabs(parent1 - parent2);
        if (distance == 0)
        {
                offspring1 = parent1;
                offspring2 = parent2;
                return;
        }
        const TT spreadFactorLower = 1 + 2 * (std::min(parent1, parent2) - lower) / distance;
        const TT spreadFactorUpper = 1 + 2 * (upper - std::max(parent1, parent2)) / distance;
        assert(spreadFactorLower >= 0);
        assert(spreadFactorUpper >= 0);
        const TT spreadFactorAttenuationLower = calculateSpreadFactorAttenuation(distributionIndex, spreadFactorLower);
        const TT spreadFactorAttenuationUpper = calculateSpreadFactorAttenuation(distributionIndex, spreadFactorUpper);
        const TT random01 = dist(random);
        assert(0 <= random01 && random01 < 1);
        const TT spreadFactor1 = calculateSpreadFactor(distributionIndex, spreadFactorAttenuationLower, random01);
        const TT spreadFactor2 = calculateSpreadFactor(distributionIndex, spreadFactorAttenuationUpper, random01);
        const TT middle = (parent1 + parent2) / 2;
        const TT halfDistance = distance / 2;
        offspring1 = easea::shared::functions::helper::checkBoundary(middle - spreadFactor1 * halfDistance, lower, upper);
        offspring2 = easea::shared::functions::helper::checkBoundary(middle + spreadFactor2 * halfDistance, lower, upper);
        if (dist(random) < 0.5)
                std::swap(offspring1, offspring2);
}


template <typename TType, typename TRandom>
typename CsbxCrossover<TType, TRandom>::TT CsbxCrossover<TType, TRandom>::getDistributionIndex(void) const
{
        return m_distributionributionIndex;
}

template <typename TType, typename TRandom>
TType CsbxCrossover<TType, TRandom>::getProbability(void) const
{
        return m_probability;
}

template <typename TType, typename TRandom>
void CsbxCrossover<TType, TRandom>::runCrossover(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2)
{
	launch(parent1.m_variable, parent2.m_variable, offspring1.m_variable, offspring2.m_variable);
}
        
template <typename TType, typename TRandom>
void CsbxCrossover<TType, TRandom>::launch(const TVariable &parent1, const TVariable &parent2, TVariable &offspring1, TVariable &offspring2)
{
        assert(!this->getBoundary().empty());
        assert(parent1.size() == this->getBoundary().size());
        assert(parent2.size() == this->getBoundary().size());
        if (m_distribution(this->getRandom()) < this->getProbability())
        {
                offspring1.resize(parent1.size());
                offspring2.resize(parent2.size());
                for (size_t i = 0; i < this->getBoundary().size(); ++i)
                {
                        const TRange &range = this->getBoundary()[i];
                        if (m_distribution(this->getRandom()) < getProbability())
                                boundedCrossover(this->getRandom(), this->getDistributionIndex(), parent1[i], parent2[i], offspring1[i], offspring2[i], range.first, range.second);
                        else
                        {
                                offspring1[i] = parent1[i];
                                offspring2[i] = parent2[i];
                        }
                }
        }
        else
        {
                offspring1 = parent1;
                offspring2 = parent2;
        }
}
}
}
}
}
}
