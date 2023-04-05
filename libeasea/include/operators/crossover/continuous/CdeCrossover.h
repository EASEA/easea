/***********************************************************************
| CdeCrossover.h 							|
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA		|
| (EAsy Specification of Evolutionary Algorithms) 			|
| https://github.com/EASEA/                                 		|
|    									|	
| Copyright (c)      							|
| ICUBE Strasbourg		                           		|
| Date: 2023-04                                                         |
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
#include <operators/crossover/base/C3x1Crossover.h>


namespace easea
{
namespace operators
{
namespace crossover
{
namespace continuous
{
namespace de
{
template <typename TType, typename TRandom>
class CdeCrossover : public C3x1Crossover<TType, std::vector<TType> >, public easea::shared::CRandom<TRandom>, public easea::shared::CProbability<TType>, public easea::shared::CBoundary<TType>
{
public:
        typedef TType TT;
        typedef TRandom TR;
        typedef std::vector<TT> TVariable;
        typedef C3x1Crossover<TT, TVariable> TBase;
        typedef typename TBase::TI TI;
        typedef typename easea::shared::CBoundary<TT>::TRange TRange;
        typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;

        CdeCrossover(TRandom random, const TT probability, const TBoundary &boundary, /*const TT distributionIndex,*/ const TT scalingFactor = 0.5);
        ~CdeCrossover(void);
        TT getScalingFactor(void) const;
    //    TT getProbability(void) const;

protected:
        void runCrossover(const TI &parent, const TI &parent1, const TI &parent2, const TI &parent3, TI &offspring);
        void launch(const TVariable &parent, const TVariable &parent1, const TVariable &parent2, const TVariable &parent3, TVariable &offspring);

private:
        std::uniform_real_distribution<TT> m_distribution;
        TT m_scalingFactor;
};



template <typename TType, typename TRandom>
CdeCrossover<TType, TRandom>::CdeCrossover(TRandom random, const TT probability, const TBoundary &boundary, const TT scalingFactor)
        : easea::shared::CRandom<TRandom>(random), easea::shared::CProbability<TT>(probability), easea::shared::CBoundary<TT>(boundary)
        , m_scalingFactor(scalingFactor)
{
    //    assert(distributionIndex >= 0);
        assert(0 <= scalingFactor && scalingFactor <= 1);
}

template <typename TType, typename TRandom>
CdeCrossover<TType, TRandom>::~CdeCrossover(void)
{
}

template <typename TType, typename TRandom>
TType CdeCrossover<TType, TRandom>::getScalingFactor(void) const
{
    return m_scalingFactor;
}
template <typename TType, typename TRandom>
void CdeCrossover<TType, TRandom>::runCrossover(const TI &parent, const TI &parent1, const TI &parent2, const TI &parent3, TI &offspring)
{
	launch(parent.m_variable, parent1.m_variable, parent2.m_variable, parent3.m_variable, offspring.m_variable);
}
template <typename TType, typename TRandom>
void CdeCrossover<TType, TRandom>::launch(const TVariable &parent, const TVariable &parent1, const TVariable &parent2, const TVariable &parent3,  TVariable &offspring)
{
	assert(!parent.empty());
        assert(!this->getBoundary().empty());
	assert(parent1.size() == parent.size());
        assert(parent1.size() == parent.size());
        assert(parent2.size() == parent.size());
	std::uniform_int_distribution<size_t> dist(0, this->getBoundary().size() - 1);
	const size_t randIndex = dist(this->getRandom());
	offspring.resize(this->getBoundary().size());
        for (size_t i = 0; i < this->getBoundary().size(); ++i)
        {
                //        const TRange &range = this->getBoundary()[i];
            if (m_distribution(this->getRandom()) < this->getProbability() || i == randIndex)
	    {
		const TRange &range = this->getBoundary()[i];
                offspring[i] =  easea::shared::functions::helper::checkBoundary( parent3[i] + m_scalingFactor * ( parent1[i] - parent2[i] ), range.first, range.second);
            }else
                offspring[i] = parent[i];
	}
}
}
}
}
}
}
