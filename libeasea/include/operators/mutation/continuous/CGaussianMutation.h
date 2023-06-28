/***********************************************************************
| CGaussianMutation.h                                                   |
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


#include <cmath>
#include <random>
#include <cassert>
#include <shared/CRandom.h>
#include <shared/CBoundary.h>
#include <shared/CProbability.h>
#include <shared/functions/helper.h>
#include <operators/mutation/base/CMutation.h>
#include <CLogger.h>

namespace easea
{
namespace operators
{
namespace mutation
{
namespace continuous
{
namespace pm
{
template <typename TObjective, typename TRandom>
class CGaussianMutation : public CMutation<TObjective, std::vector<TObjective> >, public easea::shared::CRandom<TRandom>, public easea::shared::CProbability<TObjective>, public easea::shared::CBoundary<TObjective>
{
public:
	typedef TObjective TO;
	typedef TRandom TR;
	typedef std::vector<TO> TV;
	typedef CMutation<TO, TV> TBase;
	typedef typename TBase::TI TI;
	typedef typename easea::shared::CBoundary<TO>::TRange TRange;
	typedef typename easea::shared::CBoundary<TO>::TBoundary TBoundary;
  
	CGaussianMutation(TR random, const TO probability, const TBoundary &boundary, const TO distributionIndex);
	~CGaussianMutation(void);
	TO getDistributionIndex(void) const;

protected:
	void runMutation(TI &decision);
	void launch(TV &decision);

	TO getGaussian(TRandom &random);
	TO nextGaussian(TRandom &random, TO mean, TO deviation);
	TO boundedMutate(TRandom &random, const TObjective distributionIndex, const TObjective idecision, const TObjective lower, const TObjective upper);

private:
	std::uniform_real_distribution<TO> m_distribution;	// uniform distribution
	TO m_distributionIndex;					// distribution index
	TO m_tau;
};

template <typename TObjective, typename TRandom>
CGaussianMutation<TObjective, TRandom>::CGaussianMutation(TR random, const TO probability, const TBoundary &boundary, const TO distributionIndex)
  : easea::shared::CRandom<TR>(random), easea::shared::CProbability<TO>(probability), easea::shared::CBoundary<TO>(boundary)
  , m_distribution(0, 1)
{
	if (distributionIndex < 0)
		LOG_ERROR(errorCode::value, "Wrong vqlue of distribution index");
	m_distributionIndex = distributionIndex;
	m_tau = 0.;
}
template <typename TObjective, typename TRandom> CGaussianMutation<TObjective, TRandom>::~CGaussianMutation(void)
{
}

template <typename TObjective, typename TRandom>
typename CGaussianMutation<TObjective, TRandom>::TO CGaussianMutation<TObjective,TRandom>::getGaussian(TRandom &random)
{
	TO v1, v2, s;
	static std::uniform_real_distribution<TObjective> dist(0, 1);

	do
	{
		v1 = 2.0 * dist(random) - 1.0;
		v2 = 2.0 * dist(random) - 1.0;

		s = v1 * v1 + v2 * v2;
	}while( s >= 1.0 || s == 0 );
	s = std::sqrt((-2.0 * std::log(s))/s);
	
	return v1 * s; 
    
}



template <typename TObjective, typename TRandom>
typename CGaussianMutation<TObjective, TRandom>::TO CGaussianMutation<TObjective, TRandom>::nextGaussian(TRandom &random, TO mean, TO deviation)
{
	return mean + getGaussian(random) * deviation;
}

template <typename TObjective, typename TRandom>
typename CGaussianMutation<TObjective, TRandom>::TO CGaussianMutation<TObjective, TRandom> ::boundedMutate(TRandom &random, [[maybe_unused]] const TObjective distributionIndex, const TObjective idecision, const TObjective lower, const TObjective upper)
{
	static std::uniform_real_distribution<TObjective> dist(0, 1);
	TO sigma = m_tau * 2.1;
	if (lower >= upper)
		LOG_ERROR(errorCode::value, "Wrong boundary mutation values");
	TO delta = nextGaussian(random, 0.0, sigma);
	return easea::shared::functions::helper::checkBoundary(idecision + delta, lower, upper);
}


template <typename TObjective, typename TRandom>
typename CGaussianMutation<TObjective, TRandom>::TO CGaussianMutation<TObjective, TRandom>::getDistributionIndex(void) const
{
	assert(m_distributionIndex >= 0);
	return m_distributionIndex;
}

template <typename TObjective, typename TRandom>
void CGaussianMutation<TObjective, TRandom>::runMutation(TI &decision)
{
	launch(decision.m_variable);
}

template <typename TObjective, typename TRandom>
void CGaussianMutation<TObjective, TRandom>::launch(TV &decision)
{
	assert(!this->getBoundary().empty());
	assert(decision.size() == this->getBoundary().size());
  

	m_tau = 1./(TO)decision.size();
	for (size_t i = 0; i < this->getBoundary().size(); ++i)
	{
		const TRange &range = this->getBoundary()[i];
		if (m_distribution(this->getRandom()) < this->getProbability())
    			decision[i] = boundedMutate(this->getRandom(), getDistributionIndex(), decision[i], range.first, range.second);
	}
}

// reduce compilation time and check for errors while compiling lib
extern template class CGaussianMutation<float, DefaultGenerator_t>;
extern template class CGaussianMutation<double, DefaultGenerator_t>;
}
}
}
}
}
