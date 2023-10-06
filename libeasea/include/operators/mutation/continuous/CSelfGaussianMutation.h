/***********************************************************************
| CSelfGaussianMutation.h                                               |
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
class CSelfGaussianMutation : public CMutation<TObjective, std::vector<TObjective> >, public easea::shared::CRandom<TRandom>, public easea::shared::CProbability<TObjective>, public easea::shared::CBoundary<TObjective>
{
public:
	typedef TObjective TO;
	using TProba = std::conditional_t<sizeof(TO) <= 4, float, double>;
	typedef TRandom TR;
	typedef std::vector<TO> TV;
	typedef CMutation<TO, TV> TBase;
	typedef typename TBase::TI TI;
	typedef typename easea::shared::CBoundary<TO>::TRange TRange;
	typedef typename easea::shared::CBoundary<TO>::TBoundary TBoundary;
  
	CSelfGaussianMutation(TR random, const TO probability, const TBoundary &boundary, const TO distributionIndex);
	~CSelfGaussianMutation(void);
	TO getDistributionIndex(void) const;
	

protected:
	void runMutation(TI &decision);
	void launch(TV &variables, TV &step);

	TO getGaussian(TRandom &random);
	TO valueMutate(TRandom &random, TO step);
	TO stepMutate(TO step, const TO u);
size_t getCurrentGen(){ return TBase::getCurrentGen();};
size_t getLimitGen(){ return TBase::getLimitGen();};

	TO nextGaussian(TRandom &random, TO mean, TO deviation);
	TO boundedMutate(TRandom &random, const TObjective distributionIndex, const TObjective idecision, const TObjective u, TObjective &istep, const TObjective lower, const TObjective upper);

private:
	std::uniform_real_distribution<TProba> m_distribution;	// uniform distribution
	TO m_distributionIndex;					// distribution index
	TO m_tau;
	TO m_tau_1;
};

template <typename TObjective, typename TRandom>
CSelfGaussianMutation<TObjective, TRandom>::CSelfGaussianMutation(TR random, const TO probability, const TBoundary &boundary, const TO distributionIndex)
  : easea::shared::CRandom<TR>(random), easea::shared::CProbability<TO>(probability), easea::shared::CBoundary<TO>(boundary)
  , m_distribution(0, 1)
{
	if (distributionIndex < 0)
		LOG_ERROR(errorCode::value, "Wrong vqlue of distribution index");
	m_distributionIndex = distributionIndex;
	m_tau = 0.;
	m_tau_1 = 0;
}
template <typename TObjective, typename TRandom> CSelfGaussianMutation<TObjective, TRandom>::~CSelfGaussianMutation(void)
{
}

template <typename TObjective, typename TRandom>
typename CSelfGaussianMutation<TObjective, TRandom>::TO CSelfGaussianMutation<TObjective,TRandom>::getGaussian(TRandom &random)
{
	TO v1, v2, s;
	static std::uniform_real_distribution<TProba> dist(0, 1);

	do
	{
		v1 = 2.0 * dist(random) - 1.0;
		v2 = 2.0 * dist(random) - 1.0;

		s = v1 * v1 + v2 * v2;
	}while( s >= 1.0 || s == 0 );
	s = sqrt((-2.0 * log(s))/s);
	
	return v1 * s; 
    
}



template <typename TObjective, typename TRandom>
typename CSelfGaussianMutation<TObjective, TRandom>::TO CSelfGaussianMutation<TObjective, TRandom>::stepMutate(TO step, const TO u)
{
	return  step * exp(m_tau * u);
	
}

template <typename TObjective, typename TRandom>
typename CSelfGaussianMutation<TObjective, TRandom>::TO CSelfGaussianMutation<TObjective, TRandom>::valueMutate(TRandom &random, TO step)
{
	return getGaussian(random) * step;
}

template <typename TObjective, typename TRandom>
typename CSelfGaussianMutation<TObjective, TRandom>::TO CSelfGaussianMutation<TObjective, TRandom> ::boundedMutate(TRandom &random, const TObjective distributionIndex, const TObjective idecision, const TObjective u, TObjective &istep, const TObjective lower, const TObjective upper)
{
	(void)distributionIndex; // unused
	static std::uniform_real_distribution<TProba> dist(0, 1);
//	TO sigma = m_tau * 2.1;
	if (lower >= upper)
		LOG_ERROR(errorCode::value, "Wrong boundary mutation values");
	istep = stepMutate(istep, u);
	TO delta = valueMutate(random, istep);
//	TO delta = nextGaussian(random, 0.0, sigma);
	return easea::shared::functions::helper::checkBoundary(idecision + delta, lower, upper);
}


template <typename TObjective, typename TRandom>
typename CSelfGaussianMutation<TObjective, TRandom>::TO CSelfGaussianMutation<TObjective, TRandom>::getDistributionIndex(void) const
{
	assert(m_distributionIndex >= 0);
	return m_distributionIndex;
}

template <typename TObjective, typename TRandom>
void CSelfGaussianMutation<TObjective, TRandom>::runMutation(TI &decision)
{
	launch(decision.m_variable, decision.m_mutStep);
}

template <typename TObjective, typename TRandom>
void CSelfGaussianMutation<TObjective, TRandom>::launch(TV &variables, TV &step)
{
	assert(!this->getBoundary().empty());
	assert(variables.size() == this->getBoundary().size());

	m_tau_1 = (log((TO)variables.size())/((TO)variables.size())); /* 1./sqrt(2*(TO)variables.size());*/
	const TO u = getGaussian(this->getRandom());
	TO k =1.;
	for (size_t i = 0; i < this->getBoundary().size(); ++i)
	{
		const TRange &range = this->getBoundary()[i];

		if (m_distribution(this->getRandom()) < this->getProbability()){
k = 1*fabs(variables[i]);
if (k < 1.) k = 1;
//if (this->getCurrentGen() > this->getLimitGen()/2.)
//if (m_distribution(this->getRandom()) <0.5)
m_tau = k*log((TO)variables.size())/((TO)variables.size());
//else
// m_tau = m_tau_1;
    			variables[i] = boundedMutate(this->getRandom(), getDistributionIndex(), variables[i], u, step[i], range.first, range.second);
		}
	}
}

// reduce compilation time and check for errors while compiling lib
extern template class CSelfGaussianMutation<float, DefaultGenerator_t>;
extern template class CSelfGaussianMutation<double, DefaultGenerator_t>;
}
}
}
}
}
