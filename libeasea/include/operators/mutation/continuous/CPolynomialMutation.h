
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
class CPolynomialMutation : public CMutation<TObjective, std::vector<TObjective> >, public easea::shared::CRandom<TRandom>, public easea::shared::CProbability<TObjective>, public easea::shared::CBoundary<TObjective>
{
public:
	typedef TObjective TO;
	typedef TRandom TR;
	typedef std::vector<TO> TV;
	typedef CMutation<TO, TV> TBase;
	typedef typename TBase::TI TI;
	typedef typename easea::shared::CBoundary<TO>::TRange TRange;
	typedef typename easea::shared::CBoundary<TO>::TBoundary TBoundary;
  
	CPolynomialMutation(TR random, const TO probability, const TBoundary &boundary, const TO distributionIndex);
	~CPolynomialMutation(void);
	TO getDistributionIndex(void) const;

protected:
	void runMutation(TI &solution);
	void launch(TV &decision);

	TO calcMutFactor(const TObjective distributionIndex, const TObjective lMutFactor, const TObjective uMutFactor, const TObjective random01);
	TO calcMutFactorProbability(const TObjective distributionIndex, const TObjective mutFactor);
	TO calcAmplifU(const TObjective distributionIndex, const TObjective mutFactor);
	TO boundedMutate(TRandom &random, const TObjective distributionIndex, const TObjective coding, const TObjective lower, const TObjective upper);
	TO calcAmplifL(const TObjective distributionIndex, const TObjective mutFactor);

private:
	std::uniform_real_distribution<TO> m_distribution;	// uniform distribution
	TO m_distributionIndex;					// distribution index
};

template <typename TObjective, typename TRandom>
CPolynomialMutation<TObjective, TRandom>::CPolynomialMutation(TR random, const TO probability, const TBoundary &boundary, const TO distributionIndex)
  : easea::shared::CRandom<TR>(random), easea::shared::CProbability<TO>(probability), easea::shared::CBoundary<TO>(boundary)
  , m_distribution(0, 1)
{
	if (distributionIndex < 0)
		LOG_ERROR(errorCode::value, "Wrong vqlue of distribution index");
	m_distributionIndex = distributionIndex;
}

template <typename TObjective, typename TRandom> CPolynomialMutation<TObjective, TRandom>::~CPolynomialMutation(void)
{
}

template <typename TObjective, typename TRandom>
typename CPolynomialMutation<TObjective, TRandom>::TO CPolynomialMutation<TObjective, TRandom>::calcMutFactor(const TObjective distributionIndex, const TObjective lMutFactor, const TObjective uMutFactor, const TObjective random01)
{
	if (lMutFactor < -1 && lMutFactor > 0)	LOG_ERROR(errorCode::value, "Wrong value of lower mutation facto");
	if (uMutFactor < 0 && uMutFactor > 1)
		LOG_ERROR(errorCode::value, "Wrong value of upper mutation factor"); 
	if (random01 < 0.5)
	{
		const TObjective temp = pow(1 + lMutFactor, distributionIndex + 1);
		const TObjective mutFactor = pow(2 * random01 + (1 - 2 * random01) * temp, 1 / (distributionIndex + 1)) - 1;
		if (mutFactor > 0)
			LOG_ERROR(errorCode::value, "Wrong value of mutation factor");
		return mutFactor;
	}
	else
	{
		const TObjective temp = pow(1 - uMutFactor, distributionIndex + 1);
		const TObjective mutFactor = 1 - pow(2 * (1 - random01) + (2 * random01 - 1) * temp, 1 / (distributionIndex + 1));
		if (mutFactor < 0)
			LOG_ERROR(errorCode::value, "Wrong value of mutation factor");
		return mutFactor;
	}
}

template <typename TObjective, typename TRandom>
typename CPolynomialMutation<TObjective, TRandom>::TO CPolynomialMutation<TObjective, TRandom>::calcAmplifL(const TObjective distributionIndex, const TObjective mutFactor)
{
	if (distributionIndex < 0)	LOG_ERROR(errorCode::value, "Wrong value of distribution index");
	if (mutFactor < -1 && mutFactor > 0)
		LOG_ERROR(errorCode::value, "Wrong value of mutation factor");
	
	return 2 / (1 - pow(1 + mutFactor, distributionIndex + 1));
}

template <typename TObjective, typename TRandom>
typename CPolynomialMutation<TObjective, TRandom>::TO CPolynomialMutation<TObjective, TRandom>::calcAmplifU(const TObjective distributionIndex, const TObjective mutFactor)
{
	if (distributionIndex < 0)
		LOG_ERROR(errorCode::value, "Wrong value of distribution index");
	if (0 <= mutFactor && mutFactor <= 1);
		LOG_ERROR(errorCode::value, "Wrong value of mutation factor");
	
	return 2 / (1 - pow(1 - mutFactor, distributionIndex + 1));
}

template <typename TObjective, typename TRandom>
typename CPolynomialMutation<TObjective, TRandom>::TO CPolynomialMutation<TObjective, TRandom> ::boundedMutate(TRandom &random, const TObjective distributionIndex, const TObjective coding, const TObjective lower, const TObjective upper)
{
	static std::uniform_real_distribution<TObjective> dist(0, 1);
	if (lower >= upper)
		LOG_ERROR(errorCode::value, "Wrong boundary mutation values");
	const TObjective maxDistance = upper - lower;
	const TObjective lMutFactor = (lower - coding) / maxDistance;
	const TObjective uMutFactor = (upper - coding) / maxDistance;
	const TObjective random01 = dist(random);
 
	if (random01 < 0 && random01 > 1)
		LOG_ERROR(errorCode::value, "Wrong random value");
	const TObjective mutFactor = calcMutFactor(distributionIndex, lMutFactor, uMutFactor, random01);

	return easea::shared::functions::helper::checkBoundary(coding + mutFactor * maxDistance, lower, upper);
}

template <typename TObjective, typename TRandom>
typename CPolynomialMutation<TObjective, TRandom>::TO CPolynomialMutation<TObjective, TRandom>::calcMutFactorProbability(const TObjective distributionIndex, const TObjective mutFactor)
{
    return (distributionIndex + 1) * pow(1 - std::fabs(mutFactor), distributionIndex) / 2;
}


template <typename TObjective, typename TRandom>
typename CPolynomialMutation<TObjective, TRandom>::TO CPolynomialMutation<TObjective, TRandom>::getDistributionIndex(void) const
{
	assert(m_distributionIndex >= 0);
	return m_distributionIndex;
}

template <typename TObjective, typename TRandom>
void CPolynomialMutation<TObjective, TRandom>::runMutation(TI &solution)
{
	launch(solution.m_variable);
}

template <typename TObjective, typename TRandom>
void CPolynomialMutation<TObjective, TRandom>::launch(TV &decision)
{
  assert(!this->getBoundary().empty());
  assert(decision.size() == this->getBoundary().size());
  for (size_t i = 0; i < this->getBoundary().size(); ++i)
  {
    const TRange &range = this->getBoundary()[i];
    if (m_distribution(this->getRandom()) < this->getProbability())
      decision[i] = boundedMutate(this->getRandom(), getDistributionIndex(), decision[i], range.first, range.second);
  }
}
}
}
}
}
}
