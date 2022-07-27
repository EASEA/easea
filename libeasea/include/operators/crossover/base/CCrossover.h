/***********************************************************************
| CCrossover.h                                                          |
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
#include <core/CmoIndividual.h>

namespace easea
{
namespace operators
{
namespace crossover
{
template <typename TObjective, typename TVariable>
class CCrossover
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CmoIndividual<TO, TV> TI;

        CCrossover(void);
        virtual ~CCrossover(void);
        void operator ()(std::vector<const TI *> &parent, std::vector<TI *> &offspring);
	void setLimitGen(const std::size_t nbGen);
	void setCurrentGen(const std::size_t iGen);
	std::size_t getLimitGen();
	std::size_t getCurrentGen();
protected:
        virtual void runCrossover(std::vector<const TI *> &parent, std::vector<TI *> &offspring) = 0;
	std::size_t m_nbGen;
	std::size_t m_iGen;
};

template <typename TObjective, typename TVariable>
CCrossover<TObjective, TVariable>::CCrossover(void)
{
	m_nbGen = 0;
	m_iGen = 0;
}


template <typename TObjective, typename TVariable>
CCrossover<TObjective, TVariable>::~CCrossover(void)
{
}

template <typename TObjective, typename TVariable>
void CCrossover<TObjective, TVariable>::operator ()(std::vector<const TI *> &parent, std::vector<TI *> &offspring)
{

        runCrossover(parent, offspring);
}
template <typename TObjective, typename TVariable>
void CCrossover<TObjective, TVariable>::setLimitGen(const std::size_t nbGen)
{
	m_nbGen = nbGen;
}
template <typename TObjective, typename TVariable>
std::size_t CCrossover<TObjective, TVariable>::getLimitGen()
{
	return m_nbGen;
}
template <typename TObjective, typename TVariable>
void CCrossover<TObjective, TVariable>::setCurrentGen(const std::size_t iGen)
{
	m_iGen = iGen;
}
template <typename TObjective, typename TVariable>
std::size_t CCrossover<TObjective, TVariable>::getCurrentGen()
{
	return m_iGen;
}

}
}
}
