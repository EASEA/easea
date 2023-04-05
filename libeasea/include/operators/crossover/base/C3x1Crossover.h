/***********************************************************************
| C3x1Crossover.h                                                          |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2023-04                                                         |
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
class C3x1Crossover
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CmoIndividual<TO, TV> TI;

        C3x1Crossover(void);
        virtual ~C3x1Crossover(void);
        void operator ()(const TI &parent, const TI &parent1, const TI &parent2, const TI &parent3, TI &offspring);
	void setLimitGen(std::size_t nbGen);
	void setCurrentGen(std::size_t iGen);
	std::size_t getLimitGen();
	std::size_t getCurrentGen();
protected:
        virtual void runCrossover(const TI &parent, const TI &parent1, const TI &parent2, const TI &parent3, TI &offspring) = 0;
	std::size_t m_nbGen;
	std::size_t m_iGen;
};


template <typename TObjective, typename TVariable>
C3x1Crossover<TObjective, TVariable>::C3x1Crossover(void)
{
	m_nbGen;
	m_iGen;
}

template <typename TObjective, typename TVariable>
C3x1Crossover<TObjective, TVariable>::~C3x1Crossover(void)
{
}

template <typename TObjective, typename TVariable>
void C3x1Crossover<TObjective, TVariable>::setLimitGen(std::size_t nbGen)
{
	m_nbGen = nbGen;
}
template <typename TObjective, typename TVariable>
std::size_t C3x1Crossover<TObjective, TVariable>::getLimitGen()
{
        return m_nbGen;
}

template <typename TObjective, typename TVariable>
void C3x1Crossover<TObjective, TVariable>::setCurrentGen(std::size_t iGen)
{
	m_iGen = iGen;
}
template <typename TObjective, typename TVariable>
std::size_t C3x1Crossover<TObjective, TVariable>::getCurrentGen()
{
         return m_iGen;
}

template <typename TObjective, typename TVariable>
void C3x1Crossover<TObjective, TVariable>::operator ()(const TI &parent, const TI &parent1, const TI &parent2, const TI &parent3, TI &offspring)
{	
        runCrossover(parent,  parent1, parent2, parent3, offspring);
}
}
}
}
