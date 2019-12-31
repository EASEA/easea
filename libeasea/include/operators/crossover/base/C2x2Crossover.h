/***********************************************************************
| C2x2Crossover.h                                                       |
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
class C2x2Crossover
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CmoIndividual<TO, TV> TI;

        C2x2Crossover(void);
        virtual ~C2x2Crossover(void);
        void operator ()(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2);
	void setLimitGen(size_t nbGen);
	void setCurrentGen(size_t iGen);
	size_t getLimitGen();
	size_t getCurrentGen();
protected:
        virtual void runCrossover(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2) = 0;
	size_t m_nbGen;
	size_t m_iGen;
};


template <typename TObjective, typename TVariable>
C2x2Crossover<TObjective, TVariable>::C2x2Crossover(void)
{
	m_nbGen;
	m_iGen;
}

template <typename TObjective, typename TVariable>
C2x2Crossover<TObjective, TVariable>::~C2x2Crossover(void)
{
}

template <typename TObjective, typename TVariable>
void C2x2Crossover<TObjective, TVariable>::setLimitGen(size_t nbGen)
{
	m_nbGen = nbGen;
}
template <typename TObjective, typename TVariable>
size_t C2x2Crossover<TObjective, TVariable>::getLimitGen()
{
        return m_nbGen;
}

template <typename TObjective, typename TVariable>
void C2x2Crossover<TObjective, TVariable>::setCurrentGen(size_t iGen)
{
	m_iGen = iGen;
}
template <typename TObjective, typename TVariable>
size_t C2x2Crossover<TObjective, TVariable>::getCurrentGen()
{
         return m_iGen;
}

template <typename TObjective, typename TVariable>
void C2x2Crossover<TObjective, TVariable>::operator ()(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2)
{	
        runCrossover(parent1, parent2, offspring1, offspring2);
}
}
}
}