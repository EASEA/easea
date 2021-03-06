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
#include <random>
#include <shared/CRandom.h>
#include <operators/crossover/base/CCrossover.h>
#include <operators/crossover/wrapper/CWrap2x2Crossover.h>

namespace easea
{
namespace operators
{
namespace crossover
{
template <typename TObjective, typename TVariable, typename TRandom>
class C2x2CrossoverLauncher : public CCrossover<TObjective, TVariable>, public CWrap2x2Crossover<TObjective, TVariable>, public easea::shared::CRandom<TRandom>
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef TRandom TR;
        typedef CCrossover<TO, TV> TBase;
        typedef typename TBase::TI TI;
        typedef typename CWrap2x2Crossover<TO, TV>::TC TC;
        C2x2CrossoverLauncher(TC &crossover, TR random);
        ~C2x2CrossoverLauncher(void);
        void operator ()(std::vector<const TI *> &parent, std::vector<TI *> &offspring);
        void operator ()(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2);

protected:
        void runCrossover(std::vector<const TI *> &parent, std::vector<TI *> &offspring);
};

template <typename TObjective, typename TVariable, typename TRandom>
C2x2CrossoverLauncher<TObjective, TVariable, TRandom>::C2x2CrossoverLauncher(TC &crossover, TRandom random)
        : CWrap2x2Crossover<TO, TV>(crossover), easea::shared::CRandom<TRandom>(random)
{
}

template <typename TObjective, typename TVariable, typename TRandom>
C2x2CrossoverLauncher<TObjective, TVariable, TRandom>::~C2x2CrossoverLauncher(void)
{
}
template <typename TObjective, typename TVariable, typename TRandom>
void C2x2CrossoverLauncher<TObjective, TVariable, TRandom>::operator ()(std::vector<const TI *> &parent, std::vector<TI *> &offspring)
{
        TBase::operator ()(parent, offspring);
}

template <typename TObjective, typename TVariable, typename TRandom>
void C2x2CrossoverLauncher<TObjective, TVariable, TRandom>::operator ()(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2)
{
        this->getCrossover()(parent1, parent2, offspring1, offspring2);
}

template <typename TObjective, typename TVariable, typename TRandom>
void C2x2CrossoverLauncher<TObjective, TVariable, TRandom>::runCrossover(std::vector<const TI *> &parent, std::vector<TI *> &offspring)
{
	this->getCrossover().setLimitGen(TBase::getLimitGen());
	this->getCrossover().setCurrentGen(TBase::getCurrentGen());

        for (size_t offspring1 = 0; offspring1 < offspring.size();)
        {
                std::random_shuffle(parent.begin(), parent.end(), [this](const size_t n)-> size_t{return std::uniform_int_distribution<size_t> (0, n - 1)(this->getRandom());});
                for (size_t parent1 = 0; offspring1 < offspring.size() && parent1 < parent.size(); offspring1 += 2, parent1 += 2)
                {
                        const size_t parent2 = (parent1 + 1) % parent.size();
                        const size_t offspring2 = (offspring1 + 1) % offspring.size();
                        (*this)(*parent[parent1], *parent[parent2], *offspring[offspring1], *offspring[offspring2]);
                }
        }
/*	offspring[0]->m_mutStep.resize(parent[0]->m_mutStep.size());
        offspring[1]->m_mutStep.resize(parent[1]->m_mutStep.size());
        for (size_t i = 0; i < parent[0]->m_variable.size(); ++i)
        {
                offspring[0]->m_mutStep[i] = parent[0]->m_mutStep[i];
                offspring[1]->m_mutStep[i] = parent[1]->m_mutStep[i];
        }*/

}
}
}
}
