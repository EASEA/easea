/***********************************************************************
| C3x1CrossoverLauncher.h                                                          |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
|                                                                       |
 ***********************************************************************/
#pragma once


#include <vector>
#include <random>
#include <algorithm>
#include <shared/CRandom.h>
#include <operators/crossover/base/CCrossover.h>
#include <operators/crossover/wrapper/CWrap3x1Crossover.h>

namespace easea
{
namespace operators
{
namespace crossover
{
template <typename TObjective, typename TVariable, typename TRandom>
class C3x1CrossoverLauncher : public CCrossover<TObjective, TVariable>, public CWrap3x1Crossover<TObjective, TVariable>, public easea::shared::CRandom<TRandom>
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef TRandom TR;
        typedef CCrossover<TO, TV> TBase;
        typedef typename TBase::TI TI;
        typedef typename CWrap3x1Crossover<TO, TV>::TC TC;
        C3x1CrossoverLauncher(TC &crossover, TR random);
        ~C3x1CrossoverLauncher(void);
        void operator ()(std::vector<const TI *> &parent, std::vector<TI *> &offspring);
        void operator ()(const TI &parent, const TI &parent1, const TI &parent2, const TI &parent3, TI &offspring);

protected:
        void runCrossover(std::vector<const TI *> &parent, std::vector<TI *> &offspring);
};

template <typename TObjective, typename TVariable, typename TRandom>
C3x1CrossoverLauncher<TObjective, TVariable, TRandom>::C3x1CrossoverLauncher(TC &crossover, TRandom random)
        : CWrap3x1Crossover<TO, TV>(crossover), easea::shared::CRandom<TRandom>(random)
{
}

template <typename TObjective, typename TVariable, typename TRandom>
C3x1CrossoverLauncher<TObjective, TVariable, TRandom>::~C3x1CrossoverLauncher(void)
{
}
template <typename TObjective, typename TVariable, typename TRandom>
void C3x1CrossoverLauncher<TObjective, TVariable, TRandom>::operator ()(std::vector<const TI *> &parent, std::vector<TI *> &offspring)
{
        TBase::operator ()(parent, offspring);
}

template <typename TObjective, typename TVariable, typename TRandom>
void C3x1CrossoverLauncher<TObjective, TVariable, TRandom>::operator ()(const TI &parent, const TI &parent1, const TI &parent2, const TI &parent3, TI &offspring)
{
        this->getCrossover()(parent, parent1, parent2, parent3, offspring);
}

template <typename TObjective, typename TVariable, typename TRandom>
void C3x1CrossoverLauncher<TObjective, TVariable, TRandom>::runCrossover(std::vector<const TI *> &parent, std::vector<TI *> &offspring)
{
    auto _parent = parent;
    for (size_t child = 0, prnt = 0; child < offspring.size();)
    {
	std::shuffle(_parent.begin(), _parent.end(), this->getRandom());
	for (size_t parent1 = 0; child < offspring.size() && parent1 < parent.size(); ++child, parent1 += 3, prnt = (prnt + 1) % parent.size())
	{
	    const size_t parent2 = (parent1 + 1) % parent.size();
	    const size_t parent3 = (parent2 + 1) % parent.size();
	    (*this)(*parent[prnt], *_parent[parent1], *_parent[parent2], *_parent[parent3], *offspring[child]);
	}
    }
}

// reduce compilation time and check for errors while compiling lib
extern template class C3x1CrossoverLauncher<float, std::vector<float>, DefaultGenerator_t>;
extern template class C3x1CrossoverLauncher<double, std::vector<double>, DefaultGenerator_t>;
}
}
}
