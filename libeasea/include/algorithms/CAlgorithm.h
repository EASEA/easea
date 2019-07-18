/***********************************************************************
| CAlgorithm.h                                                          |
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
#include <cctype>
#include <algorithm>
#include <string>
#include <problems/CProblem.h>

namespace easea
{
namespace algorithms
{
template <typename TObjective, typename TVariable>
class CAlgorithm
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef typename easea::shared::CBoundary<TO>::TBoundary TBoundary;

        typedef easea::problem::CProblem<TO> TP;

        CAlgorithm(TP &problem);
        virtual ~CAlgorithm(void);
        TP &getProblem() const;
        void run();

protected:
        virtual void makeOneGeneration(void) = 0;

private:
        TP &m_problem;
};

template <typename TObjective, typename TVariable>
CAlgorithm<TObjective, TVariable>::CAlgorithm(TP &problem) : m_problem(problem)
{
}

template <typename TObjective, typename TVariable>
CAlgorithm<TObjective, TVariable>::~CAlgorithm(void)
{
}

template <typename TObjective, typename TVariable>
typename CAlgorithm<TObjective, TVariable>::TP &CAlgorithm<TObjective, TVariable>::getProblem() const
{
        return m_problem;
}

template <typename TObjective, typename TVariable>
void CAlgorithm<TObjective, TVariable>::run()
{
        makeOneGeneration();
}
}
}

