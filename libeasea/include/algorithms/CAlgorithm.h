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
	bool checkIsLast();
	void setLimitGeneration(size_t nbGen);
	void setCurrentGeneration(size_t iGen);
	size_t getLimitGeneration();
	size_t getCurrentGeneration();
	virtual void on_individuals_received() {}



protected:
        virtual void makeOneGeneration(void) = 0;
	virtual void initialize() = 0;

private:
        TP &m_problem;
	size_t m_nbGen;
	size_t m_iGen;
};

template <typename TObjective, typename TVariable>
CAlgorithm<TObjective, TVariable>::CAlgorithm(TP &problem) : m_problem(problem), m_nbGen(0), m_iGen(0)
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
bool CAlgorithm<TObjective, TVariable>::checkIsLast()
{
	return m_iGen + 1 >= m_nbGen;
}

template <typename TObjective, typename TVariable>
void CAlgorithm<TObjective, TVariable>::setLimitGeneration(size_t nbGen)
{
	m_nbGen = nbGen;
}

template <typename TObjective, typename TVariable>
void CAlgorithm<TObjective, TVariable>::setCurrentGeneration(size_t iGen)
{
	m_iGen = iGen;
}
template <typename TObjective, typename TVariable>
size_t CAlgorithm<TObjective, TVariable>::getLimitGeneration()
{
	return m_nbGen;
}

template <typename TObjective, typename TVariable>
size_t CAlgorithm<TObjective, TVariable>::getCurrentGeneration()
{
	return m_iGen;
}



template <typename TObjective, typename TVariable>
void CAlgorithm<TObjective, TVariable>::run()
{
	if (m_iGen == 0) {
		initialize();
        } else {
		makeOneGeneration();
	}
	m_iGen++;
}

// speedup compîlation of .ez
extern template class CAlgorithm<double, std::vector<double>>;
}
}

