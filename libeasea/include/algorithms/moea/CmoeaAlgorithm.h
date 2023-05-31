/***********************************************************************
| CmoeaAlgorithm.h                                                      |
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

#include <algorithms/CAlgorithm.h>
#include <shared/CRandom.h>
#include <shared/CStatsPrinter.h>
#include <config.h>

#include <random>

namespace easea
{
namespace algorithms
{
template <typename TPopulation, typename TRandom>
class CmoeaAlgorithm : public CAlgorithm<typename TPopulation::value_type::TO, typename TPopulation::value_type::TV>,  public easea::shared::CRandom<TRandom>, public CStatsPrinter<CmoeaAlgorithm<TPopulation, TRandom>>, public CSerializable<CmoeaAlgorithm<TPopulation, TRandom>>
{
public:
        typedef TPopulation TPop;
        typedef typename TPop::value_type TI;
        typedef typename TI::TO TO;
        typedef typename TI::TV TV;
        typedef CAlgorithm<TO, TV> TA;
        typedef typename TA::TP TP;
	typedef TRandom TR;


        CmoeaAlgorithm(TR random, TP &problem, const std::vector<TV> &initial);
	void initialize() override;
        virtual ~CmoeaAlgorithm(void);
        const TPop &getPopulation(void) const;
	TPop& getPopulation();

protected:
        TPop m_population;
};

template <typename TPopulation, typename TRandom>
CmoeaAlgorithm<TPopulation, TRandom>::CmoeaAlgorithm(TR random, TP &problem, const std::vector<TV> &initial) : TA(problem)
											 , easea::shared::CRandom<TR>(random)    
{
        m_population.resize(initial.size());
        static std::uniform_real_distribution<TO> dist(0, 0.99);
	const size_t nb_objs = this->getProblem().getNumberOfObjectives();

#ifdef USE_OPENMP
EASEA_PRAGMA_OMP_PARALLEL
#endif
	for (int i = 0; i < static_cast<int>(initial.size()); ++i)
        {
                TI &individual = m_population[i];
                individual.m_variable = initial[i];
                individual.m_mutStep.resize(individual.m_variable.size());
		individual.m_objective.resize(nb_objs);
                for(size_t j = 0; j < individual.m_variable.size(); j++)
                        individual.m_mutStep[j] = dist(random);
        }
}

template <typename TPopulation, typename TRandom>
void CmoeaAlgorithm<TPopulation, TRandom>::initialize() {
	#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(m_population.size()); ++i)
        {
                TA::getProblem()(m_population[i]);
        }
}

template <typename TPopulation, typename TRandom>
CmoeaAlgorithm<TPopulation, TRandom>::~CmoeaAlgorithm(void)
{
}

template <typename TPopulation, typename TRandom>
const TPopulation &CmoeaAlgorithm<TPopulation, TRandom>::getPopulation(void) const
{
        return m_population;
}

template <typename TPopulation, typename TRandom>
TPopulation &CmoeaAlgorithm<TPopulation, TRandom>::getPopulation()
{
        return m_population;
}

}
}
