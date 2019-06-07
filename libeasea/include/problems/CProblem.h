/***********************************************************************
| CProblem.h                                                            |
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
#include <list>
#include <string>
#include <algorithm>
#include <cctype>
#include <third_party/aixlog/aixlog.hpp>
#include <core/CmoIndividual.h>
#include <shared/CBoundary.h>


namespace easea
{
namespace problem
{
/*
 * \brief Base class of optimization problem
 * \param[in] TType - Ttype of decision variable
 */

template <typename TType>
class CProblem : public easea::shared::CBoundary<TType>
{
public:
        typedef TType TO;
        typedef std::vector<TType> TV;
        typedef CmoIndividual<TO, TV> TI;
	typedef typename easea::shared::CBoundary<TType>::TRange TRange;
        typedef typename easea::shared::CBoundary<TType>::TBoundary TBoundary;


        CProblem(const size_t nObjectives, const size_t nVariables, const TBoundary &boundary);
        virtual ~CProblem(void);
        size_t getNumberOfObjectives(void) const;
        size_t getNumberOfVariables(void) const;
        size_t getNumberOfEvaluations(void) const;

        void operator ()(TI &individual);

private:
        size_t m_nObjectives;   // The number of objectives
	size_t m_nVariables;	// The number of disicion variables
        size_t m_nEvaluations;  // The number of evaluations
};

/*
 * Class constructor
 * \param[in] nObjectives Number of objectives
 * \param[in] nVariables  Number of variables
 * \param[in] boundary    List of up and low limit values of dicision variables
 */
template <typename TType>
CProblem<TType>::CProblem(const size_t nObjectives, const size_t nVariables, const TBoundary &boundary) : easea::shared::CBoundary<TO>(boundary)
{
        if (nObjectives < 0){
		LOG(ERROR) << COLOR(red) << "Wrong number of objectives: " << nObjectives << std::endl << COLOR(none);
    		exit(-1);
	}
	m_nObjectives = nObjectives;
	
	if (nVariables < 1){
		LOG(ERROR) << COLOR(red) << "Wrong number of variables: " << nVariables << std::endl << COLOR(none);
	}
	m_nVariables = nObjectives - 1 + nVariables;

        m_nEvaluations = 0;
}

/*
 * Class destructor  
 */

template <typename TType>
CProblem<TType>::~CProblem(void)
{
}

/*
 * \brief Get number of objectives
 * \return - Number of objectives
 */
template <typename TType>
size_t CProblem<TType>::getNumberOfObjectives(void) const
{
        return m_nObjectives;
}

/*
 * \brief Get number of variables
 * \return - Number of variables
 */
template <typename TType>
size_t CProblem<TType>::getNumberOfVariables(void) const
{
        return m_nVariables;
}


/*
 * \brief Get number of evaluations
 * \return - Number of evaluations
 */
template <typename TType>
size_t CProblem<TType>::getNumberOfEvaluations(void) const
{
        return m_nEvaluations;
}

/*
 * \brief Evaluate a solution
 * \param[in, out] - Solution to evaluate/ evaluated solution
 */
template <typename TType>
void CProblem<TType>::operator ()(TI &individual)
{	
        if (!this->isInside(individual.m_variable)){
		LOG(ERROR) << COLOR(red) << "Value of variable is out of bounds " << std::endl << COLOR(none);
		exit(-1);
	}
	const size_t i_nEvaluation = individual.evaluate(); 

	if (i_nEvaluation < 0){
		LOG(ERROR) << COLOR(red) << "Number of evaluations < 0 " << std::endl << COLOR(none);
		exit(-1);
	}
        m_nEvaluations += i_nEvaluation;
}
}
}
