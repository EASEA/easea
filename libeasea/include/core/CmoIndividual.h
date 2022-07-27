/***********************************************************************
| CmoIndividual.h                                                       |
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
#include <core/CConstraint.h>

namespace easea
{
/*
 * \brief The solution base class
 * \param[in] - TObjective : Type of objective value, must be a real number type
 * \param[in] - TVariable : Type of the decision, can be any data structure
 */
template <typename TObjective, typename TVariable>
class CmoIndividual : public CConstraint<TObjective>
{
public:

        typedef TObjective TO;
        typedef TVariable TV;
        typedef CConstraint<TO> TBase;

        std::vector<TO> m_objective;    // Objectives
        TV m_variable;			// Variables
	TV m_mutStep;

        CmoIndividual(void);
        ~CmoIndividual(void);
        virtual std::size_t evaluate() = 0;
        bool operator ==(const CmoIndividual<TO, TV> &individual) const;

};

template <typename TObjective, typename TVariable>
CmoIndividual<TObjective, TVariable>::CmoIndividual(void)
{
}

template <typename TObjective, typename TVariable>
CmoIndividual<TObjective, TVariable>::~CmoIndividual(void)
{
}

template <typename TObjective, typename TVariable>
bool CmoIndividual<TObjective, TVariable>::operator ==(const CmoIndividual<TO, TV> &individual) const
{

	for (std::size_t i = 0; i < individual.m_mutStep.size(); i++)
		m_mutStep[i] = individual.m_mutStep[i];
        return m_variable == individual.m_variable;
}

/*
template <typename TObjective, typename TVariable>
void CmoIndividual<TObjective, TVariable>::initMutStep(const size_t size, const TO value)
{
    	m_mutStep[i] = value
}*/
}
