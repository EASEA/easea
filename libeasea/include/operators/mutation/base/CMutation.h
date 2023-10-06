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

#include <core/CmoIndividual.h>


namespace easea
{
namespace operators
{
namespace mutation
{

/*
 * \brief Base class of mutation operator
 * \param[in] TObjective - type of objectives
 * \param[in] TVariable  - type of variables

 */

template <typename TObjective, typename TVariable>
class CMutation
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CmoIndividual<TO, TV> TI;
std::size_t m_nbGen;
std::size_t m_iGen;


        CMutation(void);
        virtual ~CMutation(void);
        void operator ()(TI &individual);

	void setLimitGen(const std::size_t nbGen);
        void setCurrentGen(const std::size_t iGen);
        std::size_t getLimitGen();
        std::size_t getCurrentGen();

protected:
        virtual void runMutation(TI &individual) = 0;
};

template <typename TObjective, typename TV>
CMutation<TObjective, TV>::CMutation(void)
{
    m_nbGen = 0;
    m_iGen = 0;

}

template <typename TObjective, typename TV>
CMutation<TObjective, TV>::~CMutation(void)
{
}

template <typename TObjective, typename TVariable>
void CMutation<TObjective, TVariable>::setLimitGen(const std::size_t nbGen)
{
        m_nbGen = nbGen;
}
template <typename TObjective, typename TVariable>
std::size_t CMutation<TObjective, TVariable>::getLimitGen()
{
        return m_nbGen;
}
template <typename TObjective, typename TVariable>
void CMutation<TObjective, TVariable>::setCurrentGen(const std::size_t iGen)
{
        m_iGen = iGen;
}
template <typename TObjective, typename TVariable>
std::size_t CMutation<TObjective, TVariable>::getCurrentGen()
{
        return m_iGen;
}


template <typename TObjective, typename TV>
void CMutation<TObjective, TV>::operator ()(TI &individual)
{
        runMutation(individual);
}

// reduce compilation time and check for errors while compiling lib
extern template class CMutation<float, std::vector<float>>;
extern template class CMutation<double, std::vector<double>>;
}
}
}
