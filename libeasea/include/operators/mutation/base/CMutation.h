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

        CMutation(void);
        virtual ~CMutation(void);
        void operator ()(TI &individual);

protected:
        virtual void runMutation(TI &individual) = 0;
};

template <typename TObjective, typename TV>
CMutation<TObjective, TV>::CMutation(void)
{
}

template <typename TObjective, typename TV>
CMutation<TObjective, TV>::~CMutation(void)
{
}

template <typename TObjective, typename TV>
void CMutation<TObjective, TV>::operator ()(TI &individual)
{
        runMutation(individual);
}
}
}
}