/***********************************************************************
| CWrapCrossover.h                                                      |
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

#include <operators/mutation/base/CMutation.h>

namespace easea
{
namespace operators
{
namespace mutation
{
template <typename TObjective, typename TVariable>
class CWrapMutation
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CMutation<TO, TV> TM;

        CWrapMutation(TM &mutation);
        ~CWrapMutation(void);
        TM &getMutation(void) const;

private:
        TM &m_mutation;
};

template <typename TObjective, typename TVariable>
CWrapMutation<TObjective, TVariable>::CWrapMutation(TM &mutation) : m_mutation(mutation)
{
}

template <typename TObjective, typename TVariable>
CWrapMutation<TObjective, TVariable>::~CWrapMutation(void)
{
}

template <typename TObjective, typename TVariable>
typename CWrapMutation<TObjective, TVariable>::TM &CWrapMutation<TObjective, TVariable>::getMutation(void) const
{
        return m_mutation;
}
}
}
}