/***********************************************************************
| CConstraint.h                                                         |
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

namespace easea
{
template <typename TObjective>
class CConstraint
{
public:
        typedef TObjective TO;

        std::vector<TO> m_inequality; 
        std::vector<TO> m_equality; 

        CConstraint(void);
        ~CConstraint(void);
        bool operator ()(void) const;

};

template <typename TObjective>
CConstraint<TObjective>::CConstraint(void)
{
}

template <typename TObjective>
CConstraint<TObjective>::~CConstraint(void)
{
}


template <typename TObjective>
bool CConstraint<TObjective>::operator ()(void) const
{
        for (std::size_t i = 0; i < m_inequality.size(); ++i)
        {
                if (m_inequality[i] < 0)
                        return false;
        }
        for (std::size_t i = 0; i < m_equality.size(); ++i)
        {
                if (m_equality[i] != 0)
                        return false;
        }
        return true;
}

// reduce compilation time and check for errors while compiling lib
extern template class CConstraint<float>;
extern template class CConstraint<double>;
}
