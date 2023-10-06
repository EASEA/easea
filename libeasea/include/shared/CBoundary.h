/***********************************************************************
| CBoundary.h	                                                        |
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

#include <cstddef>
#include <vector>
#include <shared/CConstant.h>
#include <CLogger.h>


namespace easea
{
namespace shared
{
/*
 * brief Base class of boundary for optimization problem
 * param[in] TType - Ttype of desicion variable value
 */

template <typename TType>
class CBoundary
{
public:
        typedef TType T;
        typedef std::pair<T, T> TRange;
        typedef std::vector<TRange> TBoundary;

        CBoundary(const TBoundary &boundary);
        const TBoundary &getBoundary(void) const;
        bool isInside(const std::vector<TType> &point) const;
        static bool validate(const TBoundary &boundary);

private:
        TBoundary m_boundary;
};

template <typename TType>
CBoundary<TType>::CBoundary(const TBoundary &boundary)
        : m_boundary(boundary)
{
        if (!validate(boundary)) LOG_ERROR(errorCode::value, "Value of boundary is not valid");
 
}

template <typename TType>
bool CBoundary<TType>::isInside(const std::vector<TType> &point) const
{
        if (point.size() != m_boundary.size())
		LOG_ERROR(errorCode::value, "Wrong number of variables");

        for (size_t i = 0; i < point.size(); ++i)
        {
                const TType i_p = point[i];
                const auto &range = m_boundary[i];
                if (i_p < range.first)
                        return false;
                else if (i_p > range.second)
                        return false;
        }
        return true;
}
template <typename TType>
const typename CBoundary<TType>::TBoundary &CBoundary<TType>::getBoundary(void) const
{
        return m_boundary;
}

template <typename TType>
bool CBoundary<TType>::validate(const TBoundary &boundary)
{
        for (size_t i = 0; i < boundary.size(); ++i)
        {
                if (boundary[i].first > boundary[i].second)
                        return false;
        }
        return true;
}

// reduce compilation time and check for errors while compiling lib
extern template class CBoundary<double>;
extern template class CBoundary<float>;

}
}
