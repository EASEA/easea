/***********************************************************************
| aggregation.h                                                         |
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
#include <numeric>
#include <CLogger.h>


namespace easea
{
namespace shared
{
namespace function
{
namespace aggregation
{

 /*
  * \brief shared utility functions : 
  *
  */

template <typename TType>
TType chebuchev(const std::vector<TType> &weight, const std::vector<TType> &direction)
{
        assert(weight.size() == direction.size());

        TType maxValue = std::numeric_limits<TType>::min();
        for(size_t i = 0; i < direction.size(); ++i)
        {
                assert(direction[i] >= 0);
                const TType value = direction[i] * weight[i];
                if(value > maxValue)
                        maxValue = value;
        }
        return maxValue;
}


}
}
}
}
