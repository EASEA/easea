/***********************************************************************
| weight.h                                                              |
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
 /*
  * \brief shared utility functions : 
  *
  */

template <typename TType>
void adjustWeight(std::vector<TType> &weight, const TType adjust)
{
        for (size_t i = 0; i < weight.size(); ++i)
                weight[i] = weight[i] == 0 ? adjust : weight[i];
}
}
}
}

