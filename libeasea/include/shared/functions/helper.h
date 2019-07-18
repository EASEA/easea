/***********************************************************************
| helper.h                                                              |
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

#include <CLogger.h>


namespace easea
{
namespace shared
{
namespace functions
{
namespace helper
{
template <typename TType>
TType checkBoundary(const TType value, const TType lower, const TType upper)
{
        if (lower > upper)	LOG_ERROR(errorCode::value, "Wrong boundary value");
	if (value < lower)
		return lower;
	else if (value > upper)
		return upper;
	else
		return value;
}
}
}
}
}