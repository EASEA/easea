/***********************************************************************
| CConstant.h                                                     |
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
#include <third_party/aixlog/aixlog.hpp>

#define PI 3.14159265358979323846

#define LOG_FATAL(message) { std::ostringstream what; what << message; LOG(FATAL) << COLOR(red) << what.str() << std::endl << COLOR(none); exit(1); } 

