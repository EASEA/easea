/***********************************************************************
| version.h                                                             |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2019-12                                                         |
|                                                                       |
 ***********************************************************************/
#pragma once
#include <string>
#include <sstream>
#include <config.h>

namespace easea
{
struct version
{
  static const unsigned int major = PROJECT_VER_MAJOR;
  static const unsigned int minor = PROJECT_VER_MINOR;

  static inline std::string as_string()
  {
    const char* nickname = PROJECT_VER;

    std::stringstream ss;
    ss << version::major << '.' << version::minor << " (" << nickname << ')';

    return ss.str();
  }

};
}