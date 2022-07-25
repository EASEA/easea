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
  static constexpr unsigned int major = PROJECT_VERSION_MAJOR;
  static constexpr unsigned int minor = PROJECT_VERSION_MINOR;
  static constexpr unsigned int patch = PROJECT_VERSION_PATCH;
  static constexpr const char* full = PROJECT_VERSION;

  static inline std::string as_string()
  {
    //const char* nickname = PROJECT_VERSION;
    //std::stringstream ss;
    //ss << version::major << '.' << version::minor << " (" << nickname << ')';
    //return ss.str();
    return full;
  }

};
}
