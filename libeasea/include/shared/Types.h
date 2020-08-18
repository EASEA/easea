/***********************************************************************
| Types.h                                                               
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
#include <random>
namespace easea{
namespace shared {

using uint_t = unsigned int;
using ulong_t = unsigned long long int;
using long_t = long long int;

using rand_engine_t = std::mt19937_64;

template<typename T>
using cond_t = typename std::conditional<std::is_integral<T>::value,double,T>::type;

template<typename ...T>
using common_t = typename std::common_type<T...>::type;

template<typename ...T>
using cond_common_t = cond_t<common_t<T...>>;

}
}