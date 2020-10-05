/***********************************************************************
| Norm.h
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

#include <shared/Types.h>

namespace easea{
namespace shared{
namespace distributions{

template<typename T>
cond_common_t<T>
unif(const T a_par, const T b_par, rand_engine_t& engine);

template<typename T>
cond_common_t<T>
unif(const T a_par, const T b_par, const ulong_t seed_val = std::random_device{}());

template<typename T>
T
unif_compute(const T a_par, const T b_par, rand_engine_t& engine)
{
    T a_par_adj = std::nextafter(a_par, b_par);

    std::uniform_real_distribution<T> unif_dist(a_par_adj, b_par);

    return unif_dist(engine);
}

template<typename T, typename TC = cond_common_t<T>>
TC
unif_type_check(const T a_par, const T b_par, rand_engine_t& engine)
{
    return unif_compute(static_cast<TC>(a_par), static_cast<TC>(b_par), engine);
}


template<typename T>
cond_common_t<T>
unif(const T a_par, const T b_par, rand_engine_t& engine)
{
    return unif_type_check(a_par, b_par, engine);
}

template<typename T>
cond_common_t<T>
unif(const T a_par, const T b_par, const ulong_t seed_val)
{
    rand_engine_t engine(seed_val);
    return unif(a_par, b_par, engine);
}
}
}
}