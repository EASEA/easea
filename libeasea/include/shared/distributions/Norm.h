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
norm(const T mu_par, const T sigma_par, rand_engine_t& engine);

template<typename T>
cond_common_t<T>
norm(const T mu_par, const T sigma_par, const ulong_t seed_val = std::random_device{}());

template<typename T>
T
norm_compute(const T mu_par, const T sigma_par, rand_engine_t& engine)
{

    std::normal_distribution<T> norm_dist(T(0),T(1));

    return mu_par + sigma_par*norm_dist(engine);
}

template<typename T, typename TC = cond_common_t<T>>
TC
norm_type_check(const T mu_par, const T sigma_par, rand_engine_t& engine)
{
    return norm_compute(static_cast<TC>(mu_par),static_cast<TC>(sigma_par),engine);
}


template<typename T>
cond_common_t<T>
norm(const T mu_par, const T sigma_par, rand_engine_t& engine)
{
    return norm_type_check(mu_par,sigma_par,engine);
}

template<typename T>
cond_common_t<T>
norm(const T mu_par, const T sigma_par, const ulong_t seed_val)
{
    rand_engine_t engine(seed_val);
    return norm(mu_par,sigma_par,engine);
}
}
}
}