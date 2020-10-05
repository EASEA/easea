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
#include <math.h>
#include <shared/distributions/Unif.h>
#include <shared/Types.h>
#define PI_L 3.1415926535897932384626433832795028841972L

namespace easea{
namespace shared{
namespace distributions{

template<typename T>
cond_common_t<T>
qcauchy(const T p, const T mu_par, const T sigma_par);

template<typename T>
cond_common_t<T>
cauchy(const T mu_par, const T sigma_par, rand_engine_t& engine);

template<typename T>
cond_common_t<T>
cauchy(const T mu_par, const T sigma_par, const ulong_t seed_val = std::random_device{}());

template<typename T>
T
qcauchy_compute(const T p, const T mu_par, const T sigma_par)
{
    return( mu_par + sigma_par*tan(PI_L*(p - T(0.5))) );
}


template<typename T, typename TC = cond_common_t<T>>
TC
qcauchy_type_check(const T p, const T mu_par, const T sigma_par)
{
    return qcauchy_compute(static_cast<TC>(p),static_cast<TC>(mu_par),static_cast<TC>(sigma_par));

}

template<typename T>
cond_common_t<T>
qcauchy(const T p, const T mu_par, const T sigma_par)
{
    return qcauchy_type_check(p, mu_par, sigma_par);
}

template<typename T>
T
cauchy_compute(const T mu_par, const T sigma_par, rand_engine_t& engine)
{

    return qcauchy(unif(T(0), T(1), engine), mu_par, sigma_par);
}

template<typename T, typename TC = cond_common_t<T>>
TC
cauchy_type_check(const T mu_par, const T sigma_par, rand_engine_t& engine)
{
    return cauchy_compute(static_cast<TC>(mu_par), static_cast<TC>(sigma_par), engine);
}


template<typename T>
cond_common_t<T>
cauchy(const T mu_par, const T sigma_par, rand_engine_t& engine)
{
    return cauchy_type_check(mu_par, sigma_par, engine);
}

template<typename T>
cond_common_t<T>
cauchy(const T mu_par, const T sigma_par, const ulong_t seed_val)
{
    rand_engine_t engine(seed_val);
    return cauchy(mu_par, sigma_par,engine);
}
}
}
}