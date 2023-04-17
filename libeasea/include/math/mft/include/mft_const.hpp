#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/math/constants/constants.hpp>

template <typename T>
constexpr auto pi() -> T { return boost::math::constants::pi<T>(); }

template <typename T>
constexpr auto pi2() -> T { return 2 * boost::math::constants::pi<T>(); }

template <typename T>
constexpr auto pi_half() -> T { return boost::math::constants::pi<T>()/2; }

template <typename T>
auto toBrad( const T & rad ) -> T { return rad * T(256) / pi2<T>() ; }

template <typename T>
auto toRad( const T & brad ) -> T { return brad * pi2<T>() / T(256) ; }