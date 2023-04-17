#pragma once

/* absolute value function */

template<typename T>
constexpr auto abs( const T & var ) noexcept -> T 
{
    return { var == T(0) ? T(0) : var < T(0) ? - var : var };
}

/* maximum function */


template<typename T1, typename T2>
constexpr auto max( const T1 & var1, const T2 & var2) noexcept -> common_t <T1,T2>
{
    return { var2 < var1 ? var1 : var2 };
}

/* minimum function */

template<typename T1, typename T2>
constexpr auto min( const T1 & var1, const T2 & var2 ) noexcept -> common_t <T1,T2>
{
    return { var2 > var1 ? var1 : var2 };
}


/*  sign function */

template<typename T>
constexpr auto sgn( const T & var ) noexcept -> int
{
    return { var > T(0) ?  1 : var < T(0) ? -1 : 0 };
}
