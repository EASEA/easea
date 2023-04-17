#pragma once

namespace impl{

template<typename T>
constexpr auto is_nan( const T & x ) noexcept -> bool
{
    return x != x;
}

template<typename T1, typename T2>
constexpr auto is_any_nan( const T1 & x, const T2 & y ) noexcept -> bool
{
    return { is_nan(x) || is_nan(y) };
}

template<typename T>
constexpr auto is_neginf( const T & x ) noexcept -> bool
{
    return x == - LIMIT<T>::infinity();
}

template<typename T>
constexpr auto is_posinf( const T & x ) noexcept -> bool
{
    return x == LIMIT<T>::infinity();
}

template<typename T>
constexpr auto is_inf( const T & x ) noexcept -> bool
{
    return { is_neginf(x) || is_posinf(x) };
}

template<typename T>
constexpr auto is_finite( const T & x ) noexcept -> bool
{
    return { !is_nan(x) && !is_inf(x) };
}

template<typename T1, typename T2>
constexpr auto is_all_finite( const T1 & x, const T2 & y ) noexcept -> bool
{
    return { is_finite(x) && is_finite(y) };
}

constexpr auto is_odd( const llint_t & x ) noexcept -> bool
{
    return (x & 1U) != 0;
}

}

