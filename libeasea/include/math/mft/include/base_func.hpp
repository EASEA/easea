#pragma once

template<typename T1, typename T2>
constexpr auto fmod( const T1 & x, const T2 & y ) noexcept -> common_return_t<T1,T2>;
template<typename T>
constexpr auto floor( const T & x ) noexcept -> return_t<T> ;
template<typename T>
constexpr auto trunc( const T & x ) noexcept -> return_t<T>;


namespace impl
{

/* floor implementation */
 
template<typename T>
constexpr auto floor_res( const T & x, const T & x_int ) noexcept -> bool
{
    return { (x < T(0)) && (x < x_int) };
}

template<typename T>
constexpr auto floor_impl( const T & x, const T & x_int ) noexcept -> T
{
    return { x_int - static_cast<T>(floor_res(x,x_int)) };
}

template<typename T>
constexpr auto floor_run( const T & x ) noexcept -> T
{   
    return { is_nan(x) ? LIMIT<T>::quiet_NaN() : !is_finite(x) ? x :
            LIMIT<T>::min() > abs(x) ? x : floor_impl(x, T(static_cast<lllint_t>(x))) };
}

/* trunc implementation */

template<typename T>
constexpr auto trunc_impl( const T & x ) noexcept -> T
{
    return { LIMIT<T>::digits10 < 16 ? T(static_cast<llint_t>(x)) :
    T(static_cast<lllint_t>(x)) };
}

template<typename T>
constexpr auto trunc_run( const T & x ) noexcept -> T
{
    return { is_nan(x) ? LIMIT<T>::quiet_NaN() :
            !is_finite(x) ? x :
            LIMIT<T>::min() > abs(x) ? x :
	    trunc_impl(x) };
}

/* fmod implementation*/

template<typename T> 
constexpr auto fmod_impl( const T & x, const T & y ) noexcept -> T
{
    return { is_any_nan(x, y) ? LIMIT<T>::quiet_NaN() :
            !is_all_finite(x, y) ? LIMIT<T>::quiet_NaN() : x - mft::trunc(x/y)*y };
}

template<typename T1, typename T2, typename TC = common_return_t<T1,T2>>
constexpr auto fmod_run( const T1 & x, const T2 & y ) noexcept -> TC
{
    return fmod_impl(static_cast<TC>(x),static_cast<TC>(y));
}

}



/* call of trunc function */

template<typename T>
constexpr auto trunc( const T & x ) noexcept -> return_t<T>
{
    return impl::trunc_run( static_cast<return_t<T>>(x) );
}


/* call of floor function */


template<typename T>
constexpr auto floor( const T & x ) noexcept -> return_t<T> 
{
    return impl::floor_run( static_cast<return_t<T>>(x) );
}

/* call of fmod function */
template<typename T1, typename T2>
constexpr auto fmod( const T1 & x, const T2 & y ) noexcept -> common_return_t<T1,T2>
{
    return impl::fmod_run(x,y);
}


