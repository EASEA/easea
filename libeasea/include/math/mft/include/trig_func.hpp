#pragma once


template<typename T>
constexpr auto tan( const T & x ) noexcept -> return_t<T>;
template<typename T>
constexpr return_t<T> cos(const T x) noexcept;

/* sine function call with Payen and Hanek's Reduction 
 * works for very large numbers ; e.g., boost::multipreciosn
 */
template<typename T>
constexpr auto ph_sin( const T &x ) noexcept -> return_t<T>;

/* sine function call with 2PI mod Reduction 
 * works for very large numbers ; e.g., boost::multipreciosn
 */
template<typename T>
constexpr auto pi_sin( const T &x ) noexcept -> return_t<T>;


/* sine function call with BRAD mod Reduction 
 * works for very large numbers ; e.g., boost::multipreciosn
 */
template<typename T>
constexpr auto b_sin( const T &x ) noexcept -> return_t<T>;



namespace impl
{

/* tangent function implementation */

template<typename T>
constexpr auto tan_recur( const T & x, const int depth, const int max_depth ) noexcept -> T
{
    return { depth < max_depth ? T(2*depth - 1) - x/tan_recur(x,depth+1,max_depth) :
                T(2*depth - 1) };
}

template<typename T>
constexpr auto tan_impl( const T & x ) noexcept -> T
{
    return { x/tan_recur(x*x, 1, 25) };
}

template<typename T>
constexpr auto tan_start( const T & x, const int count = 0 ) noexcept -> T
{ 
    return { x > pi<T>() ? count > 1 ? LIMIT<T>::quiet_NaN() : 
                tan_start( x - pi<T>() * impl::floor_run(x/pi<T>()), count+1 ) :
                tan_impl(x) };
}

template<typename T>
constexpr auto tan_run( const T & x ) noexcept -> T
{
    return { is_nan(x) ? LIMIT<T>::quiet_NaN() :
            LIMIT<T>::min() > abs(x) ? T(0) :
            x < T(0) ? - tan_start(-x) : tan_start( x) };
}

/* cosine function through tan(x/2) */


template<typename T> 
constexpr T cos_impl(const T x) noexcept
{
    return( T(1) - x*x)/(T(1) + x*x );
}

template<typename T> 
constexpr T cos_run(const T x) noexcept
{
    return( is_nan(x) ? LIMIT<T>::quiet_NaN() :
            LIMIT<T>::min() > abs(x) ? T(1) :
            LIMIT<T>::min() > abs(x - pi_half<T>()) ? T(0) :
            LIMIT<T>::min() > abs(x + pi_half<T>()) ? T(0) :
            LIMIT<T>::min() > abs(x - pi<T>()) ? - T(1) :
            LIMIT<T>::min() > abs(x + pi<T>()) ? - T(1) :
                cos_impl( tan<T>(x/T(2)) ) );
}
/* sine function through tan(x/2) */ 

template<typename T> 
constexpr auto sin_impl( const T &x ) noexcept -> T
{
    return T(2)*x/( T(1) + x*x );
}
template<typename T> 
constexpr auto sin_impl_smpl( const T & var ) noexcept -> T
{
    T result = T(1);
    T term_i = T(1);
    T x = var;

    for(size_t i = 2; ((1.0 + term_i) != 1.0); i+=2) 
    { 
        term_i = (-term_i * (x*x)) / (i*(i+1));
        result += term_i;
    };
    return T( x * result );
}


template<typename T> 
constexpr auto sin_run( const T &x ) noexcept -> T
{
    return{ is_nan(x) ? LIMIT<T>::quiet_NaN() :
            LIMIT<T>::min() > abs(x) ?  T(0) :
            LIMIT<T>::min() > abs(x - pi_half<T>()) ? T(1) :
            LIMIT<T>::min() > abs(x + pi_half<T>()) ?  - T(1) :
            LIMIT<T>::min() > abs(x - pi<T>()) ? T(0) :
            LIMIT<T>::min() > abs(x + pi<T>()) ?  - T(0) :
		 impl::sin_impl_smpl((x) ) };
               // impl::sin_impl(tan<T>(x/T(2)) ) };
}

}

/* 
 * tan function call 
 */

template<typename T>
constexpr auto tan( const T & x ) noexcept -> return_t<T>
{
    return impl::tan_run( static_cast<return_t<T>>(x) );
}


/* 
 * cos function call 
 */

template<typename T> 
constexpr return_t<T> cos(const T x) noexcept
{
    return impl::cos_run( static_cast<return_t<T>>(x) );
}



/* 
 * sine function call with Payen and Hanek's Reduction 
 * works for very large numbers ; e.g., boost::multipreciosn
 */


template<typename T> 
constexpr auto ph_sin( const T &x ) noexcept -> return_t<T>
{
      const auto k = static_cast<std::uint64_t> (x / (pi<T>()/2));
      const auto n = static_cast<std::uint_fast64_t>(k % 4U);

      T r = x - (pi<T>()/2*k);

      const auto is_neg =  (n > 1U);
      const auto is_cos = ((n == 1U) || (n == 3U));

      return { is_cos && is_neg ? -impl::cos_run( static_cast<return_t<T>>(r) ) :
               is_cos && !is_neg ? impl::cos_run( static_cast<return_t<T>>(r) ) :
               !is_cos && is_neg ? -impl::sin_run( static_cast<return_t<T>>(r) ) :
               !is_cos && !is_neg ? impl::sin_run( static_cast<return_t<T>>(r) ) : LIMIT<T>::quiet_NaN() };
}
template<typename T> 
constexpr auto pi_sin( const T &x ) noexcept -> return_t<T>
{
        auto r = fmod<T>( x, pi2<T>() );
	return impl::sin_run( static_cast<return_t<T>>(r) );
}

template<typename T> 
constexpr auto b_sin( const T &x ) noexcept -> return_t<T>
{
	auto b = toBrad<T>( x );
        auto m = fmod<T>( b, T(256) );
	auto r = toRad<T>( m );
	return impl::sin_run( static_cast<return_t<T>>(r) );
}

