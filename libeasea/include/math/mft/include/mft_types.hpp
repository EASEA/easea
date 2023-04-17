
#include <limits>
#include <type_traits>


namespace mft
{

    using uint_t = unsigned int;
    using ullint_t = unsigned long long int;
    using llint_t = long long int;
    using lllint_t = boost::multiprecision::cpp_int;

    using ldouble_t = long double;
    using float50_t = boost::multiprecision::cpp_dec_float<50>;
    typedef boost::multiprecision::number<float50_t, boost::multiprecision::et_off> float50_noet_t;

    template<class T>
    using LIMIT = std::numeric_limits<T>;

    template<typename T>
    using return_t = typename std::conditional<std::is_integral<T>::value,double,T>::type;

    template<typename ...T>
    using common_t = typename std::common_type<T...>::type;

    template<typename ...T>
    using common_return_t = return_t<common_t<T...>>;
}
