

#define TEST_PRECISION 18

#include <iomanip>
#include <iostream>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include "math/mft/include/mft.hpp"

using namespace boost::multiprecision;



int main()
{

    typedef mft::float50_noet_t ft;
    std::cout <<  std::setprecision(TEST_PRECISION);

    std::cout << "std::sin(0.1) = " << std::sin(0.1) << std::endl;

    ft hp_res = mft::ph_sin<ft>(2*pi<ft>() *1e6+0.1);
    std::cout << "HP PH sin = " << hp_res << std::endl;
    hp_res = mft::pi_sin<ft>(2*pi<ft>() *1e6+0.1);
    std::cout << "HP PI sin = " << hp_res << std::endl;
    hp_res = mft::b_sin<ft>(2*pi<ft>() *1e6+0.1);
    std::cout << "HP BRAD sin = " << hp_res << std::endl;

    typedef long double ldt;
    
    ldt mp_res = mft::ph_sin<ldt>(2*pi<ldt>() *1e6+0.1);
    std::cout << "LD PH sin(2pi*1e6+0.1) = " << mp_res << std::endl;
    mp_res = mft::pi_sin<ldt>(2*pi<ldt>() *1e6+0.1);
    std::cout << "LD PI sin(2pi*1e6+0.1) = " << mp_res << std::endl;
    mp_res = mft::b_sin<ldt>(2*pi<ldt>() *1e6+0.1);
    std::cout << "LD BRAD sin(2pi*1e6+0.1) = " << mp_res << std::endl;
    mp_res = std::sin(2*pi<ldt>()*1e6+0.1);
    std::cout << "std::sin(2pi*1e6+0.1) = " << mp_res << std::endl;


    return 0;
}
