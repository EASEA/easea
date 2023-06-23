/***********************************************************************
| CRandom.h 							        |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA		|
| (EAsy Specification of Evolutionary Algorithms) 			|
| https://github.com/EASEA/                                 		|
|    									|	
| Copyright (c)      							|
| ICUBE Strasbourg		                           		|
| Date: 2019-05                                                         |
|                                                                       |
 ***********************************************************************/

#pragma once

#include <type_traits>
#include <functional>
#include <random>

namespace easea
{
namespace shared
{
template <typename TRandom>
class CRandom
{
public:
        using TR = typename std::remove_reference<typename std::remove_cv<TRandom>::type>::type;

        CRandom(TR& random);
        ~CRandom(void);
        TR& getRandom(void);

private:
	std::reference_wrapper<TR> m_random;
};

template <typename TRandom>
CRandom<TRandom>::CRandom(CRandom<TRandom>::TR& random) : m_random(random)
{
}

template <typename TRandom>
CRandom<TRandom>::~CRandom(void)
{
}

template <typename TRandom>
typename CRandom<TRandom>::TR& CRandom<TRandom>::getRandom(void)
{
        return m_random;
}
}
}

