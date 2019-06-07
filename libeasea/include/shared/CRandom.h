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

namespace easea
{
namespace shared
{
template <typename TRandom>
class CRandom
{
public:
        typedef TRandom TR;

        CRandom(TR random);
        ~CRandom(void);
        TR getRandom(void);

private:
        TR m_random;
};

template <typename TRandom>
CRandom<TRandom>::CRandom(TRandom random) : m_random(random)
{
}

template <typename TRandom>
CRandom<TRandom>::~CRandom(void)
{
}

template <typename TRandom>
TRandom CRandom<TRandom>::getRandom(void)
{
        return m_random;
}
}
}

