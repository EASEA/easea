/***********************************************************************
| CRandom.h                                                             |
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

#include <third_party/aixlog/aixlog.hpp>


namespace easea
{
namespace shared
{
template <typename TType>
class CProbability
{
public:
        typedef TType TT;

        CProbability(const TT probability);
        ~CProbability(void);
        TT getProbability(void) const;

private:
        TT m_probability;
};

template <typename TType>
CProbability<TType>::CProbability(const TT probability)
{
        if (probability < 0 && probability > 1)
	{
	    LOG(ERROR) << COLOR(red) << "Wrong probability value: " << probability << std::endl << COLOR(none);
	    exit(-1);
	}
	m_probability = probability;
}

template <typename TType>
CProbability<TType>::~CProbability(void)
{
}

template <typename TType>
TType CProbability<TType>::getProbability(void) const
{
	return m_probability;
}
}
}
