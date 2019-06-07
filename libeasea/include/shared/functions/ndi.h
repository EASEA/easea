/***********************************************************************
| ndi.h		                                                        |
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

#include <vector>
#include <list>
#include <numeric>
#include <third_party/aixlog/aixlog.hpp>


namespace easea
{
namespace shared
{
namespace function
{
 /*
  * \brief shared utility functions : Normal Boundary Intersection
  *
  */

template <typename TType>
void ndi(const size_t component, const size_t division, std::vector<TType> &point, std::list<std::vector<TType> > &points)
{
        if (component <= 0 && component >= point.size()) 
	{
		LOG(ERROR) << COLOR(red) << "Wrong component size " << component  << std::endl << COLOR(none);
		exit(-1);
	}
        if (component == point.size() - 1)
        {
                point[component] = 1 - std::accumulate(point.begin(), --point.end(), (TType)0);
                points.push_back(point);
        }
        else
        {
                if (division <= 1)
		{
			LOG(ERROR) << COLOR(red) << "Wrong division value: " << division  << std::endl << COLOR(none);
			exit(-1);
		}
                for (size_t i = 0; i <= division; ++i)
                {
                        point[component] = (TType)i / division;
                        if (std::accumulate(point.begin(), point.begin() + component + 1, (TType)0) > 1)
                                break;
                        ndi(component + 1, division, point, points);
                }
        }
}

template <typename TType>
std::list<std::vector<TType> > runNbi(const size_t nbObjectives, const size_t division)
{
        if (nbObjectives <= 1)
	{
		LOG(ERROR) << COLOR(red) << "Wrong number of objectives: " << nbObjectives << std::endl << COLOR(none);
		exit(-1);
	}

        std::list<std::vector<TType> > points;
        std::vector<TType> point(nbObjectives);

        ndi(0, division, point, points);
        
	return points;
}
}
}
}
