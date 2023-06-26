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
#include <CLogger.h>


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
        if (component <= 0 && component >= point.size()) LOG_ERROR(errorCode::value, "Wrong component size");
        if (component == point.size() - 1)
        {
                point[component] = 1 - std::accumulate(point.begin(), --point.end(), (TType)0);
                points.push_back(point);
        }
        else
        {
                if (division <= 1) LOG_ERROR(errorCode::value, "Wrong division value");
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
std::list<std::vector<TType> > getNbi(const size_t nbObjectives, const size_t division)
{
        if (nbObjectives <= 1)	LOG_ERROR(errorCode::value, "Wrong number of objectives");

        std::list<std::vector<TType> > points;
        std::vector<TType> point(nbObjectives);

        ndi(0, division, point, points);
        
	return points;
}
}
}
}
