/***********************************************************************
| uniform.h                                                             |
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


#include <random>
#include <utility>
#include <vector>
#include <CLogger.h>


namespace easea
{
namespace variables
{
namespace continuous
{
template <typename TRandom, typename TType>
std::vector<TType> getUniform(TRandom &random, const std::vector<std::pair<TType, TType> > &boundary)
{
        std::vector<TType> individual(boundary.size());

        for (size_t i = 0; i < individual.size(); ++i)
        {
                if (boundary[i].first >= boundary[i].second)
                LOG_ERROR(errorCode::value,  "Wrong boundary values");
                std::uniform_real_distribution<TType> dist(boundary[i].first, boundary[i].second);
                individual[i] = dist(random);
        }

	        return individual;
}
    
template <typename TRandom, typename TType>
std::vector<std::vector<TType> > uniform(TRandom &random, const std::vector<std::pair<TType, TType> > &boundary, const size_t size)
{
        std::vector<std::vector<TType> > population(size);

        for (size_t i = 0; i < population.size(); ++i)
                population[i] = getUniform(random, boundary);

        return population;
}
}
}
}
