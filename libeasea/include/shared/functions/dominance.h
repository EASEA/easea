/***********************************************************************
| dominance.h 							        |
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

#include <list>
#include <CLogger.h>



namespace easea
{
namespace shared
{
namespace functions
{
/*
 * \brief shared utility functions : to check dominance
 *
 */

/*
 * \brief Check if individual is nondominated
 *
 * \param[in] 
 */

template <typename TIterator, typename TIndividual, typename TDominate>
bool isNondominated(TIterator individual, std::list<TIndividual> &population, std::list<TIndividual> &lstNondominated, TDominate dominate)
{
        for (TIterator nondominated = lstNondominated.begin(); nondominated != lstNondominated.end();)
        {
                if (dominate(*individual, *nondominated))
                {
                        typename std::list<TIndividual>::iterator move = nondominated;
                        ++nondominated;
                        population.splice(population.begin(), lstNondominated, move);
                }
                else if (dominate(*nondominated, *individual))
                        return false;
                else
                        ++nondominated;
        }
        return true;
}

template <typename TIndividual, typename TDominate>
std::list<TIndividual> getNondominated(std::list<TIndividual> &population, TDominate dominate)
{
        typedef typename std::list<TIndividual>::iterator TIterator;
        if (population.empty())		LOG_ERROR(errorCode::value,  "Population is empty");
        std::list<TIndividual> lstNondominated;

	lstNondominated.splice(lstNondominated.end(), population, population.begin());
        for (TIterator individual = population.begin(); individual != population.end();)
        {
                if (lstNondominated.empty()) LOG_ERROR(errorCode::value,"The is no nondominated solutions");
                if (isNondominated(individual, population, lstNondominated, dominate))
                {
                        typename std::list<TIndividual>::iterator move = individual;
                        ++individual;
                        lstNondominated.splice(lstNondominated.begin(), population, move);
                }
                else
                        ++individual;
        }
        return lstNondominated;
}

template <typename TObjective>
bool isDominated(const std::vector<TObjective> &point1, const std::vector<TObjective> &point2)
{
        if (point1.size() != point2.size())	LOG_ERROR(errorCode::value, "individuals have different size");
        bool dominated = false;
        
        for (size_t i = 0; i < point1.size(); ++i)
        {
                if (point2[i] < point1[i])
                        return false;        // non dominated
                if (point1[i] < point2[i])
                        dominated = true;     // dominated, the new one is better
        }
        return dominated;
}

template <typename TObjective>
bool isEqual(const std::vector<TObjective> &point1, const std::vector<TObjective> &point2)
{
	if (point1.size() != point2.size())	LOG_ERROR(errorCode::value, "individuals have different size");
	bool equal = true;

	for (size_t i = 0; i < point1.size(); ++i)
	{
	    if (point2[i] != point1[i])
		    equal = false;
	}
	return equal;
}

}
}

}