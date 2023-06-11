/***********************************************************************
| heighbors.h		                                                |
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
#include <cassert>
#include <numeric>
#include <shared/CMatrix.h>

namespace easea
{
namespace shared
{
namespace function
{
 /*
  * \brief shared utility functions : 
  *
  */

template <typename TType, typename TIter>
CMatrix<TType> calcAdjacencyMatrix(TIter begin, TIter end)
{
        const size_t count = std::distance(begin, end);
        CMatrix<TType> adMatrix(count, count);
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < count; ++j)
                adMatrix[i][j] = 0;

        size_t i = 0;

        for (TIter point1 = begin; point1 != end; ++point1)
        {
                size_t j = 0;
        
                for (TIter point2 = ++TIter(point1); point2 != end; ++point2)
                {
                        if (j != i){
                    	    const std::vector<TType> &tmp_point1 = (*point1);
                    	    const std::vector<TType> &tmp_point2 = (*point2);
                    	    adMatrix[i][j] = sqrt(std::inner_product(tmp_point1.begin(), tmp_point1.end(), tmp_point2.begin(), (TType)0, std::plus<TType>()
                                , [](TType x, TType y)->TType{TType t = x - y;return t * t;}));
                        }
                        ++j;
                }
		++i;
        }
        return adMatrix;
                  


}

template <typename TType>
std::vector<std::vector<size_t> > initNeighbors(const CMatrix<TType> &adjacencyMatrix, const size_t nNeighbors)
{
	assert(adjacencyMatrix.Rows() == adjacencyMatrix.Cols()); 
        std::vector<std::vector<size_t> > neighbors(adjacencyMatrix.Rows());

        for (size_t i = 0; i < neighbors.size(); ++i)
        {
		std::vector<TType>  aDistances(adjacencyMatrix[i]);
                typedef std::pair<TType, size_t> TA;
                std::vector<TA> adjacency(aDistances.size()); //adjacencyMatrix.Cols());
                std::vector<const TA *> pAdjacency(aDistances.size());//adjacencyMatrix.Cols());
                for (size_t j = 0; j < /*adjacencyMatrix.Cols()*/aDistances.size(); ++j)
                {
		        adjacency[j].first = aDistances[j];//adjacencyMatrix[i][j];
                        adjacency[j].second = j;
                        pAdjacency[j] = &adjacency[j];
                }
                std::partial_sort(pAdjacency.begin(), pAdjacency.begin() + nNeighbors, pAdjacency.end()
                        , [](const TA *adjacency1, const TA *adjacency2)->bool{return adjacency1->first < adjacency2->first;}
                );
                std::vector<size_t> &pNeighbors = neighbors[i];
                pNeighbors.resize(nNeighbors);
                for (size_t j = 0; j < nNeighbors; ++j)
                        pNeighbors[j] = pAdjacency[j]->second;
			assert(std::find(pNeighbors.begin(), pNeighbors.end(), i) != pNeighbors.end());
        }
        return neighbors;
}
template <typename TType>
std::vector<TType> computeDirection(const std::vector<TType> &obj, const std::vector<TType> &ideal)
{
        assert(obj.size() == ideal.size());

        std::vector<TType> dir(obj.size());

        for (size_t i = 0; i < dir.size(); ++i)
        {
                dir[i] = obj[i] - ideal[i];
                assert(dir[i] >= 0);
        }
        return dir;
}


}
}
}
